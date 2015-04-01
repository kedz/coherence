import theano
import theano.tensor as T
import numpy as np
from itertools import izip
import cPickle as pickle
import os
import datetime
from sklearn.base import BaseEstimator

class NNModel(BaseEstimator):
    def __init__(self, embeddings=None, max_sent_len=150,  
                 hidden_dim=100, window_size=3, learning_rate=.01, 
                 max_iters=50, batch_size=25, lam=1.25, fit_init=True,
                 fit_embeddings=False, fit_callback=None, 
                 update_method="adagrad"):
        """
        Base class for Neural Network based coherence models.
        All neural nets implemented in this module follow the same basic
        feedforward architecture. Subclasses of this class differ only in
        the way they compute the hiden sentence layers.
        The feedforward network has a simple 2 layer architecture.
        Given a window of n sentences this model computes
                h1 = tanh(W1_s1 * s1 + ... W1_sn * sn + b1)
                p(y|s1, s2, ..., sn) = sig(W2*h1 + b2)
        where y is a binary variable (0 = incoherent, 1 = coherent) and the
        s1 ... sn are the sentence layers.
        
        params
        ------
        embeddings -- a V x D word embedding matrix where V is the vocab size
            and D is the word embedding dimension
        max_sent_len -- int, maximum sentence length of input sentences,
            sentences shorter than this should be padded with a __PAD__ token.
            Each row in input matrix should have (window_size * max_sent_len)
            columns.
        hidden dim -- int, dimension of hidden layer h1
        window_size -- int, window size, must be at least 3 and an odd number
        learning_rate -- float, learning rate for gradient descent or adagrad
                                (alpha in the latter case)
        lam -- float, weight of regularization 
        batch_size, int -- maximum batch size for minibatch sgd
        max_iters -- int, maximum number of training iterations
        fit_init -- boolean, default True, only relevant for rnn sentence model
            where initial hidden recurrent layer is also learned.
        fit_embed -- bolean, default False, update/learn word embeddings.
        fit_callback -- callable, a function 
            callback(nnet, avg_win_err, n_iter) that is called after every
            training epoch. nnet is the nnet object (aka self), avg_win_err
            is the current avg training error and n_iter is the current 
            iteration number, starting at 1. This is a utility method
            ment to be used for possibly displaying current model accuracy
            on held out data or saving each iteration of the model.
        update -- str [sgd|adagrad], update method to use. sgd is the standard
            gradient descent update. adagrad is the diagonal adagrad update.
        """                  
        assert update_method in ["sgd", "adagrad"]

        self.embeddings = embeddings
        self.max_sent_len = max_sent_len
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.lam = lam
        self.fit_init = fit_init
        self.fit_embeddings = fit_embeddings
        self.fit_callback = fit_callback
        self.update_method = update_method

        self.sym_vars = {u"X_iw": T.imatrix(name=u"X_iw"),
                         u"y": T.ivector(name=u"y"),
                         u"M_iw": T.imatrix(name=u"M_iw")} 

        if embeddings is not None:
            self.init_params(self.embeddings)
            self.build_network(self.max_sent_len, self.window_size)

    def build_network(self, max_sent_len, win_size):
        
        # _funcs will contains compiled theano functions of several
        # variables for debugging purposes
        self._funcs = {}
        
        # _hidden_layers will contain theano Variables of h_s1, ... , h_sn 
        # and h1 
        self._hidden_layers = {}
        
        X_iw = self.sym_vars[u"X_iw"]
        E = self.params[u"E"]    
        M_iw = self.sym_vars[u"M_iw"]
        y = self.sym_vars[u"y"]
                
        H_s = self._build_sentence_layer(
            X_iw, M_iw, y, E, max_sent_len, win_size)

        self._funcs["H_s"] = theano.function([X_iw, M_iw], H_s)
        for i in xrange(win_size):
            self._hidden_layers["h_s{}".format(i)] = H_s[i]

        # Compute the first hidden layer (this is the concatenation of all
        # hidden sentence vectors).
        A1 = self.params["b1"]
        for i in xrange(win_size):
            W1_si = self.params["W1_s{}".format(i)]
            A1 += T.dot(H_s[i], W1_si)
        h1 = T.tanh(A1)
        self._hidden_layers["h1"] = h1
        
        self._nn_output = T.nnet.softmax(
            T.dot(h1, self.params["W2"]) + self.params["b2"])
        self._nn_output_func = theano.function([X_iw, M_iw], self._nn_output)

        self._nll = -T.mean(
            T.log(self._nn_output)[T.arange(y.shape[0]), y])      
        

        y_pred = T.argmax(self._nn_output, axis=1)
        self._funcs["avg win error"] = theano.function(
            [X_iw, M_iw, y], T.mean(T.neq(y, y_pred)))

        
    def _mask(self, X_iw):
        M_iw = np.ones_like(X_iw)
        M_iw[X_iw == self.embeddings.get_index("__PAD__")] = 0
        return M_iw

    def avg_win_err(self, X_iw, y):
        M_iw = self._mask(X_iw)
        return float(self._funcs["avg win error"](X_iw, M_iw, y))

    def predict_prob_windows(self, X_iw):
        M_iw = self._mask(X_iw)
        return self._nn_output_func(X_iw, M_iw)

    def log_prob_is_coherent(self, IX):
        return np.log(self.predict_prob_windows(IX)[:,1]).sum()
    
    def score(self, X_gold, X_perm):
        correct = 0
        total = len(X_gold)
        for IX_gold, IX_perm in izip(X_gold, X_perm):
            gold_lp = self.log_prob_is_coherent(IX_gold)
            perm_lp = self.log_prob_is_coherent(IX_perm)
            if gold_lp > perm_lp:
                correct += 1
        return correct / float(total)

    def fit(self, X_iw_train, y_train, X_iw_dev=None, y_dev=None):

        n_batches = X_iw_train.shape[0] / self.batch_size
        M_iw_train = self._mask(X_iw_train)

        X_iw_shared = theano.shared(X_iw_train.astype(np.int32),
            name="X_iw_shared",
            borrow=True)
        M_iw_shared = theano.shared(M_iw_train.astype(np.int32),
            name="M_iw_shared",
            borrow=True)
        y_shared = theano.shared(y_train.astype(np.int32),
            name="y_shared",
            borrow=True)
        
        if X_iw_dev is not None and y_dev is not None:
            M_iw_dev = self._mask(X_iw_dev)

            X_iw_dev_shared = theano.shared(X_iw_dev.astype(np.int32),
                name="X_iw_dev_shared",
                borrow=True)
            M_iw_dev_shared = theano.shared(M_iw_dev.astype(np.int32),
                name="M_iw_dev_shared",
                borrow=True)
            y_dev_shared = theano.shared(y_dev.astype(np.int32),
                name="y_dev_shared",
                borrow=True)
 
        X_iw = self.sym_vars["X_iw"]
        M_iw = self.sym_vars["M_iw"]
        y = self.sym_vars["y"]
        
        reg = 0
        for param in self._reg_params:
            reg += (param**2).sum() 

        cost = self._nll + self.lam * reg / (2. * X_iw.shape[0])
        gtheta = T.grad(cost, self.theta)

        if self.update_method == "adagrad":
            # diagonal adagrad update
            grad_hist_update = self.grad_hist + gtheta**2
            grad_hist_upd_safe = T.switch(
                T.eq(grad_hist_update, 0), 1, grad_hist_update)
            theta_update = self.theta - \
                    self.learning_rate / T.sqrt(grad_hist_upd_safe) * gtheta
            updates = [(self.grad_hist, grad_hist_update),
                       (self.theta, theta_update)]

        elif self.update_method == "sgd":
            updates = [(self.theta, self.theta - self.learning_rate * gtheta)]
        else:
            raise Exception("'{}' is not an implemented update method".format(
                self.update_method))
        
        index = T.scalar(dtype='int32')  # index to a [mini]batch
        train_model = theano.function(
            inputs=[index],
            outputs=[cost],
            updates=updates,
            givens={
                X_iw: X_iw_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
                y: y_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
                M_iw: M_iw_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
            }
        )
        

        for n_iter in xrange(1, self.max_iters + 1):
            for i in xrange(n_batches):
                train_model(i)
            avg_win_err = self.avg_win_err(X_iw_train, y_train)
            if self.fit_callback is None:
                print "iter", n_iter, " | avg win err: {:0.3f}".format(
                    avg_win_err) 
            else:
               self.fit_callback(self, avg_win_err, n_iter)            


    def _get_params(self):
        return {key: val
                for key, val in self.get_params().items()
                if key != "fit_callback"} #and val is not None:
        
        
    def __getstate__(self):
        """ Return state sequence."""
        params = self._get_params() 
        theta = self.theta.get_value() 
        state = (params, theta)
        return state

    def __setstate__(self, state):
        """ Set parameters from state sequence."""
        params, theta = state
        self.set_params(**params)
        self.init_params(self.embeddings)
        self.build_network(self.max_sent_len, self.window_size)
        self.theta.set_value(theta)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)
        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)
        fabspath = os.path.join(fpath, fname)
        print("Saving to %s ..." % fabspath)
        with open(fabspath, 'wb') as f:
            state = self.__getstate__()
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self, path):
        """ Load model parameters from path. """
        print("Loading from %s ..." % path)
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.__setstate__(state)


class CBOWModel(NNModel):
    def init_params(self, embeddings):
        word_dim = embeddings.embed_dim
        self.params = {}
        self._reg_params = []
        
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2

        n_feedforward_params = self.hidden_dim * word_dim * \
            self.window_size + self.hidden_dim + 2 * self.hidden_dim + 2
 
        print "n_feedforward_params:", n_feedforward_params
        
        n_params = n_feedforward_params
        
        print "n_params:", n_params
        
        self.theta = theano.shared(
            value=np.zeros(n_params,
                dtype=theano.config.floatX),
            name="theta", borrow=True)
        
        current_pointer = 0
        inits = []

        # Initialize Feed-Forward NN component

        eps = np.sqrt(6. / (self.hidden_dim + word_dim * self.window_size))
        
        # Init W1_s...
        for i in xrange(self.window_size):
            next_pointer = current_pointer + self.hidden_dim * word_dim
            W1_s = self.theta[current_pointer:next_pointer].reshape(
                (word_dim, self.hidden_dim))
            W1_s.name="W1_s{}".format(i)
            self.params[W1_s.name] = W1_s
            self._reg_params.append(W1_s)            

            W1_s_init = np.random.uniform(
                low=-eps, high=eps, 
                size=(word_dim, self.hidden_dim)).astype(
                    theano.config.floatX)
            inits.append(W1_s_init)
            current_pointer = next_pointer
            
        # Init b1 (hidden_dim)
        next_pointer = current_pointer + self.hidden_dim    
        b1 = self.theta[current_pointer:next_pointer].reshape(
                (self.hidden_dim,))
        b1.name = "b1"
        self.params[b1.name] = b1
            
        b1_init = np.zeros(self.hidden_dim, dtype=theano.config.floatX)
        inits.append(b1_init)
        current_pointer = next_pointer
    
        # Init W2 (hidden_dim x 2)
        next_pointer = current_pointer + self.hidden_dim * 2
        W2 = self.theta[current_pointer:next_pointer].reshape(
            (self.hidden_dim, 2))
        W2.name="W2"
        self.params[W2.name] = W2
        self._reg_params.append(W2)            
        
        W2_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(self.hidden_dim, 2)).astype(
                theano.config.floatX)
        inits.append(W2_init)
        current_pointer = next_pointer
        
        # Init b2 (2)
        next_pointer = current_pointer + 2    
        b2 = self.theta[current_pointer:next_pointer].reshape(
                (2,))
        b2.name = "b2"
        self.params[b2.name] = b2
            
        b2_init = np.zeros(2, dtype=theano.config.floatX)
        inits.append(b2_init)
        current_pointer = next_pointer
    
        self.theta.set_value(np.concatenate([x.ravel() for x in inits]))
        
        E = theano.shared(embeddings.W.astype(theano.config.floatX),
            name="E", borrow=True)
        self.params[E.name] = E

        if self.update_method == "adagrad":
            self.grad_hist = theano.shared(
                np.zeros_like(self.theta.get_value(borrow=True)),
                name="grad_hist",
                borrow=True)

    def _build_sentence_layer(self, X_iw, M_iw, y, E, max_sent_len, win_size): 

        # Index into word embedding matrix and construct a 
        # 3d tensor for input. After dimshuffle, dimensions should
        # be (MAX_LEN * window_size, batch_size, word_dim).
        X = E[X_iw] 
        X = X.dimshuffle(1,0,2)
       
        self._funcs["X"] = theano.function([X_iw], X)
        
        M_T = M_iw.dimshuffle(1,0).reshape((X.shape[0], X.shape[1], 1))
 
        self._funcs["M_T"] = theano.function(
            [X_iw, M_iw], M_T) 

        X_s_sums = [X[i * max_sent_len:(i + 1) * max_sent_len].sum(axis=0)
                    for i in xrange(win_size)]
        M_s_sums = [M_T[i * max_sent_len:(i + 1) * max_sent_len].sum(axis=0)
                    for i in xrange(win_size)]
        
        self._funcs["X_s_sums"] = theano.function([X_iw], X_s_sums)
        self._funcs["M_s_sums"] = theano.function([X_iw, M_iw], M_s_sums)
        
        X_s_avgs = [X_si_sums / M_si_sums
                     for X_si_sums, M_si_sums in izip(X_s_sums, M_s_sums)]

        return X_s_avgs



class RNNModel(NNModel):
        
    def init_params(self, embeddings):    

        word_dim = embeddings.embed_dim

        # params is a dict mapping variable name to theano Variable objects.
        self.params = {}
        
        # _reg_params is a list of theano Variables that will be subject
        # to regularization.
        self._reg_params = []
        
        # Number of params for W_rec + V_rec + b_rec
        n_recurrent_params = word_dim**2 + word_dim**2 + \
            word_dim
        
        # If we are fitting h_0, allocate word_dim params for this 
        # variable.
        if self.fit_init is True:
            n_recurrent_params += word_dim
        
        print "n_recurrent_params:", n_recurrent_params
        
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2
        n_feedforward_params = self.hidden_dim * word_dim * \
            self.window_size + self.hidden_dim + 2 * self.hidden_dim + 2
        
        print "n_feedforward_params:", n_feedforward_params
        
        n_params = n_recurrent_params + n_feedforward_params
        
        print "n_params:", n_params
        self.n_params = n_params
        
        # Allocate memory for all params in one block.
        self.theta = theano.shared(
            value=np.zeros(n_params,
                dtype=theano.config.floatX),
            name="theta", borrow=True)
        
        # For each parameter we assign it a slice of memory from theta,
        #     i.e. P = theta[current_pointer:next_pointer]
        # Initilization values for P are collected in inits
        # and variable name is mapped in params (i.e. params[P.name] = P).
        # Optionally P is added to _reg_params if it will be regularized.
        current_pointer = 0
        inits = []
         
        # Initialize RNN component
        
        # Init W_rec (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        W_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        W_rec.name = "W_rec"
        self.params[W_rec.name] = W_rec
        self._reg_params.append(W_rec)
        
        W_rec_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(word_dim, word_dim)).astype(
                theano.config.floatX)
        inits.append(W_rec_init)
        current_pointer = next_pointer
        
        # Init V_rec (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        V_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        V_rec.name = "V_rec"
        self.params[V_rec.name] = V_rec
        
        V_rec_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(word_dim, word_dim)).astype(
                theano.config.floatX)
        inits.append(V_rec_init)
        current_pointer = next_pointer
        
        # Init b_rec (word_dim)
        next_pointer = current_pointer + word_dim
        b_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim,))
        b_rec.name = "b_rec"
        self.params[b_rec.name] = b_rec
        
        b_rec_init = np.zeros(word_dim, dtype=theano.config.floatX)
        inits.append(b_rec_init)
        current_pointer = next_pointer       
        
        if self.fit_init is True:
            next_pointer = current_pointer + word_dim
            h0 = self.theta[current_pointer:next_pointer].reshape(
                (word_dim,))
            h0.name = "h0"
            self.params[h0.name] = h0   
            
            h0_init = np.random.uniform(
                low=-.2, high=.2,
                size=(word_dim,)).astype(theano.config.floatX)
            inits.append(h0_init)
            current_pointer = next_pointer 
        else:
            h0 = theano.shared(
                value=np.zeros(word_dim, dtype=theano.config.floatX),
                name="h0", 
                borrow=True)
            self.params[h0.name] = h0
        
        # Initialize Feed-Forward NN component
        
        eps = np.sqrt(6. / (self.hidden_dim + word_dim * self.window_size))

        # Init W1_s...
        for i in xrange(self.window_size):
            next_pointer = current_pointer + self.hidden_dim * word_dim
            W1_s = self.theta[current_pointer:next_pointer].reshape(
                (word_dim, self.hidden_dim))
            W1_s.name="W1_s{}".format(i)
            self.params[W1_s.name] = W1_s
            
            W1_s_init = np.random.uniform(
                low=-eps, high=eps, 
                size=(word_dim, self.hidden_dim)).astype(
                    theano.config.floatX)
            inits.append(W1_s_init)
            current_pointer = next_pointer
        
        # Init b1 (hidden_dim)
        next_pointer = current_pointer + self.hidden_dim    
        b1 = self.theta[current_pointer:next_pointer].reshape(
                (self.hidden_dim,))
        b1.name = "b1"
        self.params[b1.name] = b1
            
        b1_init = np.zeros(self.hidden_dim, dtype=theano.config.floatX)
        inits.append(b1_init)
        current_pointer = next_pointer
    
        # Init W2 (hidden_dim x 2)
        next_pointer = current_pointer + self.hidden_dim * 2
        W2 = self.theta[current_pointer:next_pointer].reshape(
            (self.hidden_dim, 2))
        W2.name="W2"
        self.params[W2.name] = W2
        
        W2_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(self.hidden_dim, 2)).astype(
                theano.config.floatX)
        inits.append(W2_init)
        current_pointer = next_pointer
        
        # Init b2 (2)
        next_pointer = current_pointer + 2    
        b2 = self.theta[current_pointer:next_pointer].reshape(
                (2,))
        b2.name = "b2"
        self.params[b2.name] = b2
            
        b2_init = np.zeros(2, dtype=theano.config.floatX)
        inits.append(b2_init)
        current_pointer = next_pointer
    
        self.theta.set_value(np.concatenate([x.ravel() for x in inits]))
       
        # TODO: implement learnable embeddings
        E = theano.shared(embeddings.W.astype(theano.config.floatX),
            name="E", borrow=True)
        self.params[E.name] = E

        # If using adagrad, we need to keep a running sum of historical squared
        # gradients for each parameter.
        if self.update_method == "adagrad":
            self.grad_hist = theano.shared(
                np.zeros_like(self.theta.get_value(borrow=True)),
                name="grad_hist",
                borrow=True)

    def _build_sentence_layer(self, X_iw, M_iw, y, E, max_sent_len, win_size):
 
        # Index into word embedding matrix and construct a 
        # 3d tensor for input. After dimshuffle, dimensions should
        # be (MAX_LEN * window_size, batch_size, word_dim).
        X = E[X_iw] 
        X = X.dimshuffle(1,0,2)
        
        self._funcs["X"] = theano.function([X_iw], X)  
        
        M_T = M_iw.dimshuffle(1,0).reshape((X.shape[0], X.shape[1], 1))
 
        self._funcs["M_T"] = theano.function(
            [X_iw, M_iw], M_T) 

        h0_block = T.alloc(self.params["h0"], X.shape[1], X.shape[2])

        H_s = [] 
        for i in xrange(win_size):
            h_si, _ = theano.scan(
                fn=self.step_mask,
                sequences=[
                    {"input": X,   "taps":[i * max_sent_len]}, 
                    {"input": M_T, "taps":[i * max_sent_len]},], 
                outputs_info=h0_block,
                n_steps=max_sent_len)
            h_si = h_si[-1]
            H_s.append(h_si)
            self._hidden_layers["h_s{}".format(i)] = h_si

        return H_s
 
    def step_mask(self, x_t, m_t, h_tm1):
        h_t_unmasked = self.step(x_t, h_tm1)
        h_t = m_t * h_t_unmasked + (1 - m_t) * h_tm1
        return h_t
    
    def step(self, x_t, h_tm1):
        W_rec = self.params["W_rec"]
        V_rec = self.params["V_rec"]
        b_rec = self.params["b_rec"]
        h_t = T.tanh(T.dot(h_tm1, V_rec) + T.dot(x_t, W_rec) + b_rec)
        return h_t
