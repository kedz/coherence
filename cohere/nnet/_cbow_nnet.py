import theano
import theano.tensor as T
import numpy as np
from cohere.nnet._base import _BaseNNModel
from itertools import izip

class CBOWModel(_BaseNNModel):

    def __init__(self, embeddings, max_sent_len=150,
                 window_size=3, hidden_dim=100, alpha=.01, lam=1.25,
                 batch_size=25, max_iters=10, 
                 fit_embeddings=False, fit_callback=None, 
                 update_method="adagrad", initializers=None):
         
        self.sym_vars = {u"X_iw": T.imatrix(name=u"X_iw"),
                         u"y": T.ivector(name=u"y"),
                         u"M_iw": T.imatrix(name=u"M_iw")} 
 
        super(CBOWModel, self).__init__(
            embeddings, max_sent_len=max_sent_len, hidden_dim=hidden_dim,
            window_size=window_size, alpha=alpha, lam=lam,
            batch_size=batch_size, fit_embeddings=fit_embeddings, 
            fit_callback=fit_callback, update_method=update_method,
            initializers=initializers, max_iters=max_iters)
 
    def _init_params(self):

        embeddings = self.embeddings
        initializers = self.initializers

        if initializers is None:
            initializers = dict()

        word_dim = embeddings.embed_dim

        # params is a dict mapping variable name to theano Variable objects.
        self.params = {}

        # _reg_params is a list of theano Variables that will be subject
        # to regularization in the cost function.
        self._reg_params = []
        
        # Number of params for pads.
        n_params = self.window_size / 2 * word_dim * 2
 
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2
        n_params += self.hidden_dim * word_dim * self.window_size + \
            self.hidden_dim + 1 * self.hidden_dim + 1
 
        print "n_params:", n_params
        
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

        # Initialize Feed-Forward NN component
        eps = np.sqrt(6. / (self.hidden_dim + word_dim * self.window_size))
        
        # Init W1_s...
        for i in xrange(self.window_size):
            name = "W1_s{}".format(i)
            next_pointer = current_pointer + self.hidden_dim * word_dim
            W1_s = self.theta[current_pointer:next_pointer].reshape(
                (word_dim, self.hidden_dim))
            W1_s.name = name
            self.params[name] = W1_s
            self._reg_params.append(W1_s)            

            if initializers.get(name, None) is None:
                W1_s_init = np.random.uniform(
                    low=-eps, high=eps, 
                    size=(word_dim, self.hidden_dim)).astype(
                        theano.config.floatX)
            else:
                W1_s_init = initializers.get(name) 
            inits.append(W1_s_init)
            current_pointer = next_pointer
            
        # Init b1 (hidden_dim)
        next_pointer = current_pointer + self.hidden_dim    
        b1 = self.theta[current_pointer:next_pointer].reshape(
                (self.hidden_dim,))
        b1.name = "b1"
        self.params[b1.name] = b1

        if initializers.get("b1", None) is None:
            b1_init = np.zeros(self.hidden_dim, dtype=theano.config.floatX)
        else:
            b1_init = initializers.get("b1")
        inits.append(b1_init)
        current_pointer = next_pointer

        # Init W2 (hidden_dim x 1)
        next_pointer = current_pointer + self.hidden_dim * 1
        W2 = self.theta[current_pointer:next_pointer].reshape(
            (self.hidden_dim, 1))
        W2.name="W2"
        self.params[W2.name] = W2
        self._reg_params.append(W2)            

        if initializers.get("W2", None) is None:
            W2_init = np.random.uniform(
                low=-.25, high=.25, 
                size=(self.hidden_dim, 1)).astype(
                    theano.config.floatX)
        else:
            W2_init = initializers.get("W2")
        inits.append(W2_init)
        current_pointer = next_pointer
        
        # Init b2 (2)
        next_pointer = current_pointer + 1
        b2 = self.theta[current_pointer:next_pointer].reshape(
                (1,))
        b2.name = "b2"
        self.params[b2.name] = b2

        if initializers.get("b2", None) is None:
            b2_init = np.zeros(1, dtype=theano.config.floatX)
        else:
            b2_init = initializers.get("b2")
        inits.append(b2_init)
        current_pointer = next_pointer

        # W_start (window_size / 2, word_dim)
        next_pointer = current_pointer + self.window_size / 2 * word_dim    
        W_start = self.theta[current_pointer:next_pointer].reshape(
                (self.window_size / 2, word_dim))
        W_start.name = "W_start"
        self.params[W_start.name] = W_start
            
        if initializers.get("W_start", None) is None:
            W_start_init = np.zeros((self.window_size / 2, word_dim), 
                                    dtype=theano.config.floatX)
        else:
            W_start_init = initializers.get("W_start")
        inits.append(W_start_init)
        current_pointer = next_pointer

        # W_stop (window_size / 2, word_dim)
        next_pointer = current_pointer + self.window_size / 2 * word_dim    
        W_stop = self.theta[current_pointer:next_pointer].reshape(
                (self.window_size / 2, word_dim))
        W_stop.name = "W_stop"
        self.params[W_stop.name] = W_stop
            
        if initializers.get("W_stop", None) is None:
            W_stop_init = np.zeros((self.window_size / 2, word_dim), 
                                   dtype=theano.config.floatX)
        else:
            W_stop_init = initializers.get("W_stop")
        inits.append(W_stop_init)
        current_pointer = next_pointer

        if self.fit_embeddings is True:
            next_pointer = current_pointer + n_words * word_dim
            E = self.theta[current_pointer:next_pointer].reshape(
                embeddings.W.shape)
            E.name = "E"
            self.params[E.name] = E
            
            inits.append(embeddings.W)
            current_pointer = next_pointer
            
        else:
            E = theano.shared(embeddings.W.astype(theano.config.floatX),
                name="E", borrow=True)
            self.params[E.name] = E

        assert next_pointer == self.theta.get_value(borrow=True).shape[0]     
        self.theta.set_value(np.concatenate([x.ravel() for x in inits]))

        # If using adagrad, we need to keep a running sum of historical
        # squared gradients for each parameter.
        if self.update_method == "adagrad":
            self.grad_hist = theano.shared(
                np.zeros_like(self.theta.get_value(borrow=True)),
                name="grad_hist",
                borrow=True)
    
    def _build_network(self):
        max_sent_len = self.max_sent_len
        win_size =  self.window_size
        
        # _funcs will contains compiled theano functions of several
        # variables for debugging purposes
        self._funcs = {}
        
        # _hidden_layers will contain theano Variables of h_s1, ... , h_sn 
        # and h1 
        self._hidden_layers = {}
        
        X_iw = self.sym_vars[u"X_iw"]
        E = self.params[u"E"]    
        #M_iw = self.sym_vars[u"M_iw"]
        y = self.sym_vars[u"y"]
          
        H_s = self._build_sentence_layer()

        self._funcs["H_s"] = theano.function([X_iw], H_s)
        for i in xrange(win_size):
            self._hidden_layers["h_s{}".format(i)] = H_s[i]

        # Compute the first hidden layer (this is the concatenation of all
        # hidden sentence vectors).
        a1 = self.params["b1"]
        for i in xrange(win_size):
            W1_si = self.params["W1_s{}".format(i)]
            a1 += T.dot(H_s[i], W1_si)
        h1 = T.tanh(a1)
        self._hidden_layers["h1"] = h1
        self._funcs["h1"] = theano.function([X_iw], h1)
        
        a2 = T.dot(h1, self.params["W2"]) + self.params["b2"]

        prob_coherent = 1 / (1 + T.exp(-a2))
        prob_incoherent = 1 - prob_coherent

        self._nn_output = T.concatenate(
            [prob_incoherent, prob_coherent], axis=1)
        self._funcs["P(y=1|w)"] = theano.function(
            [X_iw], prob_coherent)
        self._funcs["P(y=0|w)"] = theano.function(
            [X_iw], prob_incoherent)
        self._funcs["P(Y|w)"] = theano.function(
            [X_iw], self._nn_output)
        
        self._nn_output_func = theano.function([X_iw], self._nn_output)

        self._nll = -T.mean(
            T.log(self._nn_output)[T.arange(y.shape[0]), y])      
        
        y_pred = T.argmax(self._nn_output, axis=1)
        self._funcs["avg win error"] = theano.function(
            [X_iw, y], T.mean(T.neq(y, y_pred)))
      
    def _build_sentence_layer(self):

        X_iw = self.sym_vars[u"X_iw"]
        E = self.params[u"E"]    
        y = self.sym_vars[u"y"]
                
        max_sent_len = self.max_sent_len 
        win_size = self.window_size
        word_dim = self.embeddings.embed_dim

        E_pad = T.concatenate(
            [E, T.zeros((1, word_dim,), dtype=theano.config.floatX)])
        
        W = E_pad[X_iw]

        cntr_idx = self.window_size / 2
        H = []
        act_b_size = X_iw.shape[0]
        for i in xrange(win_size):
            X_is = X_iw[:, i * max_sent_len : (i + 1) * max_sent_len]
            W_i = W[:, i * max_sent_len : (i + 1) * max_sent_len, :]
            X_s_sum = T.sum(W_i, axis=1)
            if i < cntr_idx:
                eq = T.all(T.eq(X_is, -1), axis=1).reshape((act_b_size, 1))
                N = T.sum(T.neq(X_is, -1), axis=1).reshape((act_b_size, 1))
                X_sum_pad = T.switch(eq, self.params["W_start"][i], X_s_sum)
                h_s = X_sum_pad / T.clip(N, 1, max_sent_len)
                H.append(h_s)
            elif i > cntr_idx:
                k = i - cntr_idx - 1
                eq = T.all(T.eq(X_is, -1), axis=1).reshape((act_b_size, 1))
                N = T.sum(T.neq(X_is, -1), axis=1).reshape((act_b_size, 1))
                X_sum_pad = T.switch(eq, self.params["W_stop"][k], X_s_sum)
                h_s = X_sum_pad / T.clip(N, 1, max_sent_len)
                H.append(h_s)
            else:
                N = T.sum(T.neq(X_is, -1), axis=1).reshape((act_b_size, 1))
                h_s = X_s_sum / T.clip(N, 1, max_sent_len) 
                H.append(h_s)

        return H

    def log_prob_coherent(self, X_iw):
        P = self._funcs["P(y=1|w)"](X_iw)
        return np.sum(np.log(P))

    def score(self, X_gold, X_perm):
        correct = 0
        total = 0
        for X_iw_gold, X_iw_perm in izip(X_gold, X_perm):
            gold_lp = self.log_prob_coherent(X_iw_gold)
            perm_lp = self.log_prob_coherent(X_iw_perm)
            if gold_lp > perm_lp:
                correct += 1
            total += 1
        return float(correct) / max(total, 1)

    def fit(self, X_iw, y):

        n_batches = X_iw.shape[0] / self.batch_size 
        if X_iw.shape[0] % self.batch_size != 0:
            n_batches += 1

        X_iw_shared = theano.shared(X_iw.astype(np.int32),
            name="X_iw_shared",
            borrow=True)
        y_shared = theano.shared(y.astype(np.int32),
            name="y_shared",
            borrow=True)
 
        X_iw_sym = self.sym_vars["X_iw"]
        y_sym = self.sym_vars["y"]
        
        reg = 0
        for param in self._reg_params:
            reg += (param**2).sum() 

        reg = reg * self.lam * self.window_size / X_iw.shape[0]
        cost = self._nll + reg  
        gtheta = T.grad(cost, self.theta)

        if self.update_method == "adagrad":
            # diagonal adagrad update
            grad_hist_update = self.grad_hist + gtheta**2
            grad_hist_upd_safe = T.switch(
                T.eq(grad_hist_update, 0), 1, grad_hist_update)
            theta_update = self.theta - \
                    self.alpha / T.sqrt(grad_hist_upd_safe) * gtheta
            updates = [(self.grad_hist, grad_hist_update),
                       (self.theta, theta_update)]

        elif self.update_method == "sgd":
            updates = [(self.theta, self.theta - self.alpha * gtheta)]
        else:
            raise Exception("'{}' is not an implemented update method".format(
                self.update_method))
        
        index = T.scalar(dtype='int32')  # index to a [mini]batch
        train_model = theano.function(
            inputs=[index],
            outputs=[cost, self._nll, reg],
            updates=updates,
            givens={
                X_iw_sym: X_iw_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
                y_sym: y_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
            }
        )
        

        for n_iter in xrange(1, self.max_iters + 1):
            print "iter", n_iter
            for i in xrange(n_batches):
                #print "iter", n_iter, "batch", i+1, "/", n_batches
                cost, nll, reg = train_model(i)
                #print "nll {:0.6f}".format(float(nll)),
                #print "reg {:0.6f}".format(float(reg)),
                #print "cost {:0.6f}".format(float(cost))
            #print 
            if self.fit_callback is not None:
                self.fit_callback(self, n_iter)
            
