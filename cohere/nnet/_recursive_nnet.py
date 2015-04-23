import theano
import theano.tensor as T
import numpy as np
#import cPickle as pickle
#import os
#import datetime
from cohere.nnet._base import _BaseNNModel
from itertools import izip



class RecursiveNNModel(_BaseNNModel):
    def __init__(self, embeddings, max_sent_len=150, max_ops_len=40, 
                 window_size=3, hidden_dim=100, alpha=.01, lam=1.25,
                 batch_size=25, max_iters=10, 
                 fit_embeddings=False, fit_callback=None, 
                 update_method="adagrad", initializers=None):
         
        self.max_ops_len = max_ops_len

        self.sym_vars = {u"X_iw": T.imatrix(name=u"X_iw"),
                         u"y": T.ivector(name=u"y"),
                         u"O_iw": T.tensor3(name=u"O_iw", dtype="int32")}

        super(RecursiveNNModel, self).__init__(
            embeddings, max_sent_len=max_sent_len, hidden_dim=hidden_dim,
            window_size=window_size, alpha=alpha, lam=lam,
            batch_size=batch_size, fit_embeddings=fit_embeddings, 
            fit_callback=fit_callback, update_method=update_method,
            initializers=initializers, max_iters=max_iters)
       

        #if embeddings is not None:
        #    self.init_params(self.embeddings, initializers)
        #    self.build_network(
        #        self.max_sent_len, self.max_ops_len, self.window_size)

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
        
        # Number of params for W_rec_l + W_rec_r + b_rec
        n_recurrent_params = word_dim**2 + word_dim**2 + \
            word_dim
       
        # Number of params for pads.
        n_recurrent_params += self.window_size / 2 * word_dim * 2
        
        print "n_recurrent_params:", n_recurrent_params
        
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2
        n_feedforward_params = self.hidden_dim * word_dim * \
            self.window_size + self.hidden_dim + 1 * self.hidden_dim + 1
        
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
        
        # Init W_rec_l (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        W_rec_l = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        W_rec_l.name = "W_rec_l"
        self.params[W_rec_l.name] = W_rec_l
        self._reg_params.append(W_rec_l)
        
        if initializers.get("W_rec_l", None) is None:
            W_rec_l_init = np.random.uniform(
                low=-.25, high=.25, 
                size=(word_dim, word_dim)).astype(
                    theano.config.floatX)
        else:
            W_rec_l_init = initializers.get("W_rec_l")
        inits.append(W_rec_l_init)
        current_pointer = next_pointer
        
        # Init W_rec_r (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        W_rec_r = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        W_rec_r.name = "W_rec_r"
        self.params[W_rec_r.name] = W_rec_r
        self._reg_params.append(W_rec_r) 

        if initializers.get("W_rec_r", None) is None:
            W_rec_r_init = np.random.uniform(
                low=-.25, high=.25, 
                size=(word_dim, word_dim)).astype(
                    theano.config.floatX)
        else:
            W_rec_r_init = initializers.get("W_rec_r")
        inits.append(W_rec_r_init)
        current_pointer = next_pointer
        
        # Init b_rec (word_dim)
        next_pointer = current_pointer + word_dim
        b_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim,))
        b_rec.name = "b_rec"
        self.params[b_rec.name] = b_rec
        
        if initializers.get("b_rec", None) is None:
            b_rec_init = np.zeros(word_dim, dtype=theano.config.floatX)
        else:
            b_rec_init = initializers.get("b_rec")
        inits.append(b_rec_init)
        current_pointer = next_pointer       
        
        # Initialize Feed-Forward NN component
        
        eps = np.sqrt(6. / (self.hidden_dim + word_dim * self.window_size))

        # Init W1_s...
        for i in xrange(self.window_size):
            name = "W1_s{}".format(i)
            next_pointer = current_pointer + self.hidden_dim * word_dim
            W1_s = self.theta[current_pointer:next_pointer].reshape(
                (word_dim, self.hidden_dim))
            W1_s.name = name
            self.params[W1_s.name] = W1_s
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
        
        # Init b2 (1)
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
       
        # If using adagrad, we need to keep a running sum of historical squared
        # gradients for each parameter.
        if self.update_method == "adagrad":
            self.grad_hist = theano.shared(
                np.zeros_like(self.theta.get_value(borrow=True)),
                name="grad_hist",
                borrow=True)

    def _build_network(self):
        max_sent_len = self.max_sent_len
        max_ops_len = self.max_ops_len
        win_size = self.window_size
        
        # _funcs will contains compiled theano functions of several
        # variables for debugging purposes
        self._funcs = {}
        
        # _hidden_layers will contain theano symbolic variables of 
        # h_s1, ... , h_sn and h1.
        self._hidden_layers = {}
        
        X_iw = self.sym_vars[u"X_iw"]
        E = self.params[u"E"]    
        O_iw = self.sym_vars[u"O_iw"]
        y = self.sym_vars[u"y"]
                
        H_s = self._build_sentence_layer(
            X_iw, O_iw, y, E, max_sent_len, max_ops_len, win_size)

        self._funcs["H_s"] = theano.function([X_iw, O_iw], H_s)
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
        self._funcs["h1"] = theano.function([X_iw, O_iw], h1)

        a2 = T.dot(h1, self.params["W2"]) + self.params["b2"]
        
        prob_coherent = 1 / (1 + T.exp(-a2))
        prob_incoherent = 1 - prob_coherent

        self._nn_output = T.concatenate(
            [prob_incoherent, prob_coherent], axis=1)
        self._funcs["P(y=1|w)"] = theano.function(
            [X_iw, O_iw], prob_coherent)
        self._funcs["P(y=0|w)"] = theano.function(
            [X_iw, O_iw], prob_incoherent)
        self._funcs["P(Y|w)"] = theano.function(
            [X_iw, O_iw], self._nn_output)
        
        self._nn_output_func = theano.function([X_iw, O_iw], self._nn_output)

        self._nll = -T.mean(
            T.log(self._nn_output)[T.arange(y.shape[0]), y])      
        
        y_pred = T.argmax(self._nn_output, axis=1)
        self._funcs["avg win error"] = theano.function(
            [X_iw, O_iw, y], T.mean(T.neq(y, y_pred)))

    def _build_sentence_layer(self, X_iw, O_iw, y, E, max_sent_len, 
                              max_ops_len, win_size):
 
        # Index into word embedding matrix and construct a 
        # 3d tensor for input. After dimshuffle, dimensions should
        # be (MAX_LEN * window_size, batch_size, word_dim).
        X = E[X_iw] 
        X = X.dimshuffle(1,0,2)
        self._funcs["X"] = theano.function([X_iw], X)  
        B = []
        cntr_idx = self.window_size / 2
        for i in xrange(self.window_size):
            
            B_s = T.zeros_like(X[:max_sent_len])

            if i < cntr_idx:
                print "left", i
                B_s_slice = T.switch(
                    T.eq(X_iw[:, i * max_sent_len], -1).reshape(
                        (X.shape[1] ,1)),
                    T.alloc(self.params["W_start"][i],X.shape[1]),
                    X[i * max_sent_len,:])
                B_s = T.set_subtensor(
                    B_s[:,:],
                    B_s_slice)

            elif i > cntr_idx:
                k = i - cntr_idx - 1
                print "right", k
                B_s_slice = T.switch(
                    T.eq(X_iw[:, i * max_sent_len], -1).reshape(
                        (X.shape[1] ,1)),
                    T.alloc(self.params["W_stop"][k],X.shape[1]),
                    X[i * max_sent_len,:])
                B_s = T.set_subtensor(
                    B_s[:,:],
                    B_s_slice)

            else:
                B_s = T.set_subtensor(B_s[0,:], X[i * max_sent_len,:])
            B.append(B_s)

        self._funcs["B"] = theano.function([X_iw], B)
        O = O_iw.dimshuffle(1,0,2).reshape(
            (O_iw.shape[1], O_iw.shape[0], 6))
        self._funcs["O"] = theano.function([O_iw], O)
        
        indices = T.arange(X.shape[1], dtype="int32")
        H_s = [] 
        for i in xrange(win_size):
            results, _ = theano.scan(
                fn=self.tree_step,
                sequences=[
                    {"input": O,   "taps":[i * max_ops_len]},], 
                non_sequences=[X[i *max_sent_len:(i+1)*max_sent_len],
                               indices, indices.shape[0]],
                outputs_info=B[i],
                n_steps=max_ops_len)
            h_si = results[-1][0]
            H_s.append(h_si)
            self._hidden_layers["h_s{}".format(i)] = h_si
            self._funcs["h_s{}".format(i)] = theano.function(
                [X_iw, O_iw], h_si, on_unused_input='ignore') 
            
        return H_s

    def tree_step(self, O_t, B, X_s, index, batch_size):

        # op code (1 means simply propagate previous result)
        # (0 means compute hiden layer for left and right args).
        o_code = O_t[:,0].reshape((batch_size, 1))
        # Location in buffer to place result of computation.
        o_idx = O_t[:,1]
        # Left source (1 is word embeddings, 0 is buffer B)
        o_src_l = O_t[:,2].reshape((batch_size, 1))
        # Left index (grab the vector at this index from left source).
        i_l = O_t[:,3]
        # Right source (1 is word embeddings, 0 is buffer B)
        o_src_r = O_t[:,4].reshape((batch_size, 1))
        # Right index (grab the vector at this index from right source).
        i_r = O_t[:,5]

        left = T.switch(T.eq(o_src_l, 1), 
            (X_s[i_l, index]), (B[i_l, index]))
        left_dot = T.dot(left, self.params["W_rec_l"])

        right = T.switch(T.eq(o_src_r, 1),
            (X_s[i_r, index]), (B[i_r, index]))
        right_dot = T.dot(right, self.params["W_rec_r"])

        h = T.tanh(left_dot + right_dot + self.params["b_rec"])
        
        result = T.switch(T.eq(o_code, 1), B[0], h)
        B_tp1 = T.set_subtensor(B[o_idx, index], result)

        return B_tp1, theano.scan_module.until(T.all(T.eq(o_code, 1)))

    def log_prob_coherent(self, X_iw, O_iw):
        P = self._funcs["P(y=1|w)"](X_iw, O_iw)
        return np.sum(np.log(P))

    def fit(self, X_iw, O_iw, y):

        n_batches = X_iw.shape[0] / self.batch_size 
        if X_iw.shape[0] % self.batch_size != 0:
            n_batches += 1

        X_iw_shared = theano.shared(X_iw.astype(np.int32),
            name="X_iw_shared",
            borrow=True)
        O_iw_shared = theano.shared(O_iw.astype(np.int32),
            name="O_iw_shared",
            borrow=True)
        y_shared = theano.shared(y.astype(np.int32),
            name="y_shared",
            borrow=True)
 
        X_iw_sym = self.sym_vars["X_iw"]
        O_iw_sym = self.sym_vars["O_iw"]
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
                O_iw_sym: O_iw_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
            }
        )
        

        for n_iter in xrange(1, self.max_iters + 1):

            for i in xrange(n_batches):
                print "iter", n_iter, "batch", i+1, "/", n_batches
                cost, nll, reg = train_model(i)
                print "nll {:0.6f}".format(float(nll)),
                print "reg {:0.6f}".format(float(reg)),
                print "cost {:0.6f}".format(float(cost))
            print 
            if self.fit_callback is not None:
                self.fit_callback(self, n_iter)

    def score(self, X_gold, O_gold, X_perm, O_perm):
        correct = 0
        total = 0
        for X_iw_gold, O_iw_gold, X_iw_perm, O_iw_perm in izip(
                X_gold, O_gold, X_perm, O_perm):
            gold_lp = self.log_prob_coherent(X_iw_gold, O_iw_gold)
            perm_lp = self.log_prob_coherent(X_iw_perm, O_iw_perm)
            if gold_lp > perm_lp:
                correct += 1
            total += 1
        return float(correct) / max(total, 1)


    def dbg_print_window(self, X_iw):
        print X_iw.shape[1] / 3
        print self.max_sent_len
        for x_iw in X_iw:
            for k in xrange(self.window_size):
                #if x_iw[k * self.max_sent_len] == -1:
                #    continue
                print k,
                for xi in x_iw[k*self.max_sent_len:(k+1)*self.max_sent_len]:
                    if xi == -1:
                        continue
                    print self.embeddings.index2token[xi],
                print 
            print 
