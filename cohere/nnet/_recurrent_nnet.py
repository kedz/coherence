import theano
import theano.tensor as T
import numpy as np
from cohere.nnet._base import _BaseNNModel
from itertools import izip

class RecurrentNNModel(_BaseNNModel):
    sym_vars = {u"X_iw": T.imatrix(name=u"X_iw"),
                u"y": T.ivector(name=u"y"),
                u"M_iw": T.imatrix(name=u"M_iw")} 

    def _init_params(self):
        
        embeddings = self.embeddings
        initializers = self.initializers

        if initializers is None:
            initializers = dict()

        word_dim = embeddings.embed_dim
        
        # params is a dict mapping variable name to theano Variable objects.
        self.params = {}
        
        # _reg_params is a list of theano Variables that will be subject
        # to regularization.
        self._reg_params = []
        
        # Number of params for W_rec + V_rec + b_rec
        n_recurrent_params = word_dim**2 + word_dim**2 + \
            word_dim
        
        # We are fitting h_0, allocate word_dim params for this 
        # variable.
        n_recurrent_params += word_dim

        if self.fit_embeddings is True:
            n_recurrent_params += \
                self.embeddings.W.shape[0] * word_dim

        # If using trainable start/stop sentences, allocate memory.
        # NOTE: if window size is 5, there will be two distinct start
        # vectors and two distinct stop vectors.
        n_recurrent_params += (self.window_size - 1) * word_dim
        
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
         
        eps = np.sqrt(6. / (self.hidden_dim + word_dim * self.window_size))
        
        # Initialize RNN component
        
        # Init W_rec (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        W_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        W_rec.name = "W_rec"
        self.params[W_rec.name] = W_rec
        self._reg_params.append(W_rec)

        if initializers.get("W_rec", None) is None:
            W_rec_init = np.random.uniform(
                low=-eps, high=eps, 
                size=(word_dim, word_dim)).astype(
                    theano.config.floatX)
        else:
            W_rec_init = initializers.get("W_rec")
        inits.append(W_rec_init)
        current_pointer = next_pointer
        
        # Init V_rec (word_dim x word_dim)
        next_pointer = current_pointer + word_dim**2
        V_rec = self.theta[current_pointer:next_pointer].reshape(
            (word_dim, word_dim))
        V_rec.name = "V_rec"
        self.params[V_rec.name] = V_rec
        
        if initializers.get("V_rec", None) is None:
            V_rec_init = np.random.uniform(
                low=-eps, high=eps,
                size=(word_dim, word_dim)).astype(
                    theano.config.floatX)
        else:
            V_rec_init = initializers.get("V_rec")
        inits.append(V_rec_init)
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
        
        next_pointer = current_pointer + word_dim
        h0 = self.theta[current_pointer:next_pointer].reshape(
            (word_dim,))
        h0.name = "h0"
        self.params[h0.name] = h0   
        
        if initializers.get("h0", None) is None:
            h0_init = np.random.uniform(
                low=-eps, high=eps,
                size=(word_dim,)).astype(theano.config.floatX)
        else:
            h0_init = initializers.get("h0")
        inits.append(h0_init)
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
    
        # Init W2 (hidden_dim x 2)
        next_pointer = current_pointer + self.hidden_dim * 1
        W2 = self.theta[current_pointer:next_pointer].reshape(
            (self.hidden_dim, 1))
        W2.name="W2"
        self.params[W2.name] = W2
        
        if initializers.get("W2", None) is None:
            W2_init = np.random.uniform(
                low=-eps, high=eps, 
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
            next_pointer = current_pointer + \
                self.embeddings.W.shape[0] * word_dim
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
        win_size = self.window_size

        # _funcs will contains compiled theano functions of several
        # variables for debugging purposes
        self._funcs = {}

        # _hidden_layers will contain theano symbolic variables of 
        # h_s1, ... , h_sn and h1.
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
        a1 = self.params["b1"]
        for i in xrange(win_size):
            W1_si = self.params["W1_s{}".format(i)]
            a1 += T.dot(H_s[i], W1_si)
        h1 = T.tanh(a1)
        self._hidden_layers["h1"] = h1
        self._funcs["h1"] = theano.function([X_iw, M_iw], h1)

        a2 = T.dot(h1, self.params["W2"]) + self.params["b2"]

        prob_coherent = 1 / (1 + T.exp(-a2))
        prob_incoherent = 1 - prob_coherent

        self._nn_output = T.concatenate(
            [prob_incoherent, prob_coherent], axis=1)
        self._funcs["P(y=1|w)"] = theano.function(
            [X_iw, M_iw], prob_coherent)
        self._funcs["P(y=0|w)"] = theano.function(
            [X_iw, M_iw], prob_incoherent)
        self._funcs["P(Y|w)"] = theano.function(
            [X_iw, M_iw], self._nn_output)

        self._nn_output_func = theano.function([X_iw, M_iw], self._nn_output)

        self._nll = -T.mean(
            T.log(self._nn_output)[T.arange(y.shape[0]), y])

        y_pred = T.argmax(self._nn_output, axis=1)
        self._funcs["avg win error"] = theano.function(
            [X_iw, M_iw, y], T.mean(T.neq(y, y_pred)))

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
        cntr_idx = win_size / 2 
        for i in xrange(win_size):
            h_si, _ = theano.scan(
                fn=self.step_mask,
                sequences=[
                    {"input": X,   "taps":[i * max_sent_len]}, 
                    {"input": M_T, "taps":[i * max_sent_len]},], 
                outputs_info=h0_block,
                n_steps=max_sent_len)
            h_si = h_si[-1]
            if i < cntr_idx:
                x_iw = X_iw[:,i * max_sent_len : (i+1) * max_sent_len]
                h_si = T.switch(
                    T.all(T.eq(x_iw, -1), axis=1, keepdims=True),  
                    self.params["W_start"][i],
                    h_si)
            elif i > cntr_idx:
                x_iw = X_iw[:,i * max_sent_len : (i+1) * max_sent_len]
                h_si = T.switch(
                    T.all(T.eq(x_iw, -1), axis=1, keepdims=True),    
                    self.params["W_stop"][i - cntr_idx - 1],
                    h_si)

            H_s.append(h_si)
            self._hidden_layers["h_s{}".format(i)] = h_si

        return H_s
 
    def step_mask(self, x_t, m_t, h_tm1):
        h_t_unmasked = self.step(x_t, h_tm1)
        h_t = m_t * h_t_unmasked + (1 - m_t) * h_tm1
        return h_t, theano.scan_module.until(T.all(T.eq(m_t, 0)))
    
    def step(self, x_t, h_tm1):
        W_rec = self.params["W_rec"]
        V_rec = self.params["V_rec"]
        b_rec = self.params["b_rec"]
        h_t = T.tanh(T.dot(h_tm1, V_rec) + T.dot(x_t, W_rec) + b_rec)
        return h_t

    def log_prob_coherent(self, X_iw):
        P = self._funcs["P(y=1|w)"](X_iw, self._mask(X_iw))
        return np.sum(np.log(P))

    def fit(self, X_iw, y):

        n_batches = X_iw.shape[0] / self.batch_size
        if X_iw.shape[0] % self.batch_size != 0:
            n_batches += 1

        X_iw_shared = theano.shared(X_iw.astype(np.int32),
            name="X_iw_shared",
            borrow=True)
        M_iw_shared = theano.shared(self._mask(X_iw).astype(np.int32),
            name="M_iw_shared",
            borrow=True)
        y_shared = theano.shared(y.astype(np.int32),
            name="y_shared",
            borrow=True)

        X_iw_sym = self.sym_vars["X_iw"]
        M_iw_sym = self.sym_vars["M_iw"]
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
                M_iw_sym: M_iw_shared[
                    index * self.batch_size : (index + 1) * self.batch_size],
            }
        )


        batch_nll = np.zeros((n_batches,))
        for n_iter in xrange(1, self.max_iters + 1):
            
            for i in xrange(n_batches):
                cost, nll, reg = train_model(i)
                batch_nll[i] = nll               
            if self.fit_callback is not None:
                self.fit_callback(self, n_iter)
            else:
                print n_iter, "avg batch nll", np.mean(batch_nll)

    def _mask(self, X_iw):
        M_iw = np.ones_like(X_iw)
        M_iw[X_iw == -1] = 0
        return M_iw

    def score(self, X_gold, X_perm):
        correct = 0
        total = 0
        for X_iw_gold, X_iw_perm, in izip(X_gold, X_perm):
            gold_lp = self.log_prob_coherent(X_iw_gold)
            perm_lp = self.log_prob_coherent(X_iw_perm)
            if gold_lp > perm_lp:
                correct += 1
            total += 1
        return float(correct) / max(total, 1)
