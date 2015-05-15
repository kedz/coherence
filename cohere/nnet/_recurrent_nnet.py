import theano
import theano.tensor as T
import numpy as np
from cohere.nnet._base import _BaseNNModel
from cohere.nnet._transformers import TokensTransformer
from itertools import izip

class RecurrentNNModel(_BaseNNModel):



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
        n_recurrent_params += 2 * word_dim   #(self.window_size - 1) * word_dim
        
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
        self._reg_params.append(V_rec)
        
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
            (1, word_dim,))
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

        # Init W1...
        name = "W1"
        next_pointer = current_pointer + \
            self.hidden_dim * word_dim * self.window_size
        
        W1 = self.theta[current_pointer:next_pointer].reshape(
                (word_dim * self.window_size, self.hidden_dim))
        W1.name = name
        self.params[W1.name] = W1
        self._reg_params.append(W1)
            
        if initializers.get(name, None) is None:
            W1_init = np.random.uniform(
                low=-eps, high=eps, 
                size=(word_dim * self.window_size, self.hidden_dim)).astype(
                    theano.config.floatX)
        else:
            W1_init = initializers.get(name)
        inits.append(W1_init)
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
        self._reg_params.append(W2)
        
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
        #next_pointer = current_pointer + self.window_size / 2 * word_dim
        next_pointer = current_pointer + word_dim
        W_start = self.theta[current_pointer:next_pointer].reshape(
                (1, word_dim))
                #(self.window_size / 2, word_dim))
        W_start.name = "W_start"
        self.params[W_start.name] = W_start

        if initializers.get("W_start", None) is None:
            W_start_init = np.zeros((1, word_dim),
                                    dtype=theano.config.floatX)
        else:
            W_start_init = initializers.get("W_start")
        inits.append(W_start_init)
        current_pointer = next_pointer

        # W_stop (window_size / 2, word_dim)
        next_pointer = current_pointer + word_dim
        #next_pointer = current_pointer + self.window_size / 2 * word_dim
        W_stop = self.theta[current_pointer:next_pointer].reshape(
                (1, word_dim))
        W_stop.name = "W_stop"
        self.params[W_stop.name] = W_stop

        if initializers.get("W_stop", None) is None:
            W_stop_init = np.zeros((1, word_dim),
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

    def _init_network(self):

        self._sym_vars = {} 
        # X is a matrix of sentences used in the current batch.
        # X.shape = (n_sents, max_sent_len)
        X = T.imatrix(name="X")
        self._sym_vars[u"X"] = X
        
        # M is the associated mask with each sentence in the current batch.
        # M.shape = (n_sents, max_sent_len)
        M = T.imatrix(name="M")
        self._sym_vars[u"M"] = M
        
        # C is a matrix of cliques, with values indicating which sentences
        # are in each clique.
        # C.shape = (n_cliques, window_size)
        C = T.imatrix(name="C")
        self._sym_vars[u"C"] = C

        # y is a vector of clique labels: 1 = coherent, 0 incoherent.
        # y.shape = (n_cliques,)
        y = T.ivector(name="y")
        self._sym_vars[u"y"] = y

        # S is a matrix of slices for use during test time.
        # E.g. C[S[i,0] : S[i,1]] contains all cliques in the ith document.
        S = T.imatrix("S")

        # P is a matrix of pairs of document indices corresponding to 
        # evaluation comparisons.
        # E.g. P[i] corresponds to the index of the ith comparison
        # where P[i,0] is the index of the gold document, and P[i,1] 
        # a permuted document we are comparing it to.
        P = T.imatrix("P")

        # E is a matrix of word embeddings.
        # E.shape = (vocab_size, embedding_dim)
        E = self.params["E"]
        
        n_cliques = C.shape[0]   # The number of cliques (aka batch size).
        clique_size = C.shape[1] # The window size of the cliques.
        embed_dim = E.shape[1]   # The word embedding dimensionality.
        all_clique_size = T.scalar(dtype="int32")
        self._sym_vars["all_clique_size"] = all_clique_size
        
        X_emb = E[X] 
        X_emb = X_emb.dimshuffle(1,0,2)
        M_T = M.dimshuffle(1,0).reshape(
            (X_emb.shape[0], X_emb.shape[1], 1))
        H_s, _ = theano.scan(
                fn=self.step_mask,
                sequences=[X_emb, M_T],
                outputs_info=T.alloc(
                    self.params["h0"], X_emb.shape[1], X_emb.shape[2]))
        H_s = H_s[-1]
        H1 = H_s[C].reshape((n_cliques, clique_size * embed_dim))

        center = self.window_size / 2
        for k in xrange(self.window_size):
            if k < center:
                block = self.params["W_start"]
            elif k > center:
                block = self.params["W_stop"]
            else:
                continue

            H1 = T.set_subtensor(
                H1[:,k * embed_dim:(k+1)*embed_dim], 
                T.switch(T.eq(C[:,k], -1).reshape((n_cliques, 1)),
                         T.alloc(block, n_cliques, embed_dim),
                         H1[:,k * embed_dim:(k+1)*embed_dim]))
        
        H2 = T.tanh(T.dot(H1, self.params["W1"]) + self.params["b1"]) 
        O = T.nnet.sigmoid(
            T.dot(H2, self.params["W2"]) + self.params["b2"])
  
        # Training cost, nll, and regularization vars setup here.  
        nn_output = T.concatenate([1-O, O], axis=1)
        nll = -T.mean(
            T.log(nn_output)[T.arange(y.shape[0]), y])

        reg = 0
        for param in self._reg_params:
            reg += (param**2).sum()

        reg *= self.lam * self.window_size / all_clique_size
        cost = nll + reg
        self._training_outputs = [cost, nll, reg]

        gtheta = T.grad(cost, wrt=self.theta)

        if self.update_method == "adagrad":
            # diagonal adagrad update
            grad_hist_update = self.grad_hist + gtheta**2
            grad_hist_upd_safe = T.switch(
                T.eq(grad_hist_update, 0), 1, grad_hist_update)
            theta_update = self.theta - \
                    self.alpha / T.sqrt(grad_hist_upd_safe) * gtheta
            self._updates = [(self.grad_hist, grad_hist_update),
                             (self.theta, theta_update)]

        elif self.update_method == "sgd":
            self._updates = [(self.theta, self.theta - self.alpha * gtheta)]
        else:
            raise Exception("'{}' is not an implemented update method".format(
                self.update_method))
        
        # Testing outputs setup here.

        log_O = T.log(O)

        doc_lps, _ = theano.scan(
                fn=lambda s: T.sum(log_O[s[0]:s[1]]),
                sequences=S)
        pairwise_scores = doc_lps[P]
        n_correct = T.sum(pairwise_scores[:,0] > pairwise_scores[:,1])
        accuracy = n_correct * 1. / P.shape[0]
        self._test_acc = theano.function([X, M, C, S, P],
            accuracy)

        return H_s
 
    def step_mask(self, x_t, m_t, h_tm1):
        h_t_unmasked = self.step(x_t, h_tm1)
        h_t = T.switch(T.eq(m_t, 1), h_t_unmasked, h_tm1)
        return h_t#, theano.scan_module.until(T.eq(T.sum(m_t), 0))
    
    def step(self, x_t, h_tm1):
        W_rec = self.params["W_rec"]
        V_rec = self.params["V_rec"]
        b_rec = self.params["b_rec"]
        h_t = T.tanh(T.dot(h_tm1, V_rec) + T.dot(x_t, W_rec) + b_rec)
        return h_t

    def _mask(self, X):
        M = np.ones_like(X)
        M[X == -1] = 0
        return M

    def score(self, dataset):
        trans = TokensTransformer(self.embeddings, self.window_size)
        X, C, S, P = trans.transform_test(dataset)
        M = self._mask(X)
        return  float(self._test_acc(X, M, C, S, P))

    def fit(self, dataset):
        X, C, y, S, n_batches = self._prep_fit_data(dataset)
        M = self._mask(X)
        X_sh = theano.shared(X, name="X_shared", borrow=True)
        C_sh = theano.shared(C, name="C_shared", borrow=True)
        y_sh = theano.shared(y, name="y_shared", borrow=True)
        M_sh = theano.shared(M, name="M_shared", borrow=True)
        
        b_size = self.batch_size
        idx = T.scalar(dtype='int32')  # index to a [mini]batch
        S_var = T.ivector()
        train_model = theano.function(
            inputs=[idx, S_var, self._sym_vars["all_clique_size"]],
            outputs=self._training_outputs,
            updates=self._updates,
            givens={
                self._sym_vars["X"]: X_sh[S_var],
                self._sym_vars["M"]: M_sh[S_var],
                self._sym_vars["C"]: C_sh[idx * b_size : (idx + 1) * b_size],
                self._sym_vars["y"]: y_sh[idx * b_size : (idx + 1) * b_size], 
            }
        )

        all_clique_size = C.shape[0]
        batch_nll = np.zeros((n_batches,))
        for n_iter in xrange(1, self.max_iters + 1):
           
            for i in xrange(n_batches):
                cost, nll, reg = train_model(i, S[i], all_clique_size)
                batch_nll[i] = nll   
            if self.fit_callback is not None:
                self.fit_callback(self, n_iter, np.mean(batch_nll))

            else:
                print n_iter, "avg batch nll", np.mean(batch_nll)


    def _prep_fit_data(self, dataset):
        n_docs = len(dataset)
        win_pad = int(self.window_size) / 2
        center = win_pad 

        trans = TokensTransformer(self.embeddings, self.window_size)
        X, I = trans.transform_gold(dataset)
        
        n_cliques = np.sum((I[d + 1] - I[d])**2 for d in xrange(n_docs))
        C = np.ones((n_cliques, self.window_size), dtype="int32") 
        y = np.ones((n_cliques), dtype="int32")                

        offset = 0
        for d in xrange(n_docs):
            doc_size = I[d + 1] - I[d]
            for i in xrange(doc_size):
                for j in xrange(doc_size):
                    for pos, k in enumerate(
                            xrange(i - win_pad, i + win_pad + 1)):
                        if k < 0 or k >= doc_size:
                            C[offset,pos] = -1
                        elif pos == center:
                            C[offset,pos] = I[d] + j
                            y[offset] = 1 if i == j else 0
                        else:
                            C[offset,pos] = I[d] + k
                    offset += 1
        
        n_batches = C.shape[0] / self.batch_size
        n_batches += 1 if C.shape[0] % self.batch_size != 0 else 0
        S = []

        for b in xrange(n_batches):
            B = C[b * self.batch_size : (b + 1) * self.batch_size]
            s, B_inv = np.unique(B, return_inverse=True)
            if s[0] == -1:
                s = s[1:]
                B_inv -= 1
            B_inv = B_inv.reshape(B.shape)
            S.append(s)
            C[b * self.batch_size : (b + 1) * self.batch_size] = B_inv
        return X, C, y, S, n_batches

    def ready(self):
        self._init_params()
        self._init_network()
