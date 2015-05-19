import theano
import theano.tensor as T
import numpy as np
from cohere.nnet._base import _BaseNNModel
from cohere.nnet._transformers import TreeTransformer
from itertools import izip


class RecursiveNNModel(_BaseNNModel):

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
        n_recurrent_params += word_dim * 2
        
        #print "n_recurrent_params:", n_recurrent_params
        
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2
        n_feedforward_params = self.hidden_dim * word_dim * \
            self.window_size + self.hidden_dim + 1 * self.hidden_dim + 1
        
        #print "n_feedforward_params:", n_feedforward_params
        
        n_params = n_recurrent_params + n_feedforward_params
        
        #print "n_params:", n_params
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
        next_pointer = current_pointer + word_dim    
        W_start = self.theta[current_pointer:next_pointer].reshape(
                (1, word_dim))
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
        next_pointer = current_pointer +  word_dim    
        W_stop = self.theta[current_pointer:next_pointer].reshape(
                (1, word_dim))
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

    def _init_network(self):

        self._sym_vars = {} 
        # X is a matrix of sentences used in the current batch.
        # X.shape = (n_sents, max_sent_len)
        X = T.imatrix(name="X")
        self._sym_vars[u"X"] = X
        
        # O is the associated op sequence for each sentence in the current 
        # batch. Each instruction contains 6 integers.
        # O.shape = (n_sents, max_ops_len, 6)
        O = T.tensor3(name="O", dtype="int32")
        self._sym_vars[u"O"] = O
        
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
        n_sents = X.shape[0]
        clique_size = C.shape[1] # The window size of the cliques.
        embed_dim = E.shape[1]   # The word embedding dimensionality.
        all_clique_size = T.scalar(dtype="int32")
        self._sym_vars["all_clique_size"] = all_clique_size
        
        X_emb = E[X] 
        X_emb = X_emb.dimshuffle(1,0,2)

        O_T = O.dimshuffle(1,0,2).reshape(
            (O.shape[1], O.shape[0], 6))
    
        B = T.zeros_like(X_emb)
        B = T.set_subtensor(B[0,:], X_emb[0,:])
        indices = T.arange(X_emb.shape[1], dtype="int32")
        
        H_s, _ = theano.scan(
                fn=self.tree_step,
                sequences=O_T,
                non_sequences=[X_emb, indices, n_sents], 
                outputs_info=B)
        H_s = H_s[-1][0]
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
        prob_coherent = T.nnet.sigmoid(
            T.dot(H2, self.params["W2"]) + self.params["b2"])
        prob_incoherent = 1 - prob_coherent

        # Training cost, nll, and regularization vars setup here.  
        nn_output = T.concatenate([prob_incoherent, prob_coherent], axis=1)
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

        log_O = T.log(prob_coherent)

        # This is cause a segfault for some reason?
#        doc_lps, _ = theano.scan(
#                fn=lambda s: T.sum(log_O[s[0]:s[1]]),
#                sequences=S)
#        pairwise_scores = doc_lps[P]
#        n_correct = T.sum(pairwise_scores[:,0] > pairwise_scores[:,1])
#        accuracy = n_correct * 1. / P.shape[0]
#        self._test_acc = theano.function([X, O, C, S, P],
#            accuracy)
        self._test_log_prob = theano.function([X, O, C],
            log_O)

    def tree_step(self, O_t, B_tm1, X, indices, n_sents):
        # op code (1 means simply propagate previous result)
        # (0 means compute hiden layer for left and right args).
        o_code = O_t[:,0]
        # Location in buffer to place result of computation.
        o_idx = O_t[:,1]
        # Left source (1 is word embeddings, 0 is buffer B)
        o_src_l = O_t[:,2]
        # Left index (grab the vector at this index from left source).
        i_l = O_t[:,3]
        # Right source (1 is word embeddings, 0 is buffer B)
        o_src_r = O_t[:,4]
        # Right index (grab the vector at this index from right source).
        i_r = O_t[:,5]
        

        Op_left = T.eq(o_src_l, 1).reshape((n_sents, 1))
        X_left = X[i_l, indices]
        B_tm1_left = B_tm1[i_l, indices]
        left = T.switch(Op_left, X_left, B_tm1_left) 
        left_dot = T.dot(left, self.params["W_rec_l"])

        Op_right = T.eq(o_src_r, 1).reshape((n_sents, 1))
        X_right = X[i_r, indices]
        B_tm1_right = B_tm1[i_r, indices]
        right = T.switch(Op_right, X_right, B_tm1_right)
        right_dot = T.dot(right, self.params["W_rec_r"])

        h = T.tanh(left_dot + right_dot + self.params["b_rec"])
        
        result = T.switch(
            T.eq(o_code, 1).reshape((n_sents, 1)), 
            B_tm1[0], 
            h)

        B_t = T.set_subtensor(B_tm1[o_idx, indices], result)

        #result = T.switch(T.eq(o_code, 1), B[0], h)
        #B_tp1 = T.set_subtensor(B[o_idx, index], result)
        #B_t = T.set_subtensor(B_tm1[T.zeros_like(index), index], B_tm1[0])
        #B_t = T.set_subtensor(B_tm1[T.zeros_like(indices), indices], result)
        return B_t

    def score(self, dataset):
        trans = TreeTransformer(self.embeddings, self.window_size)
        X, O, C, S, P = trans.transform_test(dataset)
        lp = self._test_log_prob(X, O, C) 
        doc_lps = np.array([np.sum(lp[s[0]:s[1]]) for s in S])
        pairwise_scores = doc_lps[P]
        n_correct = np.sum(pairwise_scores[:,0] > pairwise_scores[:,1])
        accuracy = n_correct * 1. / P.shape[0]

        return accuracy

    def fit(self, dataset):
        trans = TreeTransformer(
            self.embeddings, self.batch_size, self.window_size)
        X, O, C, y, S, n_batches= trans.transform_gold(dataset)
        X_sh = theano.shared(X, name="X_shared", borrow=True)
        O_sh = theano.shared(O, name="O_shared", borrow=True)
        C_sh = theano.shared(C, name="C_shared", borrow=True)
        y_sh = theano.shared(y, name="y_shared", borrow=True)

        b_size = self.batch_size
        idx = T.scalar(dtype='int32')  # index to a [mini]batch
        S_var = T.ivector()
        train_model = theano.function(
            inputs=[idx, S_var, self._sym_vars["all_clique_size"]],
            outputs=self._training_outputs,
            updates=self._updates,
            givens={
                self._sym_vars["X"]: X_sh[S_var],
                self._sym_vars["O"]: O_sh[S_var],
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

    def ready(self):
        self._init_params()
        self._init_network()
