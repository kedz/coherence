import theano
import theano.tensor as T
import numpy as np
from itertools import izip

class CBOWModel(object):
    def __init__(self, hidden_dim=100, sentence_dim=50, window_size=3,
                 learning_rate=.1, max_iters=75, batch_size=25):
        self.hidden_dim_ = hidden_dim
        self.window_size_ = window_size
        self.sentence_dim_ = sentence_dim
        self.window_dim_ = window_size * sentence_dim
        self.learning_rate_ = learning_rate
        self.max_iters_ = max_iters
        self.batch_size_ = batch_size

        self.init_nnet()

    def init_nnet(self):

        # Initialize weight and biases
        W1_ = theano.shared(
            value=np.random.uniform(
                size=(self.window_dim_, self.hidden_dim_)
                    ).astype(theano.config.floatX),
            name='W1',
            borrow=True)
        W2_ = theano.shared(
            value=np.random.uniform(
                size=(self.hidden_dim_, 2)).astype(theano.config.floatX),
            name='W2',
            borrow=True)

        b1_ = theano.shared(
            value=np.zeros((self.hidden_dim_,), dtype=theano.config.floatX),
            name='b1', borrow=True)
        b2_ = theano.shared(value=np.zeros((2,), dtype=theano.config.floatX),
            name='b2', borrow=True)

        ### input and class label theano vars
        x = T.matrix('x')
        y = T.ivector('y')
        self.x_ = x
        self.y_ = y

        ### nnet architecture ###
        h1_ = T.tanh(T.dot(x, W1_) + b1_)
        nnet_out = T.nnet.softmax(T.dot(h1_, W2_) + b2_)

        # define document probability 
        doc_prob = T.sum(T.log(nnet_out)[:,1])
        self.doc_prob = theano.function([x], doc_prob)

        pred = T.argmax(nnet_out, axis=1)
        #self.pred_ = pred
        self.predict = theano.function([x], pred)        
        err = T.mean(T.neq(y, pred))
        self.err = theano.function([x, y], err)


        # Define cost, gradients, and update


        cost = -T.mean(T.log(nnet_out)[T.arange(y.shape[0]), y])
        
        dW1 = T.grad(cost, W1_)
        db1 = T.grad(cost, b1_)
        dW2 = T.grad(cost, W2_)
        db2 = T.grad(cost, b2_)
        updates = [(W1_, W1_ - self.learning_rate_ * dW1),
                   (b1_, b1_ - self.learning_rate_ * db1),
                   (W2_, W2_ - self.learning_rate_ * dW2),
                   (b2_, b2_ - self.learning_rate_ * db2),]
        self.cost_ = cost
        self.updates_ = updates

    def fit(self, X_train, y_train):

        shared_X = theano.shared(
            np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(y_train, dtype='int32'), borrow=True)

        n_batches = shared_X.get_value(borrow=True).shape[0] / self.batch_size_


        index = T.scalar(dtype='int32')  # index to a [mini]batch
        train_model = theano.function(
            inputs=[index],
            outputs=self.cost_,
            updates=self.updates_,
            givens={
                self.x_: shared_X[
                    index * self.batch_size_ : (index+1) * self.batch_size_],
                self.y_: shared_y[
                    index * self.batch_size_ : (index+1) * self.batch_size_]
            }
        )

        for n_iter in xrange(self.max_iters_):
            #print "iter", n_iter
            for i in xrange(n_batches):
                train_model(i)

            #err = self.err(
            #    shared_X.get_value(borrow=True),
            #    shared_y.get_value(borrow=True))
            #print "avg training err {:0.4f}".format(float(err))



class RNNModel(object):
    def __init__(self, embedding, hidden_dim=100, sentence_dim=50,
                 window_size=3, learning_rate=.1, max_iters=75, 
                 batch_size=25):
        self.hidden_dim_ = hidden_dim
        self.window_size_ = window_size
        self.sentence_dim_ = sentence_dim
        self.window_dim_ = window_size * sentence_dim
        self.learning_rate_ = learning_rate
        self.max_iters_ = max_iters
        self.batch_size_ = batch_size
        self._embedding = theano.shared(
            value=embedding.embed_.astype(theano.config.floatX),
            name='embeddings',
            borrow=True)

        self.init_nnet()

    def init_nnet(self):
        idxs1 = T.imatrix()
        idxs2 = T.imatrix()
        idxs3 = T.imatrix()
        self.x1_ = self._embedding[idxs1].reshape(
            (idxs1.shape[0], self.sentence_dim_))
        self.x2_ = self._embedding[idxs2].reshape(
            (idxs2.shape[0], self.sentence_dim_))
        self.x3_ = self._embedding[idxs3].reshape(
            (idxs3.shape[0], self.sentence_dim_))
        self.idxs1_ = idxs1
        self.idxs2_ = idxs2
        self.idxs3_ = idxs3
        
        comp_dims = (self.sentence_dim_, self.sentence_dim_)
        Wc_h_ = theano.shared(
            value=np.random.uniform(size=comp_dims).astype(
                theano.config.floatX),
            name='Wc_h',
            borrow=True)
        Wc_x_ = theano.shared(
            value=np.random.uniform(size=comp_dims).astype(
                theano.config.floatX),
            name='Wc_x',
            borrow=True)

        h0_1_ = theano.shared(
            value=np.zeros(self.sentence_dim_, dtype=theano.config.floatX),
            name='h0_1',
            borrow=True)

        h0_2_ = theano.shared(
            value=np.zeros(self.sentence_dim_, dtype=theano.config.floatX),
            name='h0_2',
            borrow=True)

        h0_3_ = theano.shared(
            value=np.zeros(self.sentence_dim_, dtype=theano.config.floatX),
            name='h0_3',
            borrow=True)

        bc_ = theano.shared(
            value=np.zeros(self.sentence_dim_, dtype=theano.config.floatX),
            name="bc",
            borrow=True)

        b1_ = theano.shared(
            value=np.zeros(self.hidden_dim_, dtype=theano.config.floatX),
            name="b1",
            borrow=True)


        
        def recurrence(x_t, h_tm1):
            
            h_t = T.tanh(T.dot(x_t, Wc_x_) + T.dot(h_tm1, Wc_h_) + bc_)
            #h_t = T.dot(x_t, Wc_x_) + T.dot(h_tm1, Wc_h_) + bc_
            return [h_t]

        s1_, _ = theano.scan(fn=recurrence,
                             sequences=self.x1_,
                             outputs_info=h0_1_,
                             n_steps=self.x1_.shape[0])
        s2_, _ = theano.scan(fn=recurrence,
                             sequences=self.x2_,
                             outputs_info=h0_2_,
                             n_steps=self.x2_.shape[0])
        s3_, _ = theano.scan(fn=recurrence,
                             sequences=self.x3_,
                             outputs_info=h0_3_,
                             n_steps=self.x3_.shape[0])

        s1_ = s1_[-1]
        s2_ = s2_[-1]
        s3_ = s3_[-1]

        layer1_dims = (self.sentence_dim_, self.hidden_dim_) 
        W1_1_ = theano.shared(
            value=np.random.uniform(size=layer1_dims).astype(
                theano.config.floatX),
            name="W1_1",
            borrow=True)
        W1_2_ = theano.shared(
            value=np.random.uniform(size=layer1_dims).astype(
                theano.config.floatX),
            name="W1_2",
            borrow=True)
        W1_3_ = theano.shared(
            value=np.random.uniform(size=layer1_dims).astype(
                theano.config.floatX),
            name="W1_3",
            borrow=True)
        h1_ = T.tanh(T.dot(s1_, W1_1_) + T.dot(s2_, W1_2_) \
            + T.dot(s3_, W1_3_) + b1_) 

        ### nnet architecture ###
        h1_ = T.tanh(T.dot(x, W1_) + b1_)
        nnet_out = T.nnet.softmax(T.dot(h1_, W2_) + b2_)
        
        y = T.ivector('y')
        self.x_ = x
        self.y_ = y


        ## define document probability 
        #doc_prob = T.sum(T.log(nnet_out)[:,1])
        #self.doc_prob = theano.function([x], doc_prob)

        pred = T.argmax(nnet_out, axis=1)
        #self.pred_ = pred
        self.predict = theano.function([self.idxs1, self.idxs2], pred)        
        err = T.mean(T.neq(y, pred))
        self.err = theano.function([x, y], err)


        # Define cost, gradients, and update


        cost = -T.mean(T.log(nnet_out)[T.arange(y.shape[0]), y])
        
        dW1 = T.grad(cost, W1_)
        db1 = T.grad(cost, b1_)
        dW2 = T.grad(cost, W2_)
        db2 = T.grad(cost, b2_)
        updates = [(W1_, W1_ - self.learning_rate_ * dW1),
                   (b1_, b1_ - self.learning_rate_ * db1),
                   (W2_, W2_ - self.learning_rate_ * dW2),
                   (b2_, b2_ - self.learning_rate_ * db2),]
        self.cost_ = cost
        self.updates_ = updates



    def fit(self, X_train, y_train):
        test_vec_tm1 = X_train[0][0].reshape((X_train[0][0].shape[0], 1))
        test_vec_t = X_train[0][1].reshape((X_train[0][1].shape[0], 1))
        test_vec_tp1 = X_train[0][2].reshape((X_train[0][2].shape[0], 1))
        #print theano.function([self.idxs_], self.x_)(test_vec).shape
        h = theano.function(
            [self.idxs1_, self.idxs2_, self.idxs3_], 
            [self.s1_, self.s2_, self.s3_])(
                test_vec_tm1, test_vec_t, test_vec_tp1)
        return h

class RNNModel(object):
    def __init__(self, embedding, max_sent_len, hidden_dim=100, window_size=3, 
                 learning_rate=.01, max_iters=50, batch_size=25, lam=.01,
                 fit_init=False, fit_embeddings=False):
                 
        self.max_sent_len = max_sent_len
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.lam = lam
        self._embedding = embedding
        self.pad = embedding.get_index("__PAD__")
        self.word_dim = embedding.embed_dim
        self.window_size = window_size
        self.fit_embeddings_ = fit_embeddings
        self.fit_init_ = fit_init
        
        self.init_params()
        self.build_network()
        
    def init_params(self):    

        self.sym_vars = {
            "input": T.imatrix(name="input"),
            "y gold": T.ivector(name="y gold"),
            "mask": T.imatrix(name="mask")}
        
        
        self.params = {}
        
        # Number of params for W_rec + V_rec + b_rec
        n_recurrent_params = self.word_dim**2 + self.word_dim**2 + \
            self.word_dim
        
        # If we are fitting h_0, allocate self.word_dim params for this 
        # variable.
        if self.fit_init_ is True:
            n_recurrent_params += self.word_dim
        
        print "n_recurrent_params:", n_recurrent_params
        
        # Number of params for W1_s1, W1_s2, ... W1_s(window_size) and b1,
        # and W2 + b2
        n_feedforward_params = self.hidden_dim * self.word_dim * \
            self.window_size + self.hidden_dim + 2 * self.hidden_dim + 2
        
        print "n_feedforward_params:", n_feedforward_params
        
        n_params = n_recurrent_params + n_feedforward_params
        
        print "n_params:", n_params
        
        self.theta = theano.shared(
            value=np.zeros(n_params,
                dtype=theano.config.floatX),
            name="theta", borrow=True)
        
        current_pointer = 0
        inits = []
         
        # Initialize RNN component
        
        # Init W_rec (word_dim x word_dim)
        next_pointer = current_pointer + self.word_dim**2
        W_rec = self.theta[current_pointer:next_pointer].reshape(
            (self.word_dim, self.word_dim))
        W_rec.name = "W_rec"
        self.params[W_rec.name] = W_rec
        
        W_rec_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(self.word_dim, self.word_dim)).astype(
                theano.config.floatX)
        inits.append(W_rec_init)
        current_pointer = next_pointer
        
        # Init V_rec (word_dim x word_dim)
        next_pointer = current_pointer + self.word_dim**2
        V_rec = self.theta[current_pointer:next_pointer].reshape(
            (self.word_dim, self.word_dim))
        V_rec.name = "V_rec"
        self.params[V_rec.name] = V_rec
        
        V_rec_init = np.random.uniform(
            low=-.2, high=.2, 
            size=(self.word_dim, self.word_dim)).astype(
                theano.config.floatX)
        inits.append(V_rec_init)
        current_pointer = next_pointer
        
        # Init b_rec (word_dim)
        next_pointer = current_pointer + self.word_dim
        b_rec = self.theta[current_pointer:next_pointer].reshape(
            (self.word_dim,))
        b_rec.name = "b_rec"
        self.params[b_rec.name] = b_rec
        
        b_rec_init = np.zeros(self.word_dim, dtype=theano.config.floatX)
        inits.append(b_rec_init)
        current_pointer = next_pointer       
        
        if self.fit_init_ is True:
            next_pointer = current_pointer + self.word_dim
            h_0 = self.theta[current_pointer:next_pointer].reshape(
                (self.word_dim,))
            h_0.name = "h_0"
            self.params[h_0.name] = h_0   
            
            h_0_init = np.random.uniform(
                low=-.2, high=.2,
                size=(self.word_dim,)).astype(theano.config.floatX)
            inits.append(h_0_init)
            current_pointer = next_pointer 
        else:
            h_0 = theano.shared(
                value=np.zeros(self.word_dim, dtype=theano.config.floatX),
                name="h_0", 
                borrow=True)
            self.params[h_0.name] = h_0
        
        # Initialize Feed-Forward NN component
        
        # Init W1_s...
        for i in xrange(self.window_size):
            next_pointer = current_pointer + self.hidden_dim * self.word_dim
            W1_s = self.theta[current_pointer:next_pointer].reshape(
                (self.word_dim, self.hidden_dim))
            W1_s.name="W1_s{}".format(i)
            self.params[W1_s.name] = W1_s
            
            W1_s_init = np.random.uniform(
                low=-.2, high=.2, 
                size=(self.word_dim, self.hidden_dim)).astype(
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
        
        E = theano.shared(self._embedding.W.astype(theano.config.floatX),
            name="E", borrow=True)
        self.params[E.name] = E

    def build_network(self):
        
        self._debug_funcs = {}
        
        win_size = theano.shared(
            value=np.ones(1) * self.window_size, name="win_size", borrow=True)
        II = self.sym_vars["input"]
        E = self.params["E"]    
        M = self.sym_vars["mask"]
        y_gold = self.sym_vars["y gold"]
        
        MAX_LEN = self.max_sent_len
        
        # Index into word embedding matrix and construct a 
        # 3d tensor for input. After dimshuffle, dimensions should
        # be (MAX_LEN * window_size, batch_size, word_dim).
        x = E[II] 
        x = x.dimshuffle(1,0,2)
        
        self._debug_funcs["x"] = theano.function([II], x)  
        
        tensor_mask = M.dimshuffle(1,0).reshape((x.shape[0], x.shape[1], 1))
        outputs_info = [
            T.alloc(self.params["h_0"], x.shape[1], self.word_dim)] * \
            self.window_size

        rnn_layer, _ = theano.scan(
            fn=self.step_mask,
            sequences=[{"input":x, "taps":[0, MAX_LEN, 2 * MAX_LEN]}, 
                       {"input": tensor_mask, "taps":[0, MAX_LEN, 2 * MAX_LEN]}
                      ], 
            outputs_info=outputs_info,
            n_steps=MAX_LEN)
        rnn_layer = [o[-1] for o in rnn_layer]
        self._rnn_layer = rnn_layer
        self._debug_funcs["rnn layer"] = theano.function([II, M], rnn_layer)
        a1 = self.params["b1"]
        for i in xrange(self.window_size):
            a1 += T.dot(rnn_layer[i], self.params["W1_s{}".format(i)])
        h1 = T.tanh(a1)
        self._h1 = h1
        
        self._nn_output = T.nnet.softmax(
            T.dot(h1, self.params["W2"]) + self.params["b2"])
        self._nn_output_func = theano.function([II, M], self._nn_output)

        self._nll = -T.mean(
            T.log(self._nn_output)[T.arange(y_gold.shape[0]), y_gold])      
        reg = (self.params["W_rec"]**2).sum() + \
              (self.params["W1_s1"]**2).sum() + \
              (self.params["W1_s2"]**2).sum() + \
              (self.params["W1_s0"]**2).sum() + \
              (self.params["W2"]**2).sum()
              #(self.params["V_rec"]**2).sum() + \
        self._reg = reg
        #self._cost = self._nll #+ self.lam * reg
        
    def step_mask(self, x1_t, x2_t, x3_t, 
                  mask1, mask2, mask3,
                  h1_tm1, h2_tm1, h3_tm1):

        h1_t, h2_t, h3_t = self.step(x1_t, h1_tm1, x2_t, h2_tm1, x3_t, h3_tm1)
        h1_t = mask1 * h1_t + (1 - mask1) * h1_tm1
        h2_t = mask2 * h2_t + (1 - mask2) * h2_tm1
        h3_t = mask3 * h3_t + (1 - mask3) * h3_tm1
        return h1_t, h2_t, h3_t
    
    def step(self, x1_t, h1_tm1, x2_t, h2_tm1, x3_t, h3_tm1):
        W_rec = self.params["W_rec"]
        V_rec = self.params["V_rec"]
        b_rec = self.params["b_rec"]
        h1_t = T.tanh(T.dot(h1_tm1, V_rec) + T.dot(x1_t, W_rec) + b_rec)
        h2_t = T.tanh(T.dot(h2_tm1, V_rec) + T.dot(x2_t, W_rec) + b_rec)
        h3_t = T.tanh(T.dot(h3_tm1, V_rec) + T.dot(x3_t, W_rec) + b_rec)
        return h1_t, h2_t, h3_t  
    
    def predict_prob_windows(self, IX):
        mask = np.ones_like(IX)
        mask[IX == self.pad] = 0
        return self._nn_output_func(IX, mask)


    def fit(self, X_train, y_train, X_dev=None, y_dev=None, 
            X_gold=None, X_perm=None):
        mask = np.ones_like(X_train)
        mask[X_train == self.pad] = 0
        
        if X_dev is not None:
            mask_dev = np.ones_like(X_dev)
            mask_dev[X_dev == self.pad] = 0


        X_shared = theano.shared(X_train.astype(np.int32),
            name="X_shared",
            borrow=True)
        y_shared = theano.shared(y_train.astype(np.int32),
            name="y_shared",
            borrow=True)
        mask_shared = theano.shared(mask.astype(np.int32),
            name="mask_shared",
            borrow=True)
   
        y = self.sym_vars["y gold"]
        II = self.sym_vars["input"]
        M = self.sym_vars["mask"]
        
        adagrad = theano.shared(
            np.zeros(self.theta.get_value(borrow=True).shape[0],
                dtype=theano.config.floatX),
            name="adagrad",
            borrow=True)

        n_batches = X_shared.get_value(borrow=True).shape[0] / self.batch_size
        cost = self._nll + self.lam * self._reg / (2. * II.shape[0])
        gtheta = T.grad(cost, self.theta)

        # diagonal adagrad update
        adagrad += gtheta**2
        delta = self.learning_rate * (gtheta / T.sqrt(adagrad))
        updates = [(self.theta, self.theta - self.learning_rate * gtheta)]
        
        
        index = T.scalar(dtype='int32')  # index to a [mini]batch
        train_model = theano.function(
            inputs=[index],
            outputs=[cost, gtheta, adagrad],
            updates=updates,
            givens={
                II: X_shared[
                    index * self.batch_size : (index+1) * self.batch_size],
                y: y_shared[
                    index * self.batch_size : (index+1) * self.batch_size],
                M: mask_shared[
                    index * self.batch_size : (index+1) * self.batch_size],
            }
        )
        
        pred = T.argmax(self._nn_output, axis=1)
        err = T.mean(T.neq(y, pred))
        avg_err = theano.function([II, y, M], err)

        for n_iter in xrange(self.max_iters):
            print "iter", n_iter
            for i in xrange(n_batches):
                train_model(i)
            print avg_err(X_train, y_train, mask)
            if X_dev is not None and y_dev is not None:
                print "avg err", avg_err(X_dev, y_dev, mask_dev)
            if X_gold is not None and X_perm is not None:
                rank_err = 0
                total = 0
                for x_gold, x_perm in izip(X_gold, X_perm):
                    lp_gold = np.log(
                        self.predict_prob_windows(x_gold)[:,1]).sum()
                    lp_perm = np.log(
                        self.predict_prob_windows(x_perm)[:,1]).sum()
                    if lp_gold > lp_perm:
                        rank_err += 1
                    total += 1
                print "DEV DOC ERR", float(rank_err) / total
