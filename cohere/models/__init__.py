import theano
import theano.tensor as T
import numpy as np

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

#        X_train = X_train[indices, :]
#        y_train = y_train[indices]

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




