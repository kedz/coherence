import os
import cPickle as pickle
from sklearn.base import BaseEstimator

class _BaseNNModel(BaseEstimator):
    def __init__(self, embeddings, max_sent_len=150, window_size=3,
                 hidden_dim=100, alpha=.01, lam=1.25, batch_size=25,
                 max_iters=10, fit_embeddings=False, fit_callback=None,
                 update_method="adagrad", initializers=None):
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

        if update_method not in ["sgd", "adagrad"]:
            raise Exception(
                "Invalid update method. Must be 'sgd' or 'adagrad'")

        self.embeddings = embeddings
        self.max_sent_len = max_sent_len
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.alpha = alpha
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.lam = lam
        self.fit_embeddings = fit_embeddings
        self.fit_callback = fit_callback
        self.update_method = update_method
        self.initializers = initializers

        self.ready()

    def ready(self):
        self._init_params()
        self._build_network()


    def _get_params(self):
        return {key: val
                for key, val in self.get_params().items()
                if key not in  ["fit_callback", "initializers"]}

    def __getstate__(self):
        """ Return state sequence."""
        state = self._get_params()
        state['initializers'] = {name: param.eval() 
                                 for name, param in self.params.items()}
        #theta = self.theta.get_value() 
        return (state,)

    def __setstate__(self, state):
        """ Set parameters from state sequence."""
        params, = state
        self.set_params(**params)
        self.ready()
        #self.theta.set_value(theta)

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
