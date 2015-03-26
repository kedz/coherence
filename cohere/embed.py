from cohere.structs import negative_window_gen, positive_window_gen
import os
import gzip
import numpy as np
import random

class WordEmbeddings(object):
    def __init__(self, token2index, embeddings):
        self.vocab = token2index.keys()
        self.vocab.sort(key=lambda x: token2index[x])
        self.vocab_size = embeddings.shape[0]
        self.embed_dim = embeddings.shape[1]
        self.token2index = token2index
        self.index2token = {value: key for key, value in token2index.items()}
        self.W = embeddings
        assert len(self.vocab) == embeddings.shape[0]

    def __contains__(self, token):
        return token in self.token2index

    def get_index(self, token):
        if token not in self.token2index:
            token = u"__UNKNOWN__"
        return self.token2index[token]

    def get_indices(self, tokens):
        return [self.get_index(token) for token in tokens]

    def get_embedding(self, token):
        if token not in self.token2index:
            raise Exception("Bad index")
        index = self.token2index[token]
        return self.W[index,:]

    def get_words(self, indices):
        return [self.index2token.get(index, "__UNKNOWN__") 
                for index in indices]


class GloVeEmbeddings(WordEmbeddings):
    def __init__(self, embedding_path=None):

        # Find embedding path and attempt to load it. Raise an exception
        # if GloVe word embedding file is not found.
        if embedding_path is None:
            embedding_path = os.path.join(
                os.getenv("COHERENCE_DATA", "data"), "embeddings", 
                "glove.6B.50d.txt.gz")
        if not os.path.exists(embedding_path):
            raise Exception("{} does not exist!".format(embedding_path))

        # Read in embeddings and map tokens to indices in the embedding 
        # matrix.
        embed = []
        current_index = 0
        token2index = {}
        with gzip.open(embedding_path, u"r") as f:
            for line in f:
                line = line.strip().split(" ")
                word = line.pop(0)
                embed.append([float(x) for x in line])
                token2index[word] = current_index
                current_index += 1
        embed = np.array(embed, dtype=np.float64)

        # Finally call parent constructor.
        super(GloVeEmbeddings, self).__init__(token2index, embed)


class BorrowingWordEmbeddings(WordEmbeddings):
    def __init__(self, new_vocab, bootstrap_embedding, new_vector_func=None):
        """A word embedding class for an embedding that is initialized
        from an existing embedding. `new_vector_func` takes the
        dimensionality of the word vectors and a word token 
        as arguments and returns an
        initialized row vector for words that are not covered by the 
        `bootstrap_embedding`."""
        
        if new_vector_func is None:
            def nvf(dim, token):
                #if token != "__PAD__":
                return np.zeros((dim,), dtype=np.float64)
                #else:
                #    v = np.zeros((dim,), dtype=np.float64)
                #    v.fill(float("nan"))
                #    return v
            new_vector_func = nvf

        new_vocab.add("__START__")
        new_vocab.add("__STOP__")
        new_vocab.add("__UNKNOWN__")
        new_vocab.add("__PAD__")
        new_vocab = list(new_vocab)
        new_vocab.sort()
        vocab_size = len(new_vocab)
        token2index = {}

        embed = []
        for token, index in zip(new_vocab, xrange(vocab_size)):
            token2index[token] = index
            if token in bootstrap_embedding:
                embed.append(bootstrap_embedding.get_embedding(token))
            else:
                embed.append(
                    new_vector_func(bootstrap_embedding.embed_dim, token))
        embed = np.array(embed)

        # Finally call parent constructor.
        super(BorrowingWordEmbeddings, self).__init__(token2index, embed)


class DebugWordEmbeddings(WordEmbeddings):
    def __init__(self, vocab):
        vocab.add("__START__")
        vocab.add("__STOP__")
        vocab.add("__UNKNOWN__")
        vocab.add("__PAD__")
        vocab = list(vocab)
        vocab.sort()
        v_size = len(vocab)
        E = np.zeros((v_size, 1), dtype=np.float64)
        token2index = {}
        for i, v in enumerate(vocab):
            E[i] = i
            token2index[v] = i
        super(DebugWordEmbeddings, self).__init__(token2index, E)

class StaticGloVeEmbeddings(object):
    def __init__(self, embedding_path=None):
        if embedding_path is None:
            embedding_path = os.path.join(
                os.getenv("COHERENCE_DATA", "data"), "embeddings", 
                "glove.6B.50d.txt.gz")
        embed = []
        current_index = 0
        vocab = {}
        with gzip.open(embedding_path, u"r") as f:
            for line in f:
                line = line.strip().split(" ")
                word = line.pop(0)
                embed.append([float(x) for x in line])
                vocab[word] = current_index
                current_index += 1

        self.embed_ = np.array(embed)
        self.vocab_ = vocab

    def window2vec(self, window):
        vec = []
        for sent in window:
            tot_words = 0
            sent_vec = np.zeros((50))
            for word in sent.split(" "):
                if word in self.vocab_:
                    sent_vec += self.embed_[self.vocab_[word],:]
                    #sent_indexes.append(vocab[word])
                    tot_words += 1
            if tot_words > 0:
                sent_vec /= tot_words
            vec.append(sent_vec)
        vec = np.hstack(vec)
        return vec

    def window2seq(self, window):
        vec = []
        for sent in window:
            index_vec = [] 
            for word in sent.split(" "):
                if word in self.vocab_:
                    index_vec.append(self.vocab_[word])
            vec.append(np.array(index_vec, dtype=np.int32))
        return vec

    def doc2mat(self, doc, positive=True):
        if positive is True:
            win_gen = positive_window_gen
        else: 
            win_gen = negative_window_gen
        windows = [w for w in win_gen(doc)]
        mat = np.array([self.window2vec(w) for w in windows])
        return mat

    def doc2seq(self, doc, positive=True):
        if positive is True:
            win_gen = positive_window_gen
        else: 
            win_gen = negative_window_gen
        windows = [w for w in win_gen(doc)]
        seqs = [self.window2seq(w) for w in windows]
        return seqs


class IndexDocTransformer(object):
    def __init__(self, embedding, start_pads=1, stop_pads=1, window_size=3,
                 max_sent_len=200):
        self.embedding = embedding
        self.start_pads = start_pads
        self.stop_pads = stop_pads
        self.window_size = window_size
        self.max_sent_len= max_sent_len

    def fit(self, docs):
        max_sent_len = 0
        for doc in docs:
            for sent in doc:
                if len(sent) > max_sent_len:
                    max_sent_len = len(sent)
        assert max_sent_len < self.max_sent_len

    def transform_test(self, docs_perms):
        IX_gold = []
        IX_perm = []
        for doc_perm in docs_perms:
            gold_doc = doc_perm["gold"]
            [gold_idoc] = self._docs2index_docs([gold_doc], 
                                                self.max_sent_len)
            ix_gold = self._index_docs2windows([gold_idoc], self.window_size, 
                self.max_sent_len, positive=True)
            perm_idocs = self._docs2index_docs(doc_perm["perms"], 
                                               self.max_sent_len)
            ix_perms = \
                [self._index_docs2windows(
                    [perm_idoc], self.window_size, self.max_sent_len, 
                    positive=True)
                 for perm_idoc in perm_idocs]
                                             
            for i, ix_perm in enumerate(ix_perms):
                IX_gold.append(ix_gold)
                IX_perm.append(ix_perm)
            
            assert len(IX_gold) == len(IX_perm)
            return IX_gold, IX_perm

    def transform(self, docs):
        index_docs = self._docs2index_docs(docs, self.max_sent_len) 
        IX_pos = self._index_docs2windows(index_docs, self.window_size, 
            self.max_sent_len, positive=True)
        
        IX_neg = self._index_docs2windows(index_docs, self.window_size, 
            self.max_sent_len, positive=False)
       
        y = np.zeros((IX_pos.shape[0] * 2), dtype=np.int32)
        y[:IX_pos.shape[0]] = 1

        X = np.vstack([IX_pos, IX_neg])

        rnd = range(IX_pos.shape[0] * 2)
        random.shuffle(rnd)
        X = X[rnd,:]
        y = y[rnd]

        return X, y

    def _docs2index_docs(self, docs, max_sent_len):
        
        start = self.embedding.get_index("__START__")
        stop = self.embedding.get_index("__STOP__")
        pad = self.embedding.get_index("__PAD__")
        II = [] 
        for doc in docs:
            I_d = []
            
            # Add start padding.
            for _ in xrange(self.start_pads):
                I_d.append([start,] + [pad] * (max_sent_len - 1))

            for sent in doc:
                I_s = self.embedding.get_indices(sent)
                I_s += [pad] * (max_sent_len - len(I_s))
                I_d.append(I_s)
            
            # Add stop padding.
            for _ in xrange(self.stop_pads):
                I_d.append([stop,] + [pad] * (max_sent_len - 1))

            II.append(np.array(I_d, dtype=np.int32))

        return II

    def _index_docs2windows(self, index_docs, size, max_sent_len, 
                            positive=True):

        # Validate window size -- must be an odd number.
        if size % 2 == 0:
            raise Exception("Window size must be an odd number.")
        
        # Window size is pad + 1 + pad. 
        pad = size / 2

        # Compute total number of windows in dataset and validate document
        # lengths. 
        doc_lengths = np.array([len(index_doc) for index_doc in index_docs],
            dtype=np.int32)
        if (doc_lengths < size).any():
            raise Exception("Window size larger than document size.")
        
        # This function returns a matrix of window vectors of dimension
        # (n_windows, n_tokens)
        n_windows = np.sum(doc_lengths - 2 * pad)
        n_tokens = size * max_sent_len
        IX = np.zeros((n_windows, n_tokens), dtype=np.int32)

        r = 0
        for index_doc in index_docs:
            if positive is True:
                for ixi in self._doc2pos_windows_gen(index_doc, size):
                    IX[r,:] = ixi
                    r += 1
            else:
                for ixi in self._doc2neg_windows_gen(index_doc, size):
                    IX[r,:] = ixi
                    r += 1
        return IX        
            
    def _doc2pos_windows_gen(self, index_doc, size):
        n_sents = len(index_doc)
        if n_sents < size:
            raise Exception("Window size larger than document size.")
        if size % 2 == 0:
            raise Exception("Window size must be an odd number.")
        indices = range(len(index_doc))
        pad = size / 2
        
        for i in xrange(pad, n_sents - pad):
            x = []
            for ii in range(i-pad, i):
                x.extend(index_doc[ii])        
            x.extend(index_doc[i]) 
            for ii in range(i + 1, i + 1 + pad):
                x.extend(index_doc[ii])        
            yield x

    def _doc2neg_windows_gen(self, index_doc, size):
        n_sents = len(index_doc)
        if n_sents < size - self.start_pads - self.stop_pads:
            raise Exception("Window size larger than document size.")
        if size % 2 == 0:
            raise Exception("Window size must be an odd number.")
        indices = range(len(index_doc))
        pad = size / 2
        
        for i in xrange(pad, n_sents - pad):
            bad_indices = set(
                range(i-pad, i) + [i] + range(i + 1, i + 1 + pad) + \
                range(self.start_pads) + \
                range(n_sents - self.stop_pads, n_sents))
            rand_index = range(0, n_sents)
            random.shuffle(rand_index)
            while rand_index[-1] in bad_indices:
                rand_index.pop()
            rand_index = rand_index[-1]

            x = []
            for ii in range(i-pad, i):
                x.extend(index_doc[ii])        
            

            x.extend(index_doc[rand_index]) 
            
            
            for ii in range(i + 1, i + 1 + pad):
                x.extend(index_doc[ii])        
            yield x

    def inverse_transform(self, IX):
        return np.array(
            [self.embedding.get_words(ix) for ix in IX], dtype=object)


