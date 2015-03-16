from cohere.structs import negative_window_gen, positive_window_gen
import os
import gzip
import numpy as np

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

    def doc2mat(self, doc, positive=True):
        if positive is True:
            win_gen = positive_window_gen
        else: 
            win_gen = negative_window_gen
        windows = [w for w in win_gen(doc)]
        mat = np.array([self.window2vec(w) for w in windows])
        return mat

