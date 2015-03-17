import random
import numpy as np

def positive_window_gen(doc):
    fmt = lambda x: u"{}".format(unicode(x).lower())
    n_sents = len(doc)
    for i in xrange(n_sents):
        if i == 0:

            yield [u"__START__",
                   doc.sents[i].format_tokens(fmt=fmt),
                   doc.sents[i + 1].format_tokens(fmt=fmt)]
        elif i + 1 == n_sents:
            yield [doc.sents[i - 1].format_tokens(fmt=fmt),
                   doc.sents[i].format_tokens(fmt=fmt),
                   u"__STOP__"]
        else:
            yield [doc.sents[i - 1].format_tokens(fmt=fmt),
                   doc.sents[i].format_tokens(fmt=fmt),
                   doc.sents[i + 1].format_tokens(fmt=fmt)]

def negative_window_gen(doc):
    fmt = lambda x: u"{}".format(unicode(x).lower())
    n_sents = len(doc)
    for i in xrange(n_sents):
        bad_indices = set([i-1, i, i + 1])
        rand_index = range(0, n_sents)
        random.shuffle(rand_index)
        while rand_index[-1] in bad_indices:
            rand_index.pop()
        rand_index = rand_index[-1]

        #print i-1, rand_index, i+1
        if i == 0:
            yield [u"__START__",
                   doc.sents[rand_index].format_tokens(fmt=fmt),
                   doc.sents[i + 1].format_tokens(fmt=fmt)]
        elif i + 1 == n_sents:
            yield [doc.sents[i - 1].format_tokens(fmt=fmt),
                   doc.sents[rand_index].format_tokens(fmt=fmt),
                   u"__STOP__"]
        else:
            yield [doc.sents[i - 1].format_tokens(fmt=fmt),
                   doc.sents[rand_index].format_tokens(fmt=fmt),
                   doc.sents[i + 1].format_tokens(fmt=fmt)]

def docs2train_seq(docs, embedding):
    y_docs = []
    X_docs = []
    for doc in docs:
        if len(doc) <= 3:
            continue
        pos_seq = embedding.doc2seq(doc, positive=True) 
        neg_seq = embedding.doc2seq(doc, positive=False)
        y_docs.extend([1] * len(pos_seq)) 
        y_docs.extend([0] * len(neg_seq))
        X_docs.extend(pos_seq)
        X_docs.extend(neg_seq)
    y_docs = np.array(y_docs)
    rnd = range(y_docs.shape[0])
    random.shuffle(rnd)
    X_docs = [X_docs[i] for i in rnd]
    y_docs = y_docs[rnd]
    return X_docs, y_docs.astype(np.int32).ravel()
    

def docs2trainvecs(docs, embedding):
    y_docs = []
    X_docs = []
    for doc in docs:
        if len(doc) <= 3:
            continue
        pos_mat = embedding.doc2mat(doc, positive=True)
        neg_mat = embedding.doc2mat(doc, positive=False)
        y_doc = np.vstack(
            [np.ones((pos_mat.shape[0], 1)), np.zeros((neg_mat.shape[0], 1))])
        X_doc = np.vstack([pos_mat, neg_mat])

        y_docs.append(y_doc)
        X_docs.append(X_doc)
    #print X_docs
    X_docs = np.vstack(X_docs)
    y_docs = np.vstack(y_docs)
    rnd = range(X_docs.shape[0])
    random.shuffle(rnd)
    X_docs = X_docs[rnd, :]
    y_docs = y_docs[rnd]
    return X_docs, y_docs.astype(np.int32).ravel()
            
