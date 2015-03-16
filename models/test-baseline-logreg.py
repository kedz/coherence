import os
from sklearn.externals import joblib
import cohere.embed
import cohere.data
import numpy as np

def main(model_path):

    def logprob(clf, X):
        return np.sum(clf.predict_log_proba(X)[:,1])


    embedding = cohere.embed.StaticGloVeEmbeddings()
    test_inst = cohere.data.get_barzilay_ntsb_clean_docs_perms("test")
    clf = joblib.load(model_path)
    
    correct = 0
    total = 0
    for inst in test_inst:
        gold_doc = inst[u"gold"]
        gold_mat = embedding.doc2mat(gold_doc, positive=True)
        gold_lp = logprob(clf, gold_mat)
        for perm_doc in inst[u"perms"]:
            perm_mat = embedding.doc2mat(perm_doc, positive=True)
            perm_lp = logprob(clf, perm_mat)
            if gold_lp > perm_lp:
                correct += 1
            total += 1
            print float(correct) / total
    acc = float(correct) / total
    print "Logistic Regression ACC:", acc






if __name__ == u"__main__":
    model_dir = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), "models")
    model_path = os.path.join(model_dir, "baseline-logreg.pkl.gz")

    main(model_path)


