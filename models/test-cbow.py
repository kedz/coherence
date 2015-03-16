import os
from sklearn.externals import joblib
import cohere.embed
import cohere.data
import numpy as np
import cohere.models

def main(model_path):

    embedding = cohere.embed.StaticGloVeEmbeddings()
    test_inst = cohere.data.get_barzilay_ntsb_clean_docs_perms("test")
    nnet = joblib.load(model_path)
    
    correct = 0
    total = 0
    for inst in test_inst:
        gold_doc = inst[u"gold"]
        gold_mat = embedding.doc2mat(gold_doc, positive=True)
        gold_lp = nnet.doc_prob(gold_mat)
        for perm_doc in inst[u"perms"]:
            perm_mat = embedding.doc2mat(perm_doc, positive=True)
            perm_lp = nnet.doc_prob(perm_mat)
            if gold_lp > perm_lp:
                correct += 1
            total += 1
            print float(correct) / total
    acc = float(correct) / total
    print "CBOW NNet ACC:", acc






if __name__ == u"__main__":
    model_dir = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), "models")
    model_path = os.path.join(model_dir, "avgvec-nnet.pkl.gz")

    main(model_path)


