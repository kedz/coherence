import os
import cohere.models
import cohere.data
import cohere.structs
import cohere.embed
import numpy as np
from sklearn.externals import joblib

def main(output_model_path):
    embedding = cohere.embed.StaticGloVeEmbeddings()
    train_docs = cohere.data.get_barzilay_ntsb_clean_docs_only("train")
    X_train, y_train = cohere.structs.docs2trainvecs(train_docs, embedding) 
    dev_docs = cohere.data.get_barzilay_ntsb_clean_docs_only("dev")
    X_dev, y_dev = cohere.structs.docs2trainvecs(dev_docs, embedding) 

    best_nnet = None
    best_dev_err = 1
    best_train_err = None
    best_lr = None
    for lr in np.linspace(0.00,.3, 20):
        nnet = cohere.models.CBOWModel(learning_rate=lr, max_iters=100)
        nnet.fit(X_train, y_train)
        train_err = nnet.err(X_train, y_train)
        dev_err = nnet.err(X_dev, y_dev)
        print "lr: {} Train Error: {:0.3f} Dev Error {:0.3f}".format(
            lr, float(train_err), float(dev_err))
        if dev_err < best_dev_err:
            best_dev_err = dev_err
            best_train_err = train_err
            best_lr = lr
            best_nnet = nnet
    

    print "###BEST NNET###", 
    print "lr: {:0.6f} Train Err: {:0.5f} Dev Err: {:0.5f}".format(
        best_lr, float(best_train_err), float(best_dev_err))
    
    joblib.dump(best_nnet, output_model_path, compress=9)


if __name__ ==  u"__main__":
    output_dir = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), "models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "avgvec-nnet.pkl.gz")

    main(output_path)
