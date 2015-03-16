import cohere.data
import cohere.structs
import cohere.embed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import os

def main(output_model_path):
    embedding = cohere.embed.StaticGloVeEmbeddings()
    train_docs = cohere.data.get_barzilay_ntsb_clean_docs_only("train")
    dev_docs = cohere.data.get_barzilay_ntsb_clean_docs_only("dev")
    print "loaded"
    X_train, y_train = cohere.structs.docs2trainvecs(train_docs, embedding) 
    X_dev, y_dev = cohere.structs.docs2trainvecs(dev_docs, embedding) 

    print "Training"
    clf = LogisticRegression(C=.1, penalty='l2')
    best_dev_acc = 0.0
    best_clf = None
    best_c = None
    for c in [10**i for i in range(-4, 5)]:
        clf = LogisticRegression(C=c, penalty='l2')
        clf.fit(X_train, y_train)
    
        y_pred_train = clf.predict(X_train)
        y_pred_dev = clf.predict(X_dev)
        train_acc = accuracy_score(y_train, y_pred_train) 
        dev_acc = accuracy_score(y_dev, y_pred_dev) 
        print "C: {:0.3f} Train Acc: {:0.5f} Dev Acc: {:0.5f}".format(
            c, train_acc, dev_acc)
        if dev_acc > best_dev_acc:
            best_clf = clf
            best_c = c
            best_dev_acc = dev_acc        
            best_train_acc = train_acc        
        
    print "###BEST CLF###", 
    print "C: {:0.6f} Train Acc: {:0.5f} Dev Acc: {:0.5f}".format(
        best_c, best_train_acc, best_dev_acc)
    
    joblib.dump(best_clf, output_model_path, compress=9)

if __name__ == u"__main__":
    output_dir = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), "models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "baseline-logreg.pkl.gz")

    main(output_path)


