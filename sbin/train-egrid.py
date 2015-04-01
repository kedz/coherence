import os
import sys
import cohere.data
import cohere.models.entitygrid as eg
import numpy as np
import sklearn.svm
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

def make_df(results):
    df = pd.DataFrame(results)
    #df = df.set_index(["model no."])
    df2 = df.groupby(["model no."])
    df3 = df2[["train acc", "dev acc"]].mean()
    df3.columns = ["train acc", "dev acc"]
    df3[["c"]] = \
        df2[["c"]].first()
    #df3 = df3.reset_index()
    df3.sort("dev acc", inplace=True)
    return df3

def result2path(dir, result):
    template = "mdl_no.{}.fold.{}.c.{}.pkl"
    fname = template.format(
        int(result["model no."]), int(result["fold"]), 
        float(result["c"]),)
    return os.path.join(dir, fname)


def main(output_model_dir, corpus, max_folds=10):
    
    C = [1., 2., 2.5, 10., 100., 500., 1000.]

    print C
    D_P_train = cohere.data.get_barzilay_clean_docs_perms(
        corpus=corpus, part="train") + \
        cohere.data.get_barzilay_clean_docs_perms(
            corpus=corpus, part="dev", tokens_only=False)
    
    transformer = eg.EntityGridTransformer()
    E_P_train = transformer.transform_D_P(D_P_train)
   
    vec = eg.EntityGridVectorizer()
    X, y = vec.pairwise_transform(E_P_train)

    n_settings = len(C)
    folds = KFold(X.shape[0], n_folds=max_folds)
    results = []

    for n_setting, c in enumerate(C, 1):
        print "{}/{}".format(n_setting, n_settings)
        print "c=", c
    
        for n_fold, (I_train, I_dev) in enumerate(folds, 1):
            X_train = X[I_train, :]
            y_train = y[I_train]
            X_dev = X[I_dev, :]
            y_dev = y[I_dev] 
            clf = sklearn.svm.SVC(kernel='linear', C=c)
            clf.fit(X_train, y_train)
            train_acc = clf.score(X_train, y_train)
            dev_acc = clf.score(X_dev, y_dev)
            print "fold:", n_fold, "TRAIN ACC:", train_acc, 
            print "DEV ACC:", dev_acc

            result = {"fold": n_fold, "model no.": n_setting,
                      "train acc": train_acc,
                      "dev acc": dev_acc,
                      "c": c,
                     }
        
            pkl_path = result2path(output_model_dir, result)
            print pkl_path
            joblib.dump(clf, pkl_path)
            results.append(result)
        df = make_df(results)
        print df


    df = make_df(results)
    df = df.reset_index()
    best_params = df.tail(1).to_dict(orient="records")[0]

    # Collect average thetas and intercept. We can only do this with a linear
    # svm. Get prediciton accuracy using the averaged model.
    thetas_sum = 0
    b_sum = 0
    for fold in xrange(1, max_folds + 1):
        best_params["fold"] = fold
        path = result2path(output_model_dir, best_params)
        clf = joblib.load(path)
        thetas_sum += clf.coef_ 
        b_sum += clf.intercept_

    thetas_avg = thetas_sum / float(max_folds)
    b_avg = b_sum / float(max_folds)
    
    print "Loading test data"
    D_P_test = cohere.data.get_barzilay_clean_docs_perms(
        corpus=corpus, part="test")
    E_P_test = transformer.transform_D_P(D_P_test)
    X_test, y_test = vec.pairwise_transform(E_P_test)
    scores = np.dot(X_test, thetas_avg.T) + b_avg
    indices = (scores > 0).astype(np.int32)
    y_pred = clf.classes_[indices]
    print "TEST ACC:", accuracy_score(y_test, y_pred)


if __name__ ==  u"__main__":
    args = sys.argv
    assert len(args) == 2
    assert args[1] in ["ntsb", "apws"]
    if args[1] == "ntsb": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "ntsb.egrid")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if args[1] == "apws": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "apws.egrid")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    main(output_dir, args[1])
