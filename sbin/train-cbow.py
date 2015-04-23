import sys
import os

from cohere.nnet import TokensTransformer, WordEmbeddings, CBOWModel
import cohere.data
import numpy as np
from itertools import product
from sklearn.cross_validation import KFold
import pandas as pd


def make_df(results):
    df = pd.DataFrame(results)
    df = df.set_index(["model no.", "iter"])
    df2 = df.groupby(level=["model no.", "iter"])
    df3 = df2[["train err", "dev acc"]].mean()
    df3.columns = ["train err", "dev acc"]
    df3[["lam", "win", "batch", "lr", "clean"]] = \
        df2[["lam", "win", "batch", "lr", "clean"]].first()
    df3 = df3.reset_index()
    df3.sort("dev acc", inplace=True)
    return df3

def result2path(dir, result):
    template = "mdl_no.{}.fold.{}.iter.{}.lr.{}.win.{}." + \
        "lam.{}.batch.{}{}.pkl"
    fname = template.format(
        int(result["model no."]), int(result["fold"]), int(result["iter"]), 
        float(result["lr"]), int(result["win"]), float(result["lam"]),
        int(result["batch"]), ".clean" if result["clean"] else "")
    return os.path.join(dir, fname)

def main(output_model_dir, corpus, clean):
    print "clean?", clean   
    print "Loading training data..."
    D_P = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="tokens", clean=clean, 
        convert_brackets=False)
    
    D = [dp["gold"] for dp in D_P]

    print "Loading testing data..."
    D_P_test = cohere.data.get_barzilay_data(
        corpus=corpus, part="test", format="tokens", clean=clean, 
        convert_brackets=False)
    all_docs = [dp["gold"] for dp in D_P] + [dp["gold"] for dp in D_P_test]

    embed_path = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), 
        "{}_embeddings.txt.gz".format(corpus))
    print "Reading word embeddings from {} ...".format(embed_path)
    embed = WordEmbeddings.from_file(embed_path)
    
    max_sent_len = TokensTransformer.get_max_sent_len(all_docs)

    results = []

    #max_sent_len = 115
    n_data = len(D)

    max_folds = 10
    max_iters = 20
    learning_rates = [0.01,]

    batch_sizes = [25, ]
    window_sizes = [3, 5, 7]
    lambdas = [0.01, 0.1, 0.25, 0.5, 1.0, 1.25, 2.0, 2.5, 5.0]

    n_settings = len(batch_sizes) * len(window_sizes) * len(lambdas) * \
        len(learning_rates)
    params = product(lambdas, window_sizes, batch_sizes, learning_rates)

    for n_setting, (lam, win_size, batch_size, lr) in enumerate(params, 1):
        np.random.seed(1986)
        print "{}/{}:".format(n_setting, n_settings),
        print "lambda:", lam, "window size:", win_size,
        print "batch size:", batch_size, "lr:", lr


        
        transformer = TokensTransformer(
            embed, max_sent_len=max_sent_len, window_size=win_size)

        folds = KFold(n_data, n_folds=max_folds)
        for n_fold, (I_train, I_dev) in enumerate(folds, 1):
            D_train = [D[i] for i in I_train]
            D_P_dev = [D_P[i] for i in I_dev]
            
            X_iw_gold = []
            X_iw_perm = []
            for d_P in D_P_dev:
                x_iw_gold = transformer.window_transform([d_P[u"gold"]])
                for p in d_P[u"perms"]:
                    x_iw_perm =transformer.window_transform([p])
                    X_iw_gold.append(x_iw_gold)
                    X_iw_perm.append(x_iw_perm)
            X_iw, y = transformer.training_window_transform(D_train)

            def fit_callback(nnet, n_iter):
                avg_win_err = -1.
                dev_avg_win_err = -1.
                dev_acc = nnet.score(X_iw_gold, X_iw_perm)
                result = {"fold": n_fold, "model no.": n_setting,
                          "iter": n_iter,
                          "train err": avg_win_err,
                          "dev err": dev_avg_win_err,
                          "dev acc": dev_acc,
                          "lam": lam, "win": win_size,
                          "batch": batch_size,
                          "lr": lr, "clean": clean,
                          }
        
                pkl_path = result2path(output_model_dir, result)
                nnet.save(pkl_path)
                print "iter {:3d} | train win err: {:0.3f}".format(
                    n_iter, avg_win_err),
                print " | dev win err: {0:.3f}".format(dev_avg_win_err),
                print " | dev acc: {0:.3f}".format(dev_acc)
                results.append(result)

            nnet = CBOWModel(embed, max_sent_len=max_sent_len, 
                alpha=lr,
                batch_size=batch_size, lam=lam, window_size=win_size,
                max_iters=max_iters, fit_callback=fit_callback)

            nnet.fit(X_iw, y)

        print "TOP 5"
        df = make_df(results)
        print df.tail(5)

    best_params = df.tail(1).to_dict(orient="records")[0]
    thetas_sum = 0
    for fold in xrange(1, max_folds + 1):
        best_params["fold"] = fold
        path = result2path(output_model_dir, best_params)
        nnet.load(path)
        thetas_sum += nnet.theta.get_value(borrow=True)
    thetas_avg = thetas_sum / float(max_folds)
    nnet.theta.set_value(thetas_avg)

    transformer = TokensTransformer(
        embed,
        max_sent_len=max_sent_len, window_size=int(best_params["win"])) 
    X_iw_gold = []
    X_iw_perm = []
    for d_P in D_P_test:
        x_iw_gold = transformer.window_transform([d_P[u"gold"]])
        for p in d_P[u"perms"]:
            x_iw_perm =transformer.window_transform([p])
            X_iw_gold.append(x_iw_gold)
            X_iw_perm.append(x_iw_perm)

    test_acc = nnet.score(X_iw_gold, X_iw_perm)
    print "Best Model"
    print df.tail(1)
    fname = "results_clean.csv" if clean else "results.csv"
    with open(os.path.join(output_model_dir, fname), "w") as f:
        df.to_csv(f)
    print "Test Acc", test_acc

if __name__ ==  u"__main__":
    args = sys.argv
    assert len(args) >= 2
    assert args[1] in ["ntsb", "apws"]
    if args[1] == "ntsb": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "ntsb.cbow")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if args[1] == "apws": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "apws.cbow")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if len(args) == 3 and args[2] == "clean":
        clean = True
    else:
        clean = False

    main(output_dir, args[1], clean)
