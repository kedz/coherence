import sys
import os
import argparse

from cohere.nnet import TokensTransformer, WordEmbeddings, RecurrentNNModel
import multiprocessing
import signal
import Queue

import cohere.data
import numpy as np
from itertools import product
from sklearn.cross_validation import KFold
import pandas as pd


def make_df(results):
    df = pd.DataFrame(results)
    df = df.set_index(["model no.", "iter"])
    df2 = df.groupby(level=["model no.", "iter"])
    df3 = df2[["dev acc"]].mean()
    df3.columns = ["dev acc"]
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

def get_averaged_model(params, embed, max_sent_len, max_folds,
                       output_model_dir):
    nnet = RecurrentNNModel(embed, max_sent_len=max_sent_len, 
                alpha=params["lr"],
                lam=params["lam"], 
                window_size=int(params["win"]))

    thetas_sum = 0
    tot_models = 0
    for fold in xrange(1, max_folds + 1):
        params["fold"] = fold
        path = result2path(output_model_dir, params)
        if not os.path.exists(path):
            continue
        nnet.load(path)
        thetas_sum += nnet.theta.get_value(borrow=True)
        tot_models += 1
    thetas_avg = thetas_sum / float(tot_models)

    nnet = RecurrentNNModel(embed, max_sent_len=max_sent_len, 
                alpha=params["lr"],
                lam=params["lam"], 
                window_size=int(params["win"]))
    nnet.theta.set_value(thetas_avg)
    return nnet


def main(output_model_dir, embed_path, corpus, clean, n_procs=4, max_iters=10):

    max_folds = 10
    learning_rates = [0.01,]

    folds = range(max_folds)
    batch_sizes = [25, ]
    window_sizes = [3, 5, 7]
    lambdas = [0.1, 0.25, 0.5, 1.0, 1.25, 2.0, 2.5, 5.0]

    n_settings = len(batch_sizes) * len(window_sizes) * len(lambdas) * \
        len(learning_rates)
    params = product(lambdas, window_sizes, batch_sizes, learning_rates)

    mngr = multiprocessing.Manager()
    jobs = mngr.Queue()
    result_queue = mngr.Queue()
    for n_setting, (lam, win_size, batch_size, alpha) in enumerate(params, 1):
        for fold in folds:
            jobs.put(        
                (n_setting, fold, alpha, lam, win_size, batch_size,))

    results = []
    pool = []
    for i in xrange(n_procs):
        p = multiprocessing.Process(target=worker, args=(jobs, result_queue),
            kwargs={"max folds": max_folds, "max iters": max_iters,
                    "corpus": corpus, "clean": clean,
                    "output model dir": output_model_dir})
        p.start()
        pool.append(p)
    max_jobs = n_settings * max_folds

    try:
        for n_job in xrange(max_jobs * max_iters):
            result = result_queue.get(block=True)
            print result
            results.append(result)

            print "TOP 5"
            df = make_df(results)
            print df.tail(5)

        best_params = df.tail(1).to_dict(orient="records")[0]

        print "Loading testing data..."
        test_data = cohere.data.get_barzilay_data(
            corpus=corpus, part="test", format="tokens", clean=clean, 
            convert_brackets=True)

        embed = WordEmbeddings.li_hovy_embeddings(corpus)
        max_sent_len = TokensTransformer.get_max_sent_len(test_data.gold)

        nnet = get_averaged_model(best_params, embed, max_sent_len, max_folds,
                                  output_model_dir)

        transformer = TokensTransformer(
            embed,
            max_sent_len=max_sent_len, window_size=int(best_params["win"])) 

        X_test_gold, X_test_perm = transformer.testing_window_transform(
            test_data)

        test_acc = nnet.score(X_test_gold, X_test_perm)
        print "Best Model"
        print df.tail(1)
        fname = "results_clean.csv" if clean else "results.csv"
        with open(os.path.join(output_model_dir, fname), "w") as f:
            df.to_csv(f)
        print "Test Acc", test_acc

        for p in pool:
            p.join()

    except KeyboardInterrupt:
        print "Completing current jobs and shutting down!"
        while not jobs.empty():
            jobs.get()
        for p in pool:
            p.join()
        sys.exit()

def worker(job_queue, result_queue, **kwargs):

    max_folds = kwargs.get(u'max folds')
    max_iters = kwargs.get(u'max iters')
    corpus = kwargs.get(u'corpus')
    clean = kwargs.get(u'clean')
    output_model_dir = kwargs.get(u'output model dir')

    print "Loading {}{} training data...".format(
        "clean " if clean else "", corpus)
          
    dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="tokens", clean=clean, 
        convert_brackets=True)
    
    embed = WordEmbeddings.li_hovy_embeddings(corpus)
   
    max_sent_len = TokensTransformer.get_max_sent_len(dataset.gold)
    n_data = len(dataset)

    while not job_queue.empty():
        try:
            (n_setting, n_fold, alpha, lam, 
             win_size, batch_size,) = job_queue.get(block=False)

            transformer = TokensTransformer(
                embed, max_sent_len=max_sent_len, window_size=win_size)

            folds = KFold(n_data, n_folds=max_folds)
            for k, (I_train, I_dev) in enumerate(folds):
                if k != n_fold:
                    continue

                train_data = dataset[I_train]
                dev_data = dataset[I_dev]
                 
                X_iw, y = transformer.training_window_transform(train_data.gold)
                X_dev_gold, X_dev_perm = transformer.testing_window_transform(
                    dev_data)

                def fit_callback(nnet, n_iter):

                    dev_acc = nnet.score(X_dev_gold, X_dev_perm)

                    result = {"fold": n_fold, "model no.": n_setting,
                              "iter": n_iter,
                              "dev acc": dev_acc,
                              "lam": lam, "win": win_size,
                              "batch": batch_size,
                              "lr": alpha, "clean": clean,
                              }
            
                    pkl_path = result2path(output_model_dir, result)
                    nnet.save(pkl_path)
                    result_queue.put(result)

                np.random.seed(1986)
                nnet = RecurrentNNModel(embed, max_sent_len=max_sent_len, 
                    alpha=alpha,
                    batch_size=batch_size, lam=lam, window_size=win_size,
                    max_iters=max_iters, fit_callback=fit_callback)

                nnet.fit(X_iw, y)

        except Queue.Empty:
            pass

if __name__ ==  u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c',
                        help='corpus to train on', required=True, 
                        choices=["apws", "ntsb"])
    parser.add_argument("--clean", action="store_true", default=False,
                        help="use the clean version of the corpus")
    parser.add_argument("--max-iters", default=10, type=int,
                        help="max training iterations")
    parser.add_argument("--n-procs", default=1, type=int,
                        help="number of processes to use") 

  
    args = parser.parse_args()
    corpus = args.corpus
    clean = args.clean
    max_iters = args.max_iters
    n_procs = args.n_procs
    
    print corpus
    if corpus == "ntsb": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "ntsb.recurrent")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    elif corpus == "apws": 
        output_dir = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), "models", "apws.recurrent")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    embed_path = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), 
        "{}{}_embeddings.txt.gz".format("clean_" if clean else "", corpus))
 
    main(output_dir, embed_path, corpus, clean, 
        max_iters=max_iters, n_procs=n_procs)
