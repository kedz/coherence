import os
import shutil
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from mpi4py import MPI

import time

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
TAGS = enum('READY', 'DONE', 'EXIT', 'START')

def setup_theano(rank, corpus, clean):
    worker_id = "{}.{}.{}.{}".format(
        corpus, "clean" if clean else "norm",
        MPI.Get_processor_name(), rank, corpus) 
    compile_dir=".theano.{}".format(worker_id)
    os.environ["THEANO_FLAGS"] = "base_compiledir={}".format(compile_dir)

def generate_jobs(max_iters, max_folds, clean):
    
    folds = range(max_folds)
    alphas = [0.01,]
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.25, 2.0, 2.5, 5.0]
    is_fit_embeddings = [False, True]
    win_sizes = [3, 5, 7]
    batch_sizes = [25,]

    n_models = len(alphas) * len(lambdas) * len(is_fit_embeddings) \
        * len(win_sizes) * len(batch_sizes)

    n_jobs = len(folds) * n_models

    jobs = itertools.product(alphas, lambdas, is_fit_embeddings,
        win_sizes, batch_sizes)
    
    def job_generator(jobs):
        for m, (alpha, lam, fit_emb, win_size, b_size) in enumerate(jobs, 1):
            for fold in folds:
                params = {"model_no": m, "fold": fold,
                          "alpha": alpha, "lambda": lam,
                          "fit_embeddings": fit_emb, "win_size": win_size,
                          "batch_size": b_size, "clean": clean}
                yield params    

    n_results = n_jobs * max_iters

    return n_results, n_jobs, n_models, job_generator(jobs)

def collect_results(results):
    cols = ["lambda", "win_size", "fit_embeddings", "clean", "alpha", 
            "batch_size"]
    df = pd.DataFrame(results)
    df = df.set_index(["model_no", "iter"]).groupby(
       level=["model_no", "iter"])
    df2 = df[["dev_acc", "train_acc", "train_nll"]].mean()
    df2.columns = ["dev_acc", "train_acc", "train_nll"]
    df2[cols] = df[cols].first()
    df2 = df2.reset_index()
    df2.sort("dev_acc", inplace=True)
    best_params = df2.tail(1).to_dict(orient="records")[0]
    return df2, best_params

def results2path(model_path, results):
    tmplt = "model_no.{model_no}.fold.{fold}.iter.{iter}.alpha.{alpha}." + \
        "lambda.{lambda}.win_size.{win_size}." + \
        "fit_embeddings.{fit_embeddings}.batch_size.{batch_size}." + \
        "clean.{clean}.pkl"
    fname = tmplt.format(**results)
    return os.path.join(model_path, fname)

def master_process(comm, n_workers, model_type, model_path, results_path, 
                   corpus, clean, max_iters, max_folds):
    status = MPI.Status() # get MPI status object
    # initialize jobs and get expected number of results.
    n_results, n_jobs, n_models, jobs = generate_jobs(
        max_iters, max_folds, clean)

    next_job = next(jobs)
    closed_workers = 0
    all_results = []

    start_time = time.time()
    print "Training {} nn model on {} ({}) data with {} processes.".format(
            model_type, corpus, "clean" if clean else "normal", n_workers)
    while closed_workers < n_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == TAGS.READY:
            # Worker is ready, so send it a task
            if next_job is not None:
                comm.send(next_job, dest=source, tag=TAGS.START)
                print "Training model {} of {}/fold {} of {} (proc-{})".format(
                    next_job["model_no"], n_models, 
                    next_job["fold"], max_folds-1, source)
                try:
                    next_job = next(jobs)
                except StopIteration:
                    next_job = None 
            else:
                comm.send(None, dest=source, tag=TAGS.EXIT)
        elif tag == TAGS.DONE:
            all_results.append(data)
            print "Finished model {} of {}/fold {} of {}".format(
                data["model_no"], n_models, 
                data["fold"], max_folds-1),
		print "/iter {} of {} (proc-{})".format(
			data["iter"], max_iters, source)
        elif tag == TAGS.EXIT:
            print "Process {} exited.".format(source)
            closed_workers += 1

    duration = time.time() - start_time
    avg_time = duration / float(n_jobs)

    df, best_params = collect_results(all_results)
    print "Top 5 results"    
    print df.tail(5)
    print "Avg train time / model:", avg_time
    #print results2path(model_path, best_params)
    with open(results_path, "w") as f:
        df.to_csv(f, index=False)


    setup_theano(rank, corpus, clean)
    import cohere.data
    from cohere.nnet import WordEmbeddings, RecurrentNNModel
   
    train_data = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="tokens", clean=clean,
        convert_brackets=True)
    test_data = cohere.data.get_barzilay_data(
        corpus=corpus, part="test", format="tokens", clean=clean,
        convert_brackets=True)
 
    embed = WordEmbeddings.li_hovy_embeddings(corpus)
    np.random.seed(1999)

    def cb(nnet, n_iter, avg_nll):
        train_acc = nnet.score(train_data)
        test_acc = nnet.score(test_data)
        print n_iter, "avg batch nll", avg_nll
        print "train acc", train_acc, "test acc", test_acc

    nnet = RecurrentNNModel(embed, alpha=best_params["alpha"], 
        lam=best_params["lambda"], window_size=best_params["win_size"], 
        fit_embeddings=best_params["fit_embeddings"], 
        max_iters=best_params["iter"], fit_callback=cb)

    nnet.fit(train_data)
    print "TEST ACC", nnet.score(test_data)

def worker_process(comm, rank, model_path, corpus, clean, 
                   max_iters, max_folds):
    # Worker processes execute code below
    status = MPI.Status() # get MPI status object

    setup_theano(rank, corpus, clean)    
    import cohere.data
    from cohere.nnet import WordEmbeddings, RecurrentNNModel

    dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="tokens", clean=clean,
        convert_brackets=True)
    
    n_data = len(dataset)
    embed = WordEmbeddings.li_hovy_embeddings(corpus)
     
    while True:
        comm.send(None, dest=0, tag=TAGS.READY)
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
       

        if tag == TAGS.START:
            folds = [f for f in KFold(n_data, n_folds=max_folds)]
            I_train, I_dev = folds[job["fold"]]
            train_data = dataset[I_train]
            dev_data = dataset[I_dev]

            def cb(nnet, n_iter, avg_nll):
                train_acc = nnet.score(train_data)
                dev_acc = nnet.score(dev_data)
                results = {"dev_acc": dev_acc, "train_nll": avg_nll, 
                           "train_acc": train_acc, "iter": n_iter}
                results.update(job)
                comm.send(results, dest=0, tag=TAGS.DONE)

            np.random.seed(1999)
            nnet = RecurrentNNModel(embed, alpha=job["alpha"], 
                lam=job["lambda"], window_size=job["win_size"], 
                fit_embeddings=job["fit_embeddings"], 
                fit_callback=cb, max_iters=max_iters)
            nnet.fit(train_data)

        elif tag == TAGS.EXIT:
            break

    comm.send(None, dest=0, tag=TAGS.EXIT)

def main(model_path, results_path, model_type, corpus, clean, 
         max_iters=10, max_folds=10):
    
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    n_workers = size - 1

    if model_path is None: 
        model_path = os.path.join(
            os.getenv("COHERENCE_DATA", "data"), 
            "models", "{}.{}.{}".format(
                model_type, corpus, "clean" if clean else "norm"))
    if rank == 0 and not os.path.exists(model_path):
        os.makedirs(model_path)

    if results_path is None:
        results_path = "{}.{}.{}.results.csv".format(
            model_type, corpus, "clean" if clean else "norm")
    
    results_dir = os.path.dirname(results_path)
    if rank == 0 and results_dir != "" and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if rank == 0:
        master_process(comm, n_workers, model_type, 
            model_path, results_path, corpus, clean, max_iters, max_folds)
    else:
        worker_process(
            comm, rank, model_path, corpus, clean, max_iters, max_folds)

if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', '-c',
                        help='corpus to train on', required=True, 
                        choices=["apws", "ntsb"])
    parser.add_argument("--model-type", help="model type", required=False,
                        choices=["recurrent", "recursive", "cbow"],
                        default="recurrent")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="use the clean version of the corpus")
    parser.add_argument("--max-iters", default=10, type=int,
                        help="number of training iterations")
    parser.add_argument("--max-folds", default=10, type=int,
                        help="number of training folds")
    parser.add_argument("--model-path", help="path to write models",
                        required=False, default=None)
    parser.add_argument("--results-path", help="path to write results csv",
                        required=False, default=None)



    args = parser.parse_args()
    corpus = args.corpus
    clean = args.clean
    max_iters = args.max_iters
    max_folds = args.max_folds
    model_path =args.model_path   
    results_path = args.results_path
    model_type = args.model_type


    main(model_path, results_path, model_type, corpus, clean, 
        max_iters=max_iters, max_folds=max_folds)
