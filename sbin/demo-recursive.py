import cohere.data
from cohere.pprint import pprint as pp
from cohere.nnet import TreeTransformer, WordEmbeddings, RecursiveNNModel
import os

def main(corpus="apws"):

    embed = WordEmbeddings.li_hovy_embeddings(corpus)
    train_dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="trees", clean=False, 
        convert_brackets=True)
    test_dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="test", format="trees", clean=False, 
        convert_brackets=True)

    print "Some example data: "
    pp(train_dataset[0:5])
        

    print "We need to know the maximum sentence length and the maximum" + \
        " number of tree combining operations in order to put\n" +\
        "all input windows into a single matrix for batch learning."
    max_sent, max_ops = TreeTransformer.get_max_sent_and_ops(
        train_dataset.gold + test_dataset.gold)
    print "The maximum sentence length is {}".format(max_sent)
    print "The maximum operation sequence length is {}".format(max_ops)

    transformer = TreeTransformer(embed, max_sent_len=max_sent, 
        max_ops_len=max_ops,
        window_size=7)

    X_iw_train, O_iw_train, y_train = transformer.training_window_transform(
        train_dataset.gold)

    X_train_gold, O_train_gold, X_train_perm, O_train_perm = \
        transformer.testing_window_transform(train_dataset)

    X_test_gold, O_test_gold, X_test_perm, O_test_perm = \
        transformer.testing_window_transform(test_dataset)

    def callback(nnet, n_iter):
        print n_iter, 
        print "Train Acc.", nnet.score(
            X_train_gold, O_train_gold, X_train_perm, O_train_perm),
        print "Test Acc.", nnet.score(
            X_test_gold, O_test_gold, X_test_perm, O_test_perm)
    nnet = RecursiveNNModel(embed, max_sent_len=max_sent, 
        max_ops_len=max_ops, lam=1.25, 
        alpha=.01, batch_size=25, window_size=3, max_iters=10, 
        fit_embeddings=False, fit_callback=callback)
    nnet.fit(X_iw_train, O_iw_train, y_train)


    print "Train Acc.", nnet.score(
        X_train_gold, O_train_gold, X_train_perm, O_train_perm)
    print "Test Acc.", nnet.score(
        X_test_gold, O_test_gold, X_test_perm, O_test_perm)
 

if __name__ == u"__main__":
    main()
