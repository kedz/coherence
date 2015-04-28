import cohere.data
from cohere import pprint as pp
from cohere import pprint_X as pp_X
from cohere.nnet import TokensTransformer, WordEmbeddings, RecurrentNNModel
import os

def main(corpus="apws", window_size=7):

    embed = WordEmbeddings.li_hovy_embeddings(corpus)
    train_dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="train", format="tokens", clean=False, 
        convert_brackets=True)
    test_dataset = cohere.data.get_barzilay_data(
        corpus=corpus, part="test", format="tokens", clean=False, 
        convert_brackets=True)

    print "Some example data: "
    pp(train_dataset[0:5])
        

    print "We need to know the maximum sentence length in order to put\n" +\
        "all input windows into a single matrix for batch learning."
    max_sent_len = TokensTransformer.get_max_sent_len(
        train_dataset.gold + test_dataset.gold)
    print "The maximum sentence length is {}".format(max_sent_len)

    transformer = TokensTransformer(embed, max_sent_len=max_sent_len, 
        window_size=window_size)

    X_iw_train, y_train = transformer.training_window_transform(
        train_dataset.gold)
    X_test_gold, X_test_perm = transformer.testing_window_transform(
        test_dataset)
    X_train_gold, X_train_perm = transformer.testing_window_transform(
        train_dataset)
    pp_X(X_iw_train[:25], transformer)

    def callback(nnet, n_iter):
        print n_iter, 
        print "Train Acc.", nnet.score(X_train_gold, X_train_perm),
        print "Test Acc.", nnet.score(X_test_gold, X_test_perm)
    nnet = RecurrentNNModel(embed, max_sent_len=max_sent_len, lam=1., 
        alpha=.01, batch_size=25, window_size=window_size, max_iters=10, 
        fit_embeddings=False, fit_callback=callback)
    nnet.fit(X_iw_train, y_train)


    print "Train Acc.", nnet.score(X_train_gold, X_train_perm)
    print "Test Acc.", nnet.score(X_test_gold, X_test_perm)

if __name__ == u"__main__":
    main()
