import cohere.data
from cohere.pprint import pprint as pp
from cohere.nnet import TokensTransformer, WordEmbeddings, RecurrentNNModel
import os

def main():

    train_dataset = cohere.data.get_barzilay_data(
        corpus="apws", part="train", format="tokens", clean=False, 
        convert_brackets=False)
    test_dataset = cohere.data.get_barzilay_data(
        corpus="apws", part="test", format="tokens", clean=False, 
        convert_brackets=False)

    print "Some example data: "
    pp(train_dataset[0:5])
        
    embed_path = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), 
        "apws_embeddings.txt.gz")
       # .format("clean_" if clean else "", corpus))
#    print "Reading word embeddings from {} ...".format(embed_path)
    embed = WordEmbeddings.from_file(embed_path)

    print "We need to know the maximum sentence length in order to put all" +\
        " input windows into a single matrix for batch learning."            
    max_sent_len = TokensTransformer.get_max_sent_len(
        train_dataset.gold + test_dataset.gold)
    print "The maximum sentence length is {}".format(max_sent_len)

    transformer = TokensTransformer(embed, max_sent_len=max_sent_len, 
        window_size=7)

    X_iw_train, y_train = transformer.training_window_transform(
        train_dataset.gold)

    nnet = RecurrentNNModel(embed, max_sent_len=max_sent_len, lam=1., 
        alpha=.01, batch_size=25, window_size=7, max_iters=10, 
        fit_embeddings=True)
    nnet.fit(X_iw_train, y_train)

    X_test_gold, X_test_perm = transformer.testing_window_transform(
        test_dataset)
    X_train_gold, X_train_perm = transformer.testing_window_transform(
        train_dataset)
    print "Train Acc.", nnet.score(X_train_gold, X_train_perm)
    print "Test Acc.", nnet.score(X_test_gold, X_test_perm)

if __name__ == u"__main__":
    main()
