import os
import cohere
import cohere.data
import cohere.embed
from cohere.models import RNNModel

def make_vocab(docs):
    vocab = set()
    for doc in docs:
        for sent in doc:
            for w in sent:
                vocab.add(w)
    return vocab

def pprint_emb(vec):
    return "[{:0.3f} {:0.3f} ... {:0.3f} ... {:0.3f} {:0.3f}]".format(
         vec[0], vec[1], vec[vec.shape[0] // 2], vec[-2], vec[-1])

def main(output_model_path):
    docs_train = cohere.data.get_barzilay_ntsb_clean_docs_only(
        "train", tokens_only=True)
    vocab = make_vocab(docs_train)

    docs_perms_dev = cohere.data.get_barzilay_ntsb_clean_docs_perms(
        "dev", tokens_only=True)

    glove = cohere.embed.GloVeEmbeddings()
    embed = cohere.embed.BorrowingWordEmbeddings(vocab, glove)
    print "embbeding  '__PAD__':", pprint_emb(embed.get_embedding("__PAD__"))
    print "index      '__PAD__':", embed.get_index("__PAD__")
    print "embbeding  '__START__':", pprint_emb(
        embed.get_embedding("__START__"))
    print "index      '__START__':", embed.get_index("__START__")
    print "embbeding  '__STOP__':", pprint_emb(
        embed.get_embedding("__STOP__"))
    print "index      '__STOP__':", embed.get_index("__STOP__")
    print "embbeding  '__UNKNOWN__':", pprint_emb(
        embed.get_embedding("__UNKNOWN__"))
    print "index      '__UNKNOWN__':", embed.get_index("__UNKNOWN__")

    max_sent_len = 150
    transformer = cohere.embed.IndexDocTransformer(
        embed, start_pads=1, stop_pads=1, max_sent_len=max_sent_len)
    transformer.fit(docs_train)
    X_train, y_train = transformer.transform(docs_train)
    X_gold, X_perm = transformer.transform_test(docs_perms_dev)


    nnet = RNNModel(embed, max_sent_len, learning_rate=.1, 
                    batch_size=20, fit_init=True)
    nnet.fit(X_train, y_train, X_gold=X_gold, X_perm=X_perm)



if __name__ ==  u"__main__":
    output_dir = os.path.join(
        os.getenv("COHERENCE_DATA", "data"), "models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "rnn.pkl.gz")

    main(output_path)
