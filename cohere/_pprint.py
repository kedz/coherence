import cohere.data
import corenlp.document

def _print_str_dataset(dataset):
    return u"\n".join([_print_str_instance(inst) 
                       for idx, inst in enumerate(dataset)])
        

def _print_str_instance(instance):
    string = u"[ " + unicode(instance.id)
    string += u" w/ {} permutations\n".format(instance.num_perms)
    if instance.format == "tokens":
        string += u"\n".join([" " + unicode(idx) + ") " + u" ".join(sent) 
                              for idx, sent in enumerate(instance.gold, 1)])
    elif instance.format == "trees":
        string += u"\n".join([" {}) {}".format(idx, u" ".join(sent.leaves()))
                              for idx, sent in enumerate(instance.gold, 1)])
    string += u"]\n"
    return string

def pprint(obj):
    if isinstance(obj, cohere.data.CoherenceData):
        print _print_str_dataset(obj).encode(u"utf-8")
    elif isinstance(obj, cohere.data.CoherenceInstance):
        print _print_str_instance(obj).encode(u"utf-8")

def pprint_X(X, transformer, y=None, width=80):
    
    word_dim = transformer.embeddings.W.shape[1]
    win_size = transformer.window_size
    max_words = transformer.max_sent_len

    win_width = int(width - 5 * (win_size -1)) / int(win_size)

    if len(X.shape) == 1:
        X = X.reshape((1, X.shape[0]))
    if X.shape[1] != max_words * win_size:
        raise Exception(
            "X columns {} not consistent with transfomer {}".format(
                X.shape[1], max_words * win_size))

    for n_row, x in enumerate(X):
        cols = []
        for k in xrange(win_size):
            words = []
            for i in x[k * max_words : (k + 1) * max_words]:
                if i != -1:
                    words.append(transformer.embeddings.index2token[i])
            line = u' '.join(words)[:win_width]
            line += u' ' * (win_width - len(line))
            cols.append(line)
        row = u" \u22EF  ".join(cols)   
        if len(row) < width:
            row += u" " * (width - len(row) - 1)
        if y is not None:
            row = row[:-5] + u" y={}".format(y[n_row])        
        print row.encode(u"utf-8")

