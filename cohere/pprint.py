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
