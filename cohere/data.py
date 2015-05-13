import tarfile
import os
import time
from cStringIO import StringIO
import corenlp.server
import corenlp.client
import gzip
import cPickle as pickle
import random
import re
import urllib2
import collections
from nltk.tree import Tree


class CoherenceData(object):
    def __init__(self, instances, format):
        self.instances = instances
        self.gold = [inst.gold for inst in instances]
        self.perms = [inst.perms for inst in instances]
        self.ids = [inst.id for inst in instances]
        self.format = format

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, obj):
        if isinstance(obj, collections.Iterable):
            selection = [self.instances[i] for i in obj]
        elif isinstance(obj, slice):
            selection = self.instances[obj]
        else:
            selection = [self.instances[obj]]
        return CoherenceData(selection, self.format)

    def __iter__(self):
        return iter(self.instances)        

class CoherenceInstance(object):
    def __init__(self, id, format, gold, perms):
        self.id = id
        self.format = format
        self.gold = gold
        self.perms = perms
        self.num_perms = len(perms)

def remove_apws_meta(f):
    clean_text = []
    #lines = text.split("\n")
    for line in f:
        line = line.strip()
        if line == "Greece-Earthquake|Strong Quake Hits Northern Greece.":
            ##print "DELETE",
            continue
        elif line.endswith("|By"):
            ##print "DELETE",
            continue
        elif line.startswith("BC-"):
            ##print "DELETE",
            continue
        elif line[0] == "(" and line[-1] == ")":
            ##print "DELETE",
            continue
        elif line[0] == "(" and line.endswith(") "):
            #print "DELETE",
            continue
        elif line == "pd ":
            #print "DELETE",
            continue
        elif line == "CHANGES dateline.":
            #print "DELETE",
            continue
        elif line == "No|pickup..":
            #print "DELETE",
            continue
        elif line == "nk":
            #print "DELETE",
            continue
        elif line.endswith("|By "):
            #print "DELETE",
            continue
        elif line.endswith("|Associated Press Writer. "):
            #print "DELETE",
            continue
        elif line.endswith(",0191. "):
            #print "DELETE",
            continue
        elif line.startswith("AM-BRF"):
            #print "DELETE",
            continue
        elif line.startswith("pickup.. "):
            #print "DELETE",
            continue
        elif line.startswith("pickup.."):
            #print "DELETE",
            continue
        elif line.startswith("PM-Mex"):
            #print "DELETE",
            continue
        elif line == "MORE":
            #print "DELETE",
            continue
        elif line == "MORE ":
            #print "DELETE",
            continue
        elif line == "No|pickup.. ":
            #print "DELETE",
            continue
        elif line == "ADDS byline and photo tag.|AP ":
            #print "DELETE",
            continue
        elif line == "No pickup.. ":
            #print "DELETE",
            continue
        elif line == "No pickup..":
            #print "DELETE",
            continue
        elif line == "Clarifies|earthquake in northwest. ":
            #print "DELETE",
            continue
        elif line.endswith(" quotes.|No "):
            #print "DELETE",
            continue
        elif line.startswith("intjw 87 233|"):
            #print "DELETE",
            continue
        elif line == "pq":
            #print "DELETE",
            continue
        elif line == "pq ":
            #print "DELETE",
            continue
        elif line == "CHANGES dateline. ":
            #print "DELETE",
            continue
        elif line == "nk ":
            #print "DELETE",
            continue
        elif line.endswith("|Associated Press Writer."):
            #print "DELETE",
            continue
        elif line == "CORRECTS depth|of quake.":
            #print "DELETE",
            continue
        elif line == "CORRECTS depth|of quake. ":
            #print "DELETE",
            continue
        elif line.endswith("|No"):
            #print "DELETE",
            continue
        elif line.endswith("|No "):
            #print "DELETE",
            continue

        items = re.split(
            r"^[A-Z]+.+\(A[Pp]\) |^[A-Z]+.+\(Kyodo\) ", line)
        if len(items) == 2:
            #print items[0], " {[]} ", re.sub(r"^-- ", r"(DEL ^--)", items[1])
            clean_text.append(items[1])
        else:
            clean_text.append(items[0])

    return "\n".join(clean_text)

def get_barzilay_data(corpus=u"apws", part=u"train", 
                      clean=False, format=u"document", include_perms=True,
                      convert_brackets=False):

    if convert_brackets is True and format not in [u"document", u"text"]:
        __brackets = {
            u'-LRB-': u'(',
            u'-lrb-': u'(',
            u'-RRB-': u')',
            u'-rrb-': u')',
            u'-LCB-': u'{',
            u'-lcb-': u'{',
            u'-RCB-': u'}',
            u'-rcb-': u'}',
            u'-LSB-': u'[',
            u'-lsb-': u'[',
            u'-RSB-': u']',
            u'-rsb-': u']'}

    assert format in [u"text", u"xml", u"document", u"tokens", u"trees"]

    if format == u"text":
        path_format = u""
    elif format in [u"xml", u"document"]:
        path_format = u"_xml"
    elif format == u"tokens":
        path_format = u"_tokens"
    elif format == u"trees":
        path_format = u"_trees"
    fname = u"{}_{}_{}{}.tar.gz".format(
        u"b&l" if clean is False else u"clean",
        corpus,
        part, 
        path_format)

    key_name = u"{}{}_{}{}".format(
        u"b&l_" if clean is False else u"clean_",
        corpus,
        part,
        path_format)
    
    path = os.path.join(os.getenv(u"COHERENCE_DATA", u"."), fname)

    def read_document(f):
        return corenlp.read_xml(f, convert_brackets=convert_brackets)
    def read_tokens(f):
        doc = []
        for line in f:
            tokens = line.strip().decode(u"utf-8").split(u" ")
            tokens = [t.lower() for t in tokens]
            if convert_brackets is True:
                tokens = [__brackets.get(t, t) for t in tokens]
            doc.append(tokens)
        return doc

    def read_parse(f):
        doc = []
        for line in f:
            t = Tree.fromstring(line.strip().decode(u"utf-8"))
            if convert_brackets is True:
                for pos in t.treepositions("leaves"):
                    t[pos] = __brackets.get(t[pos], t[pos])
            doc.append(t)
        return doc

    if format == u"text" or format == u"xml":
        reader = None
    elif format == u"document":
        reader = read_document
    elif format == u"tokens":
        reader = read_tokens
    elif format == u"trees":
        reader = read_parse
    if format in [u"xml", u"document"]:
        file_ext = ".xml"
    else:
        file_ext = ".txt"
    print path
    data = read_tar(
        path, text_filter=reader, file_ext=file_ext, no_perm_num=True)
    data = data[key_name]

    instances = []
    for name, instance in sorted(data.items(), key=lambda x: x[0]):
        ci = CoherenceInstance(
            name, format, instance[u"gold"], instance[u"perms"])
        instances.append(ci)
    return CoherenceData(instances, format)

def extract_barzilay_tar(path_or_url, text_filter=None):

    if text_filter is None:
        text_filter = lambda f: ''.join(f.readlines())

    is_Z = path_or_url.endswith("Z")
    if path_or_url.startswith("http://"):
        fileobj = StringIO(urllib2.urlopen(path_or_url).read())
    else:
        fileobj = open(path_or_url)

    if is_Z:
        with open("tmp.gz", mode="w") as f:
            f.write(fileobj.read())
            f.flush()
            os.system("gunzip tmp.gz")
            fileobj1 = open("tmp", "r")
            fileobj = StringIO(fileobj1.read())
            fileobj1.close()
            os.system("rm tmp")

    data = {}
    with tarfile.open(fileobj=fileobj, mode="r") as tar:
        for tarinfo in tar:
            
            if "perm" not in os.path.basename(tarinfo.name):
                continue
            
            instance_name = re.search(
                r"(ntsb\d{4}|apwsE\d{6}\.\d{4})", 
                tarinfo.name).groups(1)[0]
            
            perm_no = re.search(
                r"perm-(\d+)", 
                os.path.basename(tarinfo.name)).groups(1)[0]
            perm_no = int(perm_no)

            partition = os.path.split(os.path.dirname(tarinfo.name))[1]
            
            if partition not in data:
                data[partition] = {}
            data_dict = data[partition]

            f = tar.extractfile(tarinfo)
            text = text_filter(f)            
            
            if instance_name not in data_dict:
                data_dict[instance_name] = {
                    u"gold": None, u"perms": list()}
            if perm_no == 1:
                data_dict[instance_name][u"gold"] = text
            else:
                data_dict[instance_name][u"perms"].append( 
                    (perm_no, text))
    import math
    
    total = 0
    for inst, datum in data_dict.items():
        n_sents = len(datum[u"gold"].split("\n"))
        max_perms = math.factorial(n_sents)
        total += len(datum[u"perms"])
        if max_perms >= 20:
            if len(datum[u"perms"]) > 20:
                print "Bad instance : should have 20 perms", inst
                print "Has {} perms".format(len(datum[u"perms"]))
                print datum[u"gold"]
                raise Exception()
        elif max_perms < 20:
            if len(datum[u"perms"]) != max_perms:
                print "Bad instance: should have {} perms".format(max_perms),
                print "Has {} perms".format(len(datum[u"perms"]))
                print inst
                print datum[u"gold"]
                raise Exception()
    print "{} instances with {} permutations".format(len(data_dict), total)
    return data


def read_tar(tgz_path, text_filter=None, file_ext=".txt", no_perm_num=True):

    if text_filter is None:
        text_filter = lambda f: ''.join(f.readlines())

    parts = {}
    with tarfile.open(tgz_path, u"r:gz") as tar:
        for tarinfo in tar:
            if not tarinfo.isfile():
                continue
            path = os.path.dirname(tarinfo.name)
            if os.path.basename(tarinfo.name) == "gold" + file_ext:
                part_name, instance_name = os.path.split(path)
                if part_name not in parts:
                    parts[part_name] = {}
                if instance_name not in parts[part_name]:
                    parts[part_name][instance_name] = {u"gold": None, 
                                                       u"perms": list()}
                f = tar.extractfile(tarinfo)
                text = text_filter(f) 
                parts[part_name][instance_name][u"gold"] = text
            else:
                part_name, instance_name = os.path.split(os.path.dirname(path))
                perm_no = int(os.path.basename(
                    tarinfo.name).replace(file_ext, ""))
                f = tar.extractfile(tarinfo)
                text = text_filter(f) 
                data = parts[part_name]
                if no_perm_num:
                    data[instance_name][u"perms"].append(text)
                else:
                    data[instance_name][u"perms"].append((perm_no, text))
    return parts

def write_tar(path, data, file_ext=".txt"):
    with tarfile.open(path, u"w:gz") as tar:
        for partition, instances in data.items():
            for instance_name, instance in instances.items():
                path = os.path.join(partition, instance_name)
                info = tarfile.TarInfo(name=path)
                info.type = tarfile.DIRTYPE
                info.mode = 0755
                info.mtime = time.time()
                tar.addfile(tarinfo=info)

                gold_path = os.path.join(path, "gold" +file_ext)
                gold_text = StringIO(instance[u"gold"])
                info = tarfile.TarInfo(name=gold_path)
                info.size=len(instance[u"gold"])
                info.mtime = time.time()
                tar.addfile(tarinfo=info, fileobj=gold_text) 
                
                perms = instance[u"perms"]
                perms.sort(key=lambda x: x[0])
                for perm_no, perm in perms:
                    perm_path = os.path.join(
                        path, "perms", "{}{}".format(perm_no, file_ext))
                    perm_text = StringIO(perm)
                    info = tarfile.TarInfo(name=perm_path)
                    info.size=len(perm)
                    info.mtime = time.time()
                    tar.addfile(tarinfo=info, fileobj=perm_text) 

def download_ntsb(train_url, tgz_path, partmap):
    """Download original dataset from entity grid paper without cleaning."""

    def text_extract(f):
        text = []
        for line in f:
            text.append(line.split(" ", 1)[1])
            #    re.sub(r"-- ", r"", ))
        f.close()
        text = ''.join(text).strip()
        assert len(text) > 0
        return text

    data = extract_barzilay_tar(train_url, text_extract)
    data_renamed = {partmap[partition]: instances 
                    for partition, instances in data.items()}
    write_tar(tgz_path, data_renamed)
 

def download_apws(train_url, tgz_path, partmap):
    """Download original dataset from entity grid paper without cleaning."""

    def text_extract(f):
        text = []
        for line in f:
            text.append(
                re.sub(r"-- ", r"", line.split(" ", 1)[1]))
        f.close()
        text = ''.join(text).strip()
        assert len(text) > 0
        return text

    data = extract_barzilay_tar(train_url, text_extract)
    data_renamed = {partmap[partition]: instances 
                    for partition, instances in data.items()}
    write_tar(tgz_path, data_renamed)
    
def clean_apws_data(apws_tgz, apws_clean_tgz):
    data = read_tar(apws_tgz, remove_apws_meta, no_perm_num=False)
    data_renamed = {"clean_" + key: val for key, val in data.items()}
    write_tar(apws_clean_tgz, data_renamed)


def make_xml(text_tgz, xml_tgz, mem=u'7G', n_procs=4):
    data = read_tar(text_tgz, no_perm_num=False)
    keys = []
    texts = []
    for partition, instances in data.items():
        for instance_name, instance in instances.items():
            texts.append(instance[u"gold"])
            keys.append((partition + "_xml", instance_name))
            for perm_no, perm in instance[u"perms"]:
                texts.append(perm)
                keys.append((partition + "_xml", instance_name, perm_no))
    corenlp_props = {"ssplit.newlineIsSentenceBreak": "always"}
    anns = ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "dcoref"]

    xml_data = {}

    with corenlp.Server(mem=mem, annotators=anns, threads=n_procs,
                        corenlp_props=corenlp_props) as pipeline:
        for key, xml in pipeline.annotate_mp_unordered_iter(
                texts, keys=keys, n_procs=n_procs, return_xml=True,
                show_progress=True):
            
            xml = xml.encode(u"utf-8")
             
            if len(key) == 2:
                part, name = key
                perm_no = -1
            else:
                part, name, perm_no = key
            
            if part not in xml_data:
                xml_data[part] = {}
            if name not in xml_data[part]:
                xml_data[part][name] = {u"gold": None, u"perms": list()}

            if perm_no == -1:
                xml_data[part][name][u"gold"] = xml
                assert xml_data[part][name][u"gold"] is not None
            else:
                xml_data[part][name][u"perms"].append((perm_no, xml))

    for part in xml_data.values():
        for instance in part.values():
            instance[u"perms"].sort(key=lambda x: x[0])

    write_tar(xml_tgz, xml_data, file_ext=".xml")


def make_tokens(corpus, clean, part, tokens_tgz):
    data = get_barzilay_data(
        corpus=corpus, part=part, clean=clean, format=u"document", 
        include_perms=True, convert_brackets=False)

    part_name = u"{}_{}_{}_tokens".format(
        u"b&l" if clean is False else u"clean",
        corpus,
        part)

    def cnvrt(doc):
        doc = [[unicode(t).lower() for t in s]
               for s in doc]
        return u'\n'.join(u' '.join(sent) for sent in doc).encode(u"utf-8")

    tokens_data = {part_name: {}}
    for inst in data:
        tokens_data[part_name][inst.id] = {
            u"gold": cnvrt(inst.gold), 
            u"perms": [(idx, cnvrt(perm)) 
                       for idx, perm in enumerate(inst.perms, 1)]}
    write_tar(tokens_tgz, tokens_data, file_ext=".txt")

def make_trees(corpus, clean, part, trees_tgz):
    data = get_barzilay_data(
        corpus=corpus, part=part, clean=clean, format=u"document", 
        include_perms=True, convert_brackets=False)

    part_name = u"{}_{}_{}_trees".format(
        u"b&l" if clean is False else u"clean",
        corpus,
        part)

    def cnvrt(doc):
        doc = [s.parse._pformat_flat("", "()", False) for s in doc]
        return u'\n'.join(doc).encode(u"utf-8")

    trees_data = {part_name: {}}
    for inst in data:
        trees_data[part_name][inst.id] = {
            u"gold": cnvrt(inst.gold), 
            u"perms": [(idx, cnvrt(perm)) 
                       for idx, perm in enumerate(inst.perms, 1)]}
    write_tar(trees_tgz, trees_data, file_ext=".txt")



def make_word_embeddings(path, corpus, clean, max_iters):
    docs_train = get_barzilay_data(corpus=corpus, part=u"train", 
                                   clean=clean, format=u"tokens", 
                                   include_perms=False, convert_brackets=False)
    
    docs_test = get_barzilay_data(corpus=corpus, part=u"test", 
                                  clean=clean, format=u"tokens", 
                                  include_perms=False, convert_brackets=False)
    S = []
    for doc in docs_train.gold:
        for sent in doc:
            S.append(sent)
    for doc in docs_test.gold:
        for sent in doc:
            S.append(sent)
    model = gensim.models.Word2Vec(S, size=50, min_count=1, workers=16,
                                   iter=max_iters)
    with gzip.open(path, "w") as f:
        for word in model.vocab.keys():
            word_str = word.encode(u'utf-8')
            f.write(' '.join([word_str] + [str(x) for x in model[word]]))
            f.write('\n')

def main(n_procs=2, mem="8G", embed_iters=1000):

    ntsb_url_train = \
     "http://people.csail.mit.edu/regina/coherence/data2-train-perm.tar.Z"
    ntsb_url_test = \
     "http://people.csail.mit.edu/regina/coherence/CLsubmission/ntsb-test.tgz"
    apws_url_train = \
     "http://people.csail.mit.edu/regina/coherence/data1-train-perm.tar"
    apws_url_test = \
     "http://people.csail.mit.edu/regina/coherence/CLsubmission/data1-test.tgz"
    
    data_dir = os.getenv("COHERENCE_DATA", os.path.join(os.getcwd(), "data"))
    print "Installing data sets for this package in\n\t{} ...".format(
        data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    apws_train_tgz = os.path.join(
        data_dir, "b&l_apws_train.tar.gz")
    has_apws_train = os.path.exists(apws_train_tgz)

    apws_test_tgz = os.path.join(
        data_dir, "b&l_apws_test.tar.gz")
    has_apws_test = os.path.exists(apws_test_tgz)

    clean_apws_train_tgz = os.path.join(
        data_dir, "clean_apws_train.tar.gz")
    has_clean_apws_train = os.path.exists(clean_apws_train_tgz)

    clean_apws_test_tgz = os.path.join(
        data_dir, "clean_apws_test.tar.gz")
    has_clean_apws_test = os.path.exists(clean_apws_test_tgz)

    apws_train_xml_tgz = os.path.join(
        data_dir, "b&l_apws_train_xml.tar.gz")
    has_apws_train_xml = os.path.exists(apws_train_xml_tgz)

    apws_test_xml_tgz = os.path.join(
        data_dir, "b&l_apws_test_xml.tar.gz")
    has_apws_test_xml = os.path.exists(apws_test_xml_tgz)

    apws_train_tokens_tgz = os.path.join(
        data_dir, "b&l_apws_train_tokens.tar.gz")
    has_apws_train_tokens = os.path.exists(apws_train_tokens_tgz)

    apws_test_tokens_tgz = os.path.join(
        data_dir, "b&l_apws_test_tokens.tar.gz")
    has_apws_test_tokens = os.path.exists(apws_test_tokens_tgz)

    apws_train_trees_tgz = os.path.join(
        data_dir, "b&l_apws_train_trees.tar.gz")
    has_apws_train_trees = os.path.exists(apws_train_trees_tgz)

    apws_test_trees_tgz = os.path.join(
        data_dir, "b&l_apws_test_trees.tar.gz")
    has_apws_test_trees = os.path.exists(apws_test_trees_tgz)

    clean_apws_train_xml_tgz = os.path.join(
        data_dir, "clean_apws_train_xml.tar.gz")
    has_clean_apws_train_xml = os.path.exists(clean_apws_train_xml_tgz)

    clean_apws_test_xml_tgz = os.path.join(
        data_dir, "clean_apws_test_xml.tar.gz")
    has_clean_apws_test_xml = os.path.exists(clean_apws_test_xml_tgz)

    clean_apws_train_tokens_tgz = os.path.join(
        data_dir, "clean_apws_train_tokens.tar.gz")
    has_clean_apws_train_tokens = os.path.exists(clean_apws_train_tokens_tgz)

    clean_apws_test_tokens_tgz = os.path.join(
        data_dir, "clean_apws_test_tokens.tar.gz")
    has_clean_apws_test_tokens = os.path.exists(clean_apws_test_tokens_tgz)

    clean_apws_train_trees_tgz = os.path.join(
        data_dir, "clean_apws_train_trees.tar.gz")
    has_clean_apws_train_trees = os.path.exists(clean_apws_train_trees_tgz)

    clean_apws_test_trees_tgz = os.path.join(
        data_dir, "clean_apws_test_trees.tar.gz")
    has_clean_apws_test_trees = os.path.exists(clean_apws_test_trees_tgz)

    ntsb_train_tgz = os.path.join(
        data_dir, "b&l_ntsb_train.tar.gz")
    has_ntsb_train = os.path.exists(ntsb_train_tgz)

    ntsb_test_tgz = os.path.join(
        data_dir, "b&l_ntsb_test.tar.gz")
    has_ntsb_test = os.path.exists(ntsb_test_tgz)

    clean_ntsb_train_tgz = os.path.join(
        data_dir, "clean_ntsb_train.tar.gz")
    has_clean_ntsb_train = os.path.exists(clean_ntsb_train_tgz)

    clean_ntsb_test_tgz = os.path.join(
        data_dir, "clean_ntsb_test.tar.gz")
    has_clean_ntsb_test = os.path.exists(clean_ntsb_test_tgz)

    ntsb_train_xml_tgz = os.path.join(
        data_dir, "b&l_ntsb_train_xml.tar.gz")
    has_ntsb_train_xml = os.path.exists(ntsb_train_xml_tgz)

    ntsb_test_xml_tgz = os.path.join(
        data_dir, "b&l_ntsb_test_xml.tar.gz")
    has_ntsb_test_xml = os.path.exists(ntsb_test_xml_tgz)

    ntsb_train_tokens_tgz = os.path.join(
        data_dir, "b&l_ntsb_train_tokens.tar.gz")
    has_ntsb_train_tokens = os.path.exists(ntsb_train_tokens_tgz)

    ntsb_test_tokens_tgz = os.path.join(
        data_dir, "b&l_ntsb_test_tokens.tar.gz")
    has_ntsb_test_tokens = os.path.exists(ntsb_test_tokens_tgz)

    ntsb_train_trees_tgz = os.path.join(
        data_dir, "b&l_ntsb_train_trees.tar.gz")
    has_ntsb_train_trees = os.path.exists(ntsb_train_trees_tgz)

    ntsb_test_trees_tgz = os.path.join(
        data_dir, "b&l_ntsb_test_trees.tar.gz")
    has_ntsb_test_trees = os.path.exists(ntsb_test_trees_tgz)

    clean_ntsb_train_xml_tgz = os.path.join(
        data_dir, "clean_ntsb_train_xml.tar.gz")
    has_clean_ntsb_train_xml = os.path.exists(clean_ntsb_train_xml_tgz)

    clean_ntsb_test_xml_tgz = os.path.join(
        data_dir, "clean_ntsb_test_xml.tar.gz")
    has_clean_ntsb_test_xml = os.path.exists(clean_ntsb_test_xml_tgz)

    ### Embeddings paths. ###
    apws_embeddings = os.path.join(
        data_dir, "apws_embeddings.txt.gz")
    has_apws_embeddings = os.path.exists(apws_embeddings)

    clean_apws_embeddings = os.path.join(
        data_dir, "clean_apws_embeddings.txt.gz")
    has_clean_apws_embeddings = os.path.exists(clean_apws_embeddings)

    ntsb_embeddings = os.path.join(
        data_dir, "ntsb_embeddings.txt.gz")
    has_ntsb_embeddings = os.path.exists(ntsb_embeddings)

    clean_ntsb_embeddings = os.path.join(
        data_dir, "clean_ntsb_embeddings.txt.gz")
    has_clean_ntsb_embeddings = os.path.exists(clean_ntsb_embeddings)

    ### Print status. ###
    print "[{}] Barzilay&Lapata NTSB train txt".format(
        "X" if has_ntsb_train else " ")
    print "[{}] Barzilay&Lapata NTSB test txt".format(
        "X" if has_ntsb_test else " ")

    print "[{}] Barzilay&Lapata NTSB train xml".format(
        "X" if has_ntsb_train_xml else " ")
    print "[{}] Barzilay&Lapata NTSB test xml".format(
        "X" if has_ntsb_test_xml else " ")

    print "[{}] Barzilay&Lapata NTSB train tokens".format(
        "X" if has_ntsb_train_tokens else " ")
    print "[{}] Barzilay&Lapata NTSB test tokens".format(
        "X" if has_ntsb_test_tokens else " ")
 
    print "[{}] Barzilay&Lapata NTSB train trees".format(
        "X" if has_ntsb_train_trees else " ")
    print "[{}] Barzilay&Lapata NTSB test trees".format(
        "X" if has_ntsb_test_trees else " ")
    
    print "[{}] Clean NTSB train txt".format(
        "X" if has_clean_ntsb_train else " ")
    print "[{}] Clean NTSB test txt".format(
        "X" if has_clean_ntsb_test else " ")

    print "[{}] Clean NTSB train xml".format(
        "X" if has_clean_ntsb_train_xml else " ")
    print "[{}] Clean NTSB test xml".format(
        "X" if has_clean_ntsb_test_xml else " ")

    print "[{}] NTSB Embeddings".format(
        "X" if has_ntsb_embeddings else " ")
    
    print "[{}] Clean NTSB Embeddings".format(
        "X" if has_clean_ntsb_embeddings else " ")

    print "[{}] Barzilay&Lapata APWS train txt".format(
        "X" if has_apws_train else " ")
    print "[{}] Barzilay&Lapata APWS test txt".format(
        "X" if has_apws_test else " ")

    print "[{}] Barzilay&Lapata APWS train xml".format(
        "X" if has_apws_train_xml else " ")
    print "[{}] Barzilay&Lapata APWS test xml".format(
        "X" if has_apws_test_xml else " ")

    print "[{}] Barzilay&Lapata APWS train tokens".format(
        "X" if has_apws_train_tokens else " ")
    print "[{}] Barzilay&Lapata APWS test tokens".format(
        "X" if has_apws_test_tokens else " ")
 
    print "[{}] Barzilay&Lapata APWS train trees".format(
        "X" if has_apws_train_trees else " ")
    print "[{}] Barzilay&Lapata APWS test trees".format(
        "X" if has_apws_test_trees else " ")
 
    print "[{}] Clean APWS train txt".format(
        "X" if has_clean_apws_train else " ")
    print "[{}] Clean APWS test txt".format(
        "X" if has_clean_apws_test else " ")

    print "[{}] Clean APWS train xml".format(
        "X" if has_clean_apws_train_xml else " ")
    print "[{}] Clean APWS test xml".format(
        "X" if has_clean_apws_test_xml else " ")

    print "[{}] clean APWS train tokens".format(
        "X" if has_clean_apws_train_tokens else " ")
    print "[{}] clean APWS test tokens".format(
        "X" if has_clean_apws_test_tokens else " ")
 
    print "[{}] clean APWS train trees".format(
        "X" if has_clean_apws_train_trees else " ")
    print "[{}] clean APWS test trees".format(
        "X" if has_clean_apws_test_trees else " ")
 


    print "[{}] APWS Embeddings".format(
        "X" if has_apws_embeddings else " ")
    
    print "[{}] Clean APWS Embeddings".format(
        "X" if has_clean_apws_embeddings else " ")

    if not has_ntsb_train:
        print "Downloading NTSB training data"
        print "from:\n\t{}\nto:\n\t{} ...".format(
            ntsb_url_train, ntsb_train_tgz)
        download_ntsb(
            ntsb_url_train, ntsb_train_tgz, {"training": "ntsb_train"})

    if not has_ntsb_test:
        print "Downloading NTSB testing data"
        print "from:\n\t{}\nto:\n\t{} ...".format(
            ntsb_url_test, ntsb_test_tgz)
        download_ntsb(
            ntsb_url_test, ntsb_test_tgz, {"ntsb-test": "ntsb_test"})

    if not has_ntsb_train_xml:
        print "Processing NTSB training data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                ntsb_train_tgz, ntsb_train_xml_tgz)
        make_xml(ntsb_train_tgz, ntsb_train_xml_tgz, n_procs=n_procs, mem=mem)

    if not has_ntsb_test_xml:
        print "Processing NTSB testing data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                ntsb_test_tgz, ntsb_test_xml_tgz)
        make_xml(
            ntsb_test_tgz, ntsb_test_xml_tgz, n_procs=n_procs, mem=mem)

    if not has_ntsb_train_tokens:
        print "Writing NTSB training data as tokens."
        print "to:\n\t{} ...".format(ntsb_train_tokens_tgz)
        make_tokens(u"ntsb", False, u"train", ntsb_train_tokens_tgz)

    if not has_ntsb_test_tokens:
        print "Writing NTSB testing data as tokens."
        print "to:\n\t{} ...".format(ntsb_test_tokens_tgz)
        make_tokens(u"ntsb", False, u"test", ntsb_test_tokens_tgz)

    if not has_ntsb_train_trees:
        print "Writing NTSB training data as trees."
        print "to:\n\t{} ...".format(ntsb_train_trees_tgz)
        make_trees(u"ntsb", False, u"train", ntsb_train_trees_tgz)

    if not has_ntsb_test_trees:
        print "Writing NTSB testing data as trees."
        print "to:\n\t{} ...".format(ntsb_test_trees_tgz)
        make_trees(u"ntsb", False, u"test", ntsb_test_trees_tgz)




#    if not has_ntsb_embeddings:
#        print "Learning ntsb word embeddings"
#        print "to:\n\t {} ...".format(ntsb_embeddings) 
#        make_word_embeddings(ntsb_embeddings, "ntsb", False, embed_iters)
 
    if not has_apws_train:
        print "Downloading APWS training data"
        print "from:\n\t{}\nto:\n\t{} ...".format(
            apws_url_train, apws_train_tgz)
        download_apws(
            apws_url_train, apws_train_tgz, {"training": "apws_train"})

    if not has_apws_test:
        print "Downloading APWS testing data"
        print "from:\n\t{}\nto:\n\t{} ...".format(
            apws_url_test, apws_test_tgz)
        download_apws(
            apws_url_test, apws_test_tgz, {"data1-test": "apws_test"})

    if not has_apws_train_xml:
        print "Processing APWS training data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                apws_train_tgz, apws_train_xml_tgz)
        make_xml(apws_train_tgz, apws_train_xml_tgz, n_procs=n_procs, mem=mem)

    if not has_apws_test_xml:
        print "Processing APWS testing data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                apws_test_tgz, apws_test_xml_tgz)
        make_xml(
            apws_test_tgz, apws_test_xml_tgz, n_procs=n_procs, mem=mem)

    if not has_apws_train_tokens:
        print "Writing APWS training data as tokens."
        print "to:\n\t{} ...".format(apws_train_tokens_tgz)
        make_tokens(u"apws", False, u"train", apws_train_tokens_tgz)

    if not has_apws_test_tokens:
        print "Writing APWS testing data as tokens."
        print "to:\n\t{} ...".format(apws_test_tokens_tgz)
        make_tokens(u"apws", False, u"test", apws_test_tokens_tgz)

    if not has_apws_train_trees:
        print "Writing APWS training data as trees."
        print "to:\n\t{} ...".format(apws_train_trees_tgz)
        make_trees(u"apws", False, u"train", apws_train_trees_tgz)

    if not has_apws_test_trees:
        print "Writing APWS testing data as trees."
        print "to:\n\t{} ...".format(apws_test_trees_tgz)
        make_trees(u"apws", False, u"test", apws_test_trees_tgz)

    if not has_clean_apws_train:
        print "Extracting clean APWS training data"
        print "from: \n\t{}\nto:\n\t{} ...".format(
            apws_train_tgz, clean_apws_train_tgz)
        clean_apws_data(apws_train_tgz, clean_apws_train_tgz)

    if not has_clean_apws_test:
        print "Extracting clean APWS testing data"
        print "from: \n\t{}\nto:\n\t{} ...".format(
            apws_test_tgz, clean_apws_test_tgz)
        clean_apws_data(apws_test_tgz, clean_apws_test_tgz)

    if not has_clean_apws_train_xml:
        print "Processing clean APWS training data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                clean_apws_train_tgz, clean_apws_train_xml_tgz)
        make_xml(
            clean_apws_train_tgz, clean_apws_train_xml_tgz, 
            n_procs=n_procs, mem=mem)

    if not has_clean_apws_test_xml:
        print "Processing clean APWS testing data w/ CoreNLP pipeline"
        print "from:\n\t{}\nto:\n\t{} ...".format(
                clean_apws_test_tgz, clean_apws_test_xml_tgz)
        make_xml(
            clean_apws_test_tgz, clean_apws_test_xml_tgz, 
            n_procs=n_procs, mem=mem)

    if not has_clean_apws_train_tokens:
        print "Writing clean APWS training data as tokens."
        print "to:\n\t{} ...".format(clean_apws_train_tokens_tgz)
        make_tokens(u"apws", True, u"train", clean_apws_train_tokens_tgz)

    if not has_clean_apws_test_tokens:
        print "Writing clean APWS testing data as tokens."
        print "to:\n\t{} ...".format(clean_apws_test_tokens_tgz)
        make_tokens(u"apws", True, u"test", clean_apws_test_tokens_tgz)

    if not has_clean_apws_train_trees:
        print "Writing clean APWS training data as trees."
        print "to:\n\t{} ...".format(clean_apws_train_trees_tgz)
        make_trees(u"apws", True, u"train", clean_apws_train_trees_tgz)

    if not has_clean_apws_test_trees:
        print "Writing clean APWS testing data as trees."
        print "to:\n\t{} ...".format(clean_apws_test_trees_tgz)
        make_trees(u"apws", True, u"test", clean_apws_test_trees_tgz)




#    if not has_apws_embeddings:
#        print "Learning apws word embeddings"
#        print "to:\n\t {} ...".format(apws_embeddings) 
#        make_word_embeddings(apws_embeddings, "apws", False, embed_iters)
#    
#    if not has_clean_apws_embeddings:
#        print "Learning clean apws word embeddings"
#        print "to:\n\t {} ...".format(clean_apws_embeddings) 
#        make_word_embeddings(clean_apws_embeddings, "apws", True, embed_iters)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-procs', default=2, type=int)
    parser.add_argument('--mem', default="4G", type=str)
    parser.add_argument('--embed-iters', default=1000, type=int)

    args = parser.parse_args()
    n_procs = args.n_procs
    mem = args.mem
    embed_iters = args.embed_iters
    main(n_procs=n_procs, mem=mem, embed_iters=embed_iters)

                             
