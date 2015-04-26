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
import gensim
import collections


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

def preprocess_barzilay_apws(data_dir, corpus):
        
        
#        path_to_apws_tgz, 
#        
#                             path_to_xml_train_perm_pkl_tgz,
#                             path_to_xml_dev_perm_pkl_tgz,
#                             path_to_xml_test_perm_pkl_tgz,
#                             path_to_doc_train_perm_pkl_tgz,
#                             path_to_doc_dev_perm_pkl_tgz,
#                             path_to_doc_test_perm_pkl_tgz,
#                             path_to_doc_train_pkl_tgz,
#                             path_to_doc_dev_pkl_tgz,
#                             path_to_doc_test_pkl_tgz):

    orig_path = os.path.join(data_dir, corpus + ".tar.gz")
    clean_tgz = os.path.join(data_dir, corpus + "_clean.tar.gz")
    clean_no_meta_tgz = os.path.join(
        data_dir, corpus + "_clean_no_meta.tar.gz")

#    u"data/barzilay_apws_clean.tar.gz",
#                             u"data/barzilay_apws_clean_xml_perm_train.pkl.gz",
#                             u"data/barzilay_apws_clean_xml_perm_dev.pkl.gz",
#                             u"data/barzilay_apws_clean_xml_perm_test.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_perm_train.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_perm_dev.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_perm_test.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_train.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_dev.pkl.gz",
#                             u"data/barzilay_apws_clean_doc_test.pkl.gz")
        
        
    test_instances = {}
    train_instances = {}

    train_instances_no_meta = {}
    test_instances_no_meta = {}

    with tarfile.open(orig_path, "r:gz") as tar:
        for tarinfo in tar:
            if tarinfo.isreg():
                if "perm" not in os.path.basename(tarinfo.name):
                    continue
                instance_name, _, perm = os.path.basename(
                    tarinfo.name).split(".")
                items = os.path.basename(tarinfo.name).split("-")    
                instance_name = items[0]
                
                perm_no = re.search(
                    r"perm-(\d+)", 
                    os.path.basename(tarinfo.name)).groups(1)[0]
                perm_no = int(perm_no)

                partition = os.path.split(os.path.dirname(tarinfo.name))[1]
                f = tar.extractfile(tarinfo)
                text = []
                for line in f:
                    text.append(line.split(" ", 1)[1])
                f.close()
                text = ''.join(text).strip()

                assert partition == u"test" or partition == u"train"
                if partition == u"test":
                    data_dict = test_instances
                    data_dict_no_meta = test_instances_no_meta
                else:
                    data_dict = train_instances
                    data_dict_no_meta = train_instances_no_meta

                if instance_name not in data_dict:
                    data_dict[instance_name] = {
                        u"gold": None, u"perms": list()}
                if instance_name not in data_dict_no_meta:
                    data_dict_no_meta[instance_name] = {
                        u"gold": None, u"perms": list()}
                if perm_no == 1:
                    data_dict[instance_name][u"gold"] = text
                    no_meta_text = remove_apws_meta(text)
                    data_dict_no_meta[instance_name][u"gold"] = no_meta_text
                else:
                    data_dict[instance_name][u"perms"].append( 
                        (perm_no, text))
                    no_meta_text = remove_apws_meta(text)
                    data_dict_no_meta[instance_name][u"perms"].append(
                        (perm_no, no_meta_text))

    for name in train_instances.keys():
        assert len(train_instances[name]["perms"]) <= 20

    for name in test_instances.keys():
        assert len(test_instances[name]["perms"]) <= 20

    def write_partition(tar, instances, label, root_label):    
        for instance_name in sorted(instances.keys()):
            instance = instances[instance_name]
            path = os.path.join(root_label, label, instance_name)
            info = tarfile.TarInfo(name=path)
            info.type = tarfile.DIRTYPE
            info.mode = 0755
            info.mtime = time.time()
            tar.addfile(tarinfo=info)

            gold_path = os.path.join(path, "gold.txt")
            gold_text = StringIO(instance[u"gold"])
            info = tarfile.TarInfo(name=gold_path)
            info.size=len(gold_text.buf)
            info.mtime = time.time()
            tar.addfile(tarinfo=info, fileobj=gold_text) 
            
            perms = instance[u"perms"]
            perms.sort(key=lambda x: x[0])
            for perm_no, perm in perms:
                perm_path = os.path.join(
                    path, "perms", "{}.txt".format(perm_no))
                perm_text = StringIO(perm)
                info = tarfile.TarInfo(name=perm_path)
                info.size = len(perm_text.buf)
                info.mtime = time.time()
                tar.addfile(tarinfo=info, fileobj=perm_text) 

    with tarfile.open(clean_tgz, u"w:gz") as tar:
        print "Writing to {} ...".format(clean_tgz)
        print "Writing train clean docs/perms..."
        write_partition(tar, train_instances, "train", corpus + "_clean")
        print "Writing test apws clean docs/perms..."
        write_partition(tar, test_instances, "test", corpus + "_clean")

    with tarfile.open(clean_no_meta_tgz, u"w:gz") as tar:
        print "Writing to {} ...".format(clean_no_meta_tgz)
        print "Writing train clean docs/perms w/o meta..."
        write_partition(tar, train_instances_no_meta, 
            "train", corpus + "_clean_no_meta")
        print "Writing test clean docs/perms w/o meta..."
        write_partition(tar, test_instances_no_meta, 
            "test", corpus + "_clean_no_meta")

    import sys
    sys.exit()
    corenlp_props = {"ssplit.newlineIsSentenceBreak": "always"}
    anns = ["tokenize", "ssplit",] # "pos", "lemma", "ner", "parse"]

    

    def ann(instances, pipeline, n_procs=2):
        xml_data = []
        
        keys = []
        texts = []
        for instance_name, instance in instances:
            keys.append(instance_name)
            texts.append(instance[u"gold"])
            for perm_no, perm in instance[u"perm"]:
                keys.append((perm_no, instance_name))
                texts.append(perm)
             
        for key, xml in pipeline.annotate_mp_iter_unordered(
                texts, keys=keys, n_procs=n_procs):

            print key
        
#        doc_data = []
#        instances = instances.items()
#        random.shuffle(instances)
#        n_instances = len(instances)
#        for n_instance, (instance_name, instance) in enumerate(instances, 1):
#            gold_text = instance[u"gold"]
#            print "{}/{}".format(n_instance, n_instances), instance_name
#            gold_xml = client.annotate(gold_text, return_xml=True)
#            perm_texts = [perm for perm_no, perm in instance[u"perms"]] 
#            perm_xmls = client.annotate_mp(
#                perm_texts, return_xml=True, n_procs=2)
#      
#            xml_data.append({u"gold": gold_xml, u"perms": perm_xmls})
#            gold_doc = corenlp.read_xml(StringIO(gold_xml.encode(u"utf-8")))
#
#            perm_bufs = [StringIO(perm_xml.encode(u"utf-8"))
#                         for perm_xml in perm_xmls]
#            perm_docs = [corenlp.read_xml(perm_buf) for perm_buf in perm_bufs]
#            doc_data.append({u"gold": gold_doc, u"perms": perm_docs})
#        return xml_data, doc_data

    with corenlp.Server(mem=u'3G', annotators=anns, threads=2,
                        corenlp_props=corenlp_props) as client:
 
        print "Annotating test data..."
        test_xml_perm_data, test_doc_perm_data = ann(test_instances, client)
        import sys
        sys.exit()
        print "Writing test data..."
        with gzip.open(path_to_xml_test_perm_pkl_tgz, u"w") as f:
            pickle.dump(test_xml_perm_data, f)
        with gzip.open(path_to_doc_test_perm_pkl_tgz, u"w") as f:
            pickle.dump(test_doc_perm_data, f)
        with gzip.open(path_to_doc_test_pkl_tgz, u"w") as f:
            pickle.dump([inst[u"gold"] for inst in test_doc_perm_data], f)

        print "Annotating train data..."
        train_xml_perm_data, train_doc_perm_data = ann(train_instances, client)
#        dev_xml_perm_data = train_xml_perm_data[90:]
#        dev_doc_perm_data = train_doc_perm_data[90:]
#        train_xml_perm_data = train_xml_perm_data[:90]
#        train_doc_perm_data = train_doc_perm_data[:90]

        print "Writing training data..."
        with gzip.open(path_to_xml_train_perm_pkl_tgz, u"w") as f:
            pickle.dump(train_xml_perm_data, f)
        with gzip.open(path_to_doc_train_perm_pkl_tgz, u"w") as f:
            pickle.dump(train_doc_perm_data, f)
        with gzip.open(path_to_doc_train_pkl_tgz, u"w") as f:
            pickle.dump([inst[u"gold"] for inst in train_doc_perm_data], f)

     #   print "Writing dev data..."
     #   with gzip.open(path_to_xml_dev_perm_pkl_tgz, u"w") as f:
     #       pickle.dump(dev_xml_perm_data, f)
     #   with gzip.open(path_to_doc_dev_perm_pkl_tgz, u"w") as f:
     #       pickle.dump(dev_doc_perm_data, f)
     #   with gzip.open(path_to_doc_dev_pkl_tgz, u"w") as f:
     #       pickle.dump([inst[u"gold"] for inst in dev_doc_perm_data], f)


def preprocess_barzilay_ntsb(path_to_ntsb_tgz, path_to_ntsb_clean_tgz,
                             path_to_xml_train_perm_pkl_tgz, 
                             path_to_xml_dev_perm_pkl_tgz, 
                             path_to_xml_test_perm_pkl_tgz, 
                             path_to_doc_train_perm_pkl_tgz, 
                             path_to_doc_dev_perm_pkl_tgz, 
                             path_to_doc_test_perm_pkl_tgz, 
                             path_to_doc_train_pkl_tgz, 
                             path_to_doc_dev_pkl_tgz, 
                             path_to_doc_test_pkl_tgz):

    test_instances = {}
    train_instances = {}

    with tarfile.open(path_to_ntsb_tgz, "r:gz") as tar:
        for tarinfo in tar:
            if tarinfo.isreg():
                if "perm" not in os.path.basename(tarinfo.name):
                    continue
                instance_name, _, perm = os.path.basename(
                    tarinfo.name).split(".")
                perm_no = int(perm.split("-")[1])
                
                partition = os.path.split(os.path.dirname(tarinfo.name))[1]

                f = tar.extractfile(tarinfo)
                text = []
                for line in f:
                    assert instance_name in line
                    text.append(line.split(" ", 1)[1])
                f.close()
                text = ''.join(text).strip()
                assert partition == u"test" or partition == u"train"
                if partition == u"test":
                    data_dict = test_instances
                else:
                    data_dict = train_instances
                if instance_name not in data_dict:
                    data_dict[instance_name] = {
                        u"gold": None, u"perms": list()}
                if perm_no == 1:
                    data_dict[instance_name][u"gold"] = text
                    print text
                    print
                else:
                    data_dict[instance_name][u"perms"].append( 
                        (perm_no, text))
    for name in train_instances.keys():
        assert len(train_instances[name]["perms"]) <= 20
    for name in test_instances.keys():
        assert len(test_instances[name]["perms"]) <= 20

    import sys
    sys.exit()
    def write_partition(tar, instances, label):    
        for instance_name, instance in instances.items():
            path = os.path.join("barzilay_ntsb_clean", label, instance_name)
            info = tarfile.TarInfo(name=path)
            info.type = tarfile.DIRTYPE
            info.mode = 0755
            info.mtime = time.time()
            tar.addfile(tarinfo=info)

            gold_path = os.path.join(path, "gold.txt")
            gold_text = StringIO(instance[u"gold"])
            info = tarfile.TarInfo(name=gold_path)
            info.size=len(gold_text.buf)
            info.mtime = time.time()
            tar.addfile(tarinfo=info, fileobj=gold_text) 
            
            perms = instance[u"perms"]
            perms.sort(key=lambda x: x[0])
            for perm_no, perm in perms:
                perm_path = os.path.join(
                    path, "perms", "{}.txt".format(perm_no))
                perm_text = StringIO(perm)
                info = tarfile.TarInfo(name=perm_path)
                info.size=len(perm_text.buf)
                info.mtime = time.time()
                tar.addfile(tarinfo=info, fileobj=perm_text) 

    with tarfile.open(path_to_ntsb_clean_tgz, u"w:gz") as tar:
        print "Writing train ntsb clean docs/perms..."
        write_partition(tar, train_instances, "train")
        print "Writing test ntsb clean docs/perms..."
        write_partition(tar, test_instances, "test")

    corenlp_props = {"ssplit.newlineIsSentenceBreak": "always"}

    annotators = ["tokenize", "ssplit", "pos", "lemma", "ner", "depparse"]
    corenlp.server.start(corenlp_props=corenlp_props, mem="32G",
                         annotators=annotators, threads=8)
    client = corenlp.client.CoreNLPClient()

    def ann(instances):
        xml_data = []
        doc_data = []
        instances = instances.items()
        n_instances = len(instances)
        random.shuffle(instances)
        for n_instance, (instance_name, instance) in enumerate(instances, 1):
            print "{}/{}".format(n_instance, n_instances), instance_name
            gold_text = instance[u"gold"]
            gold_xml = client.annotate(gold_text, return_xml=True)
            perm_texts = [perm for perm_no, perm in instance[u"perms"]] 
            perm_xmls = client.annotate_mp(
                perm_texts, return_xml=True, n_procs=2)       
            
            xml_data.append({u"gold": gold_xml, u"perms": perm_xmls})
            gold_doc = corenlp.read_xml(StringIO(gold_xml.encode(u"utf-8")))
            
            perm_bufs = [StringIO(perm_xml.encode(u"utf-8"))
                         for perm_xml in perm_xmls]
            perm_docs = [corenlp.read_xml(perm_buf) for perm_buf in perm_bufs]
            doc_data.append({u"gold": gold_doc, u"perms": perm_docs})
        return xml_data, doc_data

    print "Annotating test data..."
    test_xml_perm_data, test_doc_perm_data = ann(test_instances) 
    print "Writing test data..."
    with gzip.open(path_to_xml_test_perm_pkl_tgz, u"w") as f:
        pickle.dump(test_xml_perm_data, f)
    with gzip.open(path_to_doc_test_perm_pkl_tgz, u"w") as f:
        pickle.dump(test_doc_perm_data, f)
    with gzip.open(path_to_doc_test_pkl_tgz, u"w") as f:
        pickle.dump([inst[u"gold"] for inst in test_doc_perm_data], f)

    print "Annotating train data..."
    train_xml_perm_data, train_doc_perm_data = ann(train_instances) 
    dev_xml_perm_data = train_xml_perm_data[90:]
    dev_doc_perm_data = train_doc_perm_data[90:]
    train_xml_perm_data = train_xml_perm_data[:90]
    train_doc_perm_data = train_doc_perm_data[:90]

    print "Writing train data..."
    with gzip.open(path_to_xml_train_perm_pkl_tgz, u"w") as f:
        pickle.dump(train_xml_perm_data, f)
    with gzip.open(path_to_doc_train_perm_pkl_tgz, u"w") as f:
        pickle.dump(train_doc_perm_data, f)
    with gzip.open(path_to_doc_train_pkl_tgz, u"w") as f:
        pickle.dump([inst[u"gold"] for inst in train_doc_perm_data], f)

    with gzip.open(path_to_xml_dev_perm_pkl_tgz, u"w") as f:
        pickle.dump(dev_xml_perm_data, f)
    with gzip.open(path_to_doc_dev_perm_pkl_tgz, u"w") as f:
        pickle.dump(dev_doc_perm_data, f)
    with gzip.open(path_to_doc_dev_pkl_tgz, u"w") as f:
        pickle.dump([inst[u"gold"] for inst in dev_doc_perm_data], f)
      
    corenlp.server.stop()

def get_barzilay_clean_docs_only(corpus="ntsb", part="train",
                                 tokens_only=False):
    data_dir = os.getenv("COHERENCE_DATA", "data")
    path = os.path.join(
        data_dir, "barzilay_{}_clean_doc_{}.pkl.gz".format(corpus, part))
    with gzip.open(path, u"r") as f:
        docs = pickle.load(f)
    if tokens_only:
        token_only_docs = [[[unicode(t).lower() for t in sent]
                            for sent in doc]
                           for doc in docs] 
        return token_only_docs
    else:
        return docs

def get_barzilay_clean_docs_perms(corpus="ntsb", part="train", 
                                  tokens_only=False):
    data_dir = os.getenv("COHERENCE_DATA", "data")
    path = os.path.join(
        data_dir, "barzilay_{}_clean_doc_perm_{}.pkl.gz".format(corpus, part))
    with gzip.open(path, u"r") as f:
        docs_perms = pickle.load(f)
    if tokens_only is True:
        for doc_perm in docs_perms:
            doc = doc_perm["gold"]
            doc_perm["gold"] = [[unicode(t).lower() for t in sent]
                                for sent in doc_perm["gold"]]
            doc_perm["perms"] = [[[unicode(t).lower() for t in sent]
                                 for sent in perm]
                                 for perm in doc_perm["perms"]]
            
    return docs_perms

def get_barzilay_data(corpus=u"apws", part=u"train", 
                      clean=False, format=u"document", include_perms=True,
                      convert_brackets=False):

    assert format in [u"text", u"xml", u"document", u"tokens", u"trees"]

    fname = u"{}_{}_{}{}.tar.gz".format(
        u"b&l" if clean is False else u"clean",
        corpus,
        part, 
        u"" if format == u"text" else u"_xml")
    key_name = u"{}{}_{}{}".format(
        u"" if clean is False else u"clean_",
        corpus,
        part,
        u"" if format == u"text" else u"_xml")
    
    path = os.path.join(os.getenv(u"COHERENCE_DATA", u"."), fname)

    def read_document(f):
        return corenlp.read_xml(f, convert_brackets=convert_brackets)
    def read_tokens(f):
        doc = corenlp.read_xml(f, convert_brackets=convert_brackets)
        return [[unicode(token).lower() for token in sent]
                for sent in doc]
    def read_parse(f):
        doc = corenlp.read_xml(f, convert_brackets=convert_brackets)
        return [sent.parse for sent in doc]

    if format == u"text" or format == u"xml":
        reader = None
    elif format == u"document":
        reader = read_document
    elif format == u"tokens":
        reader = read_tokens
    elif format == u"trees":
        reader = read_parse

    data = read_tar(
        path, text_filter=reader, file_ext=".xml", no_perm_num=True)
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
            os.system("rm rmp")

    data = {}
    with tarfile.open(fileobj=fileobj, mode="r") as tar:
        for tarinfo in tar:
            if "perm" not in os.path.basename(tarinfo.name):
                continue
            instance_name, _, perm = os.path.basename(
                tarinfo.name).split(".")
            items = os.path.basename(tarinfo.name).split("-")    
            instance_name = items[0]
            
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

    clean_apws_train_xml_tgz = os.path.join(
        data_dir, "clean_apws_train_xml.tar.gz")
    has_clean_apws_train_xml = os.path.exists(clean_apws_train_xml_tgz)

    clean_apws_test_xml_tgz = os.path.join(
        data_dir, "clean_apws_test_xml.tar.gz")
    has_clean_apws_test_xml = os.path.exists(clean_apws_test_xml_tgz)

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
    
    print "[{}] Clean APWS train txt".format(
        "X" if has_clean_apws_train else " ")
    print "[{}] Clean APWS test txt".format(
        "X" if has_clean_apws_test else " ")

    print "[{}] Clean APWS train xml".format(
        "X" if has_clean_apws_train_xml else " ")
    print "[{}] Clean APWS test xml".format(
        "X" if has_clean_apws_test_xml else " ")

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

    if not has_ntsb_embeddings:
        print "Learning ntsb word embeddings"
        print "to:\n\t {} ...".format(ntsb_embeddings) 
        make_word_embeddings(ntsb_embeddings, "ntsb", False, embed_iters)
 
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

    if not has_apws_embeddings:
        print "Learning apws word embeddings"
        print "to:\n\t {} ...".format(apws_embeddings) 
        make_word_embeddings(apws_embeddings, "apws", False, embed_iters)
    
    if not has_clean_apws_embeddings:
        print "Learning clean apws word embeddings"
        print "to:\n\t {} ...".format(clean_apws_embeddings) 
        make_word_embeddings(clean_apws_embeddings, "apws", True, embed_iters)

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

                             
