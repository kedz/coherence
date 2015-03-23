import tarfile
import os
import time
from StringIO import StringIO
import corenlp.server
import corenlp.client
import gzip
import pickle
import random

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
                else:
                    data_dict[instance_name][u"perms"].append( 
                        (perm_no, text))
  

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
        write_partition(tar, train_instances, "train")
        write_partition(tar, test_instances, "test")

    corenlp_props = {"ssplit.newlineIsSentenceBreak": "always"}
    annotators = ["tokenize", "ssplit"]
    corenlp.server.start(corenlp_props=corenlp_props, 
                         annotators=annotators, threads=4)
    client = corenlp.client.CoreNLPClient()

    def ann(instances):
        xml_data = []
        doc_data = []
        instances = instances.items()
        random.shuffle(instances)
        for instance_name, instance in instances:
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

    test_xml_perm_data, test_doc_perm_data = ann(test_instances) 
    with gzip.open(path_to_xml_test_perm_pkl_tgz, u"w") as f:
        pickle.dump(test_xml_perm_data, f)
    with gzip.open(path_to_doc_test_perm_pkl_tgz, u"w") as f:
        pickle.dump(test_doc_perm_data, f)
    with gzip.open(path_to_doc_test_pkl_tgz, u"w") as f:
        pickle.dump([inst[u"gold"] for inst in test_doc_perm_data], f)



    train_xml_perm_data, train_doc_perm_data = ann(train_instances) 
    dev_xml_perm_data = train_xml_perm_data[90:]
    dev_doc_perm_data = train_doc_perm_data[90:]
    train_xml_perm_data = train_xml_perm_data[:90]
    train_doc_perm_data = train_doc_perm_data[:90]

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



def get_barzilay_ntsb_clean_docs_only(part="train", tokens_only=False):
    data_dir = os.getenv("COHERENCE_DATA", "data")
    path = os.path.join(
        data_dir, "barzilay_ntsb_clean_doc_{}.pkl.gz".format(part))
    with gzip.open(path, u"r") as f:
        docs = pickle.load(f)
    if tokens_only:
        token_only_docs = [[[unicode(t).lower() for t in sent]
                            for sent in doc]
                           for doc in docs] 
        return token_only_docs
    else:
        return docs

def get_barzilay_ntsb_clean_docs_perms(part="train"):
    data_dir = os.getenv("COHERENCE_DATA", "data")
    path = os.path.join(
        data_dir, "barzilay_ntsb_clean_doc_perm_{}.pkl.gz".format(part))
    with gzip.open(path, u"r") as f:
        docs = pickle.load(f)
    return docs



if __name__ == "__main__":
    preprocess_barzilay_ntsb(u"data/barzilay_ntsb.tar.gz", 
                             u"data/barzilay_ntsb_clean.tar.gz",
                             u"data/barzilay_ntsb_clean_xml_perm_train.pkl.gz",
                             u"data/barzilay_ntsb_clean_xml_perm_dev.pkl.gz",
                             u"data/barzilay_ntsb_clean_xml_perm_test.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_perm_train.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_perm_dev.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_perm_test.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_train.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_dev.pkl.gz",
                             u"data/barzilay_ntsb_clean_doc_test.pkl.gz")
