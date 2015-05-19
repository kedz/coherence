from nltk.tree import Tree
import numpy as np
from sklearn.base import BaseEstimator
from cohere.data import CoherenceData

class TokensTransformer(BaseEstimator):
    def __init__(self, embeddings, window_size=3):
        self.embeddings = embeddings
        self.window_size = window_size

    def _get_max_sent_len(self, dataset):
        max_sent = 0
        for doc in dataset.gold:
            for sent in doc:
                max_sent = max(max_sent, len(sent))
        return max_sent

    @classmethod
    def _make_docs_safe(self, docs):
        assert isinstance(docs, list) or isinstance(docs, tuple)
        assert len(docs) > 0
        if not (isinstance(docs[0], list) or isinstance(docs[0], tuple)):
            docs = [docs]
        return docs

    def transform_gold(self, dataset):
        if not isinstance(dataset, CoherenceData):
            raise Exception(u"Argument dataset must be a CoherenceData object")
        if not dataset.format == u"tokens":
            raise Exception(u"Argument dataset must be of format 'tokens'.")

        max_sent = self._get_max_sent_len(dataset)

        I = np.cumsum(
            np.array(
                [0] + [len(doc) for doc in dataset.gold],
                dtype=u"int32"))
        X = -1 * np.ones((I[-1], max_sent), dtype="int32")       
        
        for i, doc in enumerate(dataset.gold):
            self._transform_single_doc(X, I[i], doc)

        return X, I

    def transform_test(self, dataset):
        if not isinstance(dataset, CoherenceData):
            raise Exception(u"Argument dataset must be a CoherenceData object")
        if not dataset.format == u"tokens":
            raise Exception(u"Argument dataset must be of format 'tokens'.")

        max_sent = self._get_max_sent_len(dataset)
        
        sent2index = {}
        sents = []
        P = []
        I = [0]
        offset = 0
        sent_offset = 0
        for inst in dataset:
            gold_off = offset
            offset += 1
            I.append(len(inst.gold))
            for sent in inst.gold:
                as_tuple = tuple(sent)
                if as_tuple not in sent2index:
                    sent2index[as_tuple] = sent_offset
                    sent_offset += 1
                    sents.append(as_tuple)
           
            for perm in inst.perms:
                P.append([gold_off, offset])
                I.append(len(perm))
                offset += 1
                for sent in perm:
                    as_tuple = tuple(sent)
                    if as_tuple not in sent2index:
                        sent2index[as_tuple] = sent_offset
                        sent_offset += 1
                        sents.append(as_tuple)

        n_sents = len(sents)
        X = np.ones((n_sents, max_sent), dtype="int32") * -1
        self._transform_single_doc(X, 0, sents)
        I = np.cumsum(np.array(I), dtype="int32")
        P = np.array(P, dtype="int32") 

        C = np.ones((I[-1], self.window_size), dtype="int32") * -1
        offset = 0
        pad = self.window_size / 2
        for inst in dataset:
            gold = [sent2index[tuple(sent)] for sent in inst.gold]
            dlen = len(gold)
            for i in xrange(dlen): 
                for pos, k in enumerate(xrange(i - pad, i + 1 + pad)):
                    if k >= 0 and k < dlen:
                        C[offset, pos] = gold[k]
                offset += 1
            for perm_doc in inst.perms:
                perm = [sent2index[tuple(sent)] for sent in perm_doc]
                plen = len(perm)
                for i in xrange(plen):
                    for pos, k in enumerate(xrange(i - pad, i + 1 + pad)):
                        if k >= 0 and k < plen:
                            C[offset, pos] = perm[k]
                    offset += 1

        S = np.concatenate([I[0:-1].reshape((I.shape[0] - 1, 1)), 
                            I[1:].reshape((I.shape[0] - 1, 1))], axis=1)
        return X, C, S, P


    def _transform_single_doc(self, X, row_offset, doc):
        as_indices = [self.embeddings.indices(tokens) for tokens in doc]
        for i, indices in enumerate(as_indices):
            X[row_offset + i, 0 : len(indices)] = indices

class TreeTransformer(object):
    def __init__(self, embeddings, batch_size=25, window_size=3):
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.window_size = window_size

    def _get_max_sent_and_ops(self, dataset):
        max_sent = 0
        max_op = 0    
        for doc in dataset.gold:
            max_sent = max(max_sent, max([len(tree.leaves()) for tree in doc]))
            op_lens = [len(self._get_command_sequence(tree)) for tree in doc]
            max_op = max(max_op, max(op_lens))
        return max_sent, max_op

    def transform_gold(self, dataset):
        if not isinstance(dataset, CoherenceData):
            raise Exception(u"Argument dataset must be a CoherenceData object")
        if not dataset.format == u"trees":
            raise Exception(u"Argument dataset must be of format 'trees'.")
        
        max_sent, max_ops = self._get_max_sent_and_ops(dataset)

        I = np.cumsum(
            np.array(
                [0] + [len(doc) for doc in dataset.gold],
                dtype=u"int32"))
        n_sents = I[-1]
        X = -1 * np.ones((n_sents, max_sent), dtype="int32")       
        O = np.zeros((n_sents, max_ops, 6), dtype="int32")
        O[:,:,0] = 1
        
        for i, doc in enumerate(dataset.gold):
            self._transform_single_doc(X, O, I[i], doc)

        C, y, S, n_batches = self._transform_training_cliques(I)

        return X, O, C, y, S, n_batches

    def _transform_single_doc(self, X, O, row_offset, doc):
        
        as_tokens = [[word.lower() for word in tree.leaves()]
                     for tree in doc]

        as_indices = [self.embeddings.indices(tokens) for tokens in as_tokens]
        as_ops = [self._get_command_sequence(tree) for tree in doc]

        for i, indices in enumerate(as_indices):
            n_ops = len(as_ops[i])
            X[row_offset + i, 0 : len(indices)] = indices
            if len(as_ops[i]) > 0:
                O[row_offset + i, 0 : n_ops, :] = as_ops[i]

    def _transform_training_cliques(self, I):
        n_docs = I.shape[0] - 1
        n_cliques = np.sum((I[d + 1] - I[d])**2 for d in xrange(n_docs))
        C = np.ones((n_cliques, self.window_size), dtype="int32")
        y = np.ones((n_cliques,), dtype="int32")
        
        win_pad = self.window_size / 2
        center = win_pad

        offset = 0
        for d in xrange(n_docs):
            doc_size = I[d + 1] - I[d]
            for i in xrange(doc_size):
                for j in xrange(doc_size):
                    for pos, k in enumerate(
                            xrange(i - win_pad, i + win_pad + 1)):
                        if k < 0 or k >= doc_size:
                            C[offset,pos] = -1
                        elif pos == center:
                            C[offset,pos] = I[d] + j
                            y[offset] = 1 if i == j else 0
                        else:
                            C[offset,pos] = I[d] + k
                    offset += 1

        n_batches = C.shape[0] / self.batch_size
        n_batches += 1 if C.shape[0] % self.batch_size != 0 else 0
        S = []

        for b in xrange(n_batches):
            B = C[b * self.batch_size : (b + 1) * self.batch_size]
            s, B_inv = np.unique(B, return_inverse=True)
            if s[0] == -1:
                s = s[1:]
                B_inv -= 1
            B_inv = B_inv.reshape(B.shape)
            S.append(s)
            C[b * self.batch_size : (b + 1) * self.batch_size] = B_inv
        return C, y, S, n_batches



    def _get_command_sequence(self, tree, commands=None):
        # Initial case, convert tree to binary, and start recursion.
        if commands is None:
            commands = []
            tree = tree.copy(deep=True)
            tree.chomsky_normal_form(factor="left")
            word_position = 0
            for position in tree.treepositions():
                if not isinstance(tree[position], Tree):
                    tree[position] = word_position
                    word_position += 1
            cmds = []
            self._get_command_sequence(tree, commands=cmds)
            return cmds

        # Recursive case, with 3 cases:
        else:
            # Base case, this is a preterminal node, return 
            # child.
            if isinstance(tree, Tree) and not isinstance(tree[0], Tree):
                return (1, tree[0])
            # Recursive case with two children, add a command sequence.
            elif len(tree) == 2:
                left_source, left_position = self._get_command_sequence(
                    tree[0], commands=commands)
                right_source, right_position = self._get_command_sequence(
                    tree[1], commands=commands)
                commands.append([0, min(left_position, right_position),
                                 left_source, left_position,
                                 right_source, right_position])
                return [0, min(left_position, right_position)]
            # Recursive case with one child, propagate child up the tree.
            else:
                return self._get_command_sequence(tree[0], commands=commands)

    def transform_test(self, dataset):
        if not isinstance(dataset, CoherenceData):
            raise Exception(u"Argument dataset must be a CoherenceData object")
        if not dataset.format == u"trees":
            raise Exception(u"Argument dataset must be of format 'trees'.")

        max_sent, max_ops = self._get_max_sent_and_ops(dataset)
        
        leaves2index = {}
        trees = []
        P = []
        I = [0]
        offset = 0
        tree_offset = 0
        for inst in dataset:
            gold_off = offset
            offset += 1
            I.append(len(inst.gold))
            for tree in inst.gold:
                as_tuple = tuple(tree.leaves())
                if as_tuple not in leaves2index:
                    leaves2index[as_tuple] = tree_offset
                    tree_offset += 1
                    trees.append(tree)
           
            for perm in inst.perms:
                P.append([gold_off, offset])
                I.append(len(perm))
                offset += 1
                for tree in perm:
                    as_tuple = tuple(tree.leaves())
                    if as_tuple not in leaves2index:
                        leaves2index[as_tuple] = tree_offset
                        tree_offset += 1
                        trees.append(tree)
        
        n_trees = len(trees)
        X = np.ones((n_trees, max_sent), dtype="int32") * -1
        O = np.zeros((n_trees, max_ops, 6), dtype="int32")
        O[:,:,0] = 1

        self._transform_single_doc(X, O, 0, trees)
        I = np.cumsum(np.array(I), dtype="int32")
        P = np.array(P, dtype="int32") 

        C = np.ones((I[-1], self.window_size), dtype="int32") * -1

        offset = 0
        pad = self.window_size / 2
        for inst in dataset:
            gold = [leaves2index[tuple(tree.leaves())] for tree in inst.gold]
            dlen = len(gold)
            for i in xrange(dlen): 
                for pos, k in enumerate(xrange(i - pad, i + 1 + pad)):
                    if k >= 0 and k < dlen:
                        C[offset, pos] = gold[k]
                offset += 1
            for perm_doc in inst.perms:
                perm = [leaves2index[tuple(tree.leaves())] 
                        for tree in perm_doc]
                plen = len(perm)
                for i in xrange(plen):
                    for pos, k in enumerate(xrange(i - pad, i + 1 + pad)):
                        if k >= 0 and k < plen:
                            C[offset, pos] = perm[k]
                    offset += 1

        S = np.concatenate([I[0:-1].reshape((I.shape[0] - 1, 1)), 
                            I[1:].reshape((I.shape[0] - 1, 1))], axis=1)
        return X, O, C, S, P





class TreeTransformer2(object):
    def __init__(self, embeddings, window_size=3, max_sent_len=120, 
                 max_ops_len=15):
        self.embeddings = embeddings
        self.window_size = window_size
        self.max_sent_len = max_sent_len
        self.max_ops_len = max_ops_len

    @classmethod        
    def get_max_sent_and_ops(self, docs):
        max_sent = 0
        max_op = 0    
        for doc in docs:
            max_sent = max(max_sent, max([len(tree.leaves()) for tree in doc]))
            op_lens = [len(self._get_command_sequence(tree)) for tree in doc]
            max_op = max(max_op, max(op_lens))
        return max_sent, max_op


    def transform(self, docs):
        n_sents = np.sum([len(doc) for doc in docs])
        pad_sym = -1 #self.embeddings.get_index("__PAD__")
        X_idx_sent = pad_sym * np.ones(
            (n_sents, self.max_sent_len), dtype=np.int64)
        O_sent = np.zeros((n_sents, self.max_ops_len, 6), np.int32)
        O_sent[:,:,0] = 1
        row_offset = 0
        for doc in docs:
            self._transform_single_doc(X_idx_sent, O_sent, row_offset, doc)
            row_offset += len(doc)
        return X_idx_sent, O_sent 

    def _transform_single_doc(self, X, O, row_offset, doc):
        
        as_tokens = [[word.lower() for word in tree.leaves()]
                     for tree in doc]

        as_indices = [self.embeddings.indices(tokens) for tokens in as_tokens]
        as_ops = [self._get_command_sequence(tree) for tree in doc]

        for i, indices in enumerate(as_indices):
            n_ops = len(as_ops[i])
            X[row_offset + i, 0 : len(indices)] = indices
            if len(as_ops[i]) > 0:
                O[row_offset + i, 0 : n_ops, :] = as_ops[i]

    def window_transform(self, docs):
        if len(docs) == 0:
            raise Exception("Input is an empty list or document")
        elif isinstance(docs[0], Tree):
            docs = [docs]
        
        #print "I AM A", type(docs)
        #if not isinstance(docs, list) and not isinstance(docs, tuple):
         
        pad_size = self.window_size / 2

        max_toks = self.max_sent_len
        max_ops = self.max_ops_len
        pad_sym = -1 

        n_rows = np.sum([len(doc) for doc in docs])
        X_iw = np.ones((n_rows, self.max_sent_len * self.window_size), 
                       dtype=np.int32) * pad_sym
        O_iw = np.zeros((n_rows, self.max_ops_len * self.window_size, 6),
                        dtype=np.int32)
        # Make default instruction the pad instruction (1,0,0,0,0,0).
        O_iw[:,:,0] = 1

        # Transform doc to word index sentence matrix and operation sentence
        # matrix.
        X_is, O_s = self.transform(docs)

        row_offset = 0
        for idx, doc in enumerate(docs):
            doc_len = len(doc)
            X_is_doc = X_is[row_offset : row_offset + doc_len]
            O_s_doc = O_s[row_offset : row_offset + doc_len]
            for i in xrange(0, doc_len):
                for pos, k in enumerate(xrange(i-pad_size, i + pad_size + 1)):
                    if k < 0:
                        pass
                    elif k >= doc_len:
                        pass
                    else:
                        X_iw[row_offset + i, 
                             pos * max_toks : (pos+1) * max_toks] = X_is_doc[k]
                        O_iw[row_offset + i,
                             pos * max_ops : (pos + 1) * max_ops] = O_s_doc[k]
                    
            row_offset += doc_len
        return X_iw, O_iw

    def training_window_transform(self, docs):
        if not isinstance(docs, list) and not isinstance(docs, tuple):
            docs = [docs]
        
        pad_size = self.window_size / 2

        max_toks = self.max_sent_len
        max_ops = self.max_ops_len
        pad_sym = -1 

        n_rows = np.sum([len(doc) * len(doc) for doc in docs])
        X_iw = np.ones((n_rows, self.max_sent_len * self.window_size), 
                       dtype=np.int32) * pad_sym
        O_iw = np.zeros((n_rows, self.max_ops_len * self.window_size, 6),
                        dtype=np.int32)
        y = np.zeros((n_rows,), dtype=np.int32)
        # Make default instruction the pad instruction (1,0,0,0,0,0).
        O_iw[:,:,0] = 1

        # Transform doc to word index sentence matrix and operation sentence
        # matrix.
        X_is, O_s = self.transform(docs)

        row_offset = 0
        input_row_offset = 0
        for idx, doc in enumerate(docs):
            doc_len = len(doc)
            X_is_doc = X_is[input_row_offset : input_row_offset + doc_len]
            O_s_doc = O_s[input_row_offset : input_row_offset + doc_len]
            for i in xrange(0, doc_len):
                for j in xrange(0, doc_len):
                    if i == j:
                        y[row_offset + i * doc_len + j] = 1
                    for pos, k in enumerate(xrange(i-pad_size, i+pad_size+1)):
                        if k < 0:
                            pass
                            #X_iw[row_offset + i, pos * max_toks] = start_sym
                        elif k >= doc_len:
                            pass
                            #X_iw[row_offset + i, pos * max_toks] = stop_sym
                        else:
                            if pos == pad_size:
                                # selecting the focus which is j.
                                l = j
                            else:
                                # select the left or right sentences (k).
                                l = k
                            X_iw[row_offset + i * doc_len + j, 
                                 pos * max_toks:(pos+1)*max_toks] = X_is_doc[l]
                            O_iw[row_offset + i * doc_len + j,
                                 pos * max_ops:(pos + 1)*max_ops] = O_s_doc[l]
                    
            row_offset += doc_len * doc_len
            input_row_offset += doc_len
        return X_iw, O_iw, y

    def testing_window_transform(self, dataset):
        X_iw_gold = []
        O_iw_gold = []
        X_iw_perm = []
        O_iw_perm = []

        for inst in dataset:
            x_iw_gold, o_iw_gold = self.window_transform([inst.gold]) 
            
            for perm in inst.perms:
                x_iw_perms, o_iw_perms = self.window_transform([perm])
                X_iw_gold.append(x_iw_gold)
                O_iw_gold.append(o_iw_gold)
                X_iw_perm.append(x_iw_perms)
                O_iw_perm.append(o_iw_perms)
        return X_iw_gold, O_iw_gold, X_iw_perm, O_iw_perm


    def pprint_token_index_sequences(self, X_is, cutoff=20):
        if len(X_is.shape) == 1:
            X_is = X_is.reshape((1, X_is.shape[0]))
        for row, x_is in enumerate(X_is):
            words = []
            budget = 0
            for i in xrange(self.max_sent_len):
                if x_is[i] == -1:
                    continue
                else:
                    w = self.embeddings.index2token[x_is[i]]
                    words.append(w)
             
            print u"{}] ".format(row) + u" ".join(words)[:cutoff]

    def pprint_token_index_windows(self, X_iw, cutoff=5):
        max_sent = X_iw.shape[1] / self.window_size
        s_len = min(cutoff, max_sent)
        pads = [self.embeddings.token2index.get(u"__PAD__", -1), -1]
        for row, x_iw in enumerate(X_iw):
            sents = []
            for k in xrange(self.window_size):
                words = []
                for i in xrange(k * max_sent, k * max_sent + s_len):
                    if x_iw[i] in pads:
                        continue
                        #words.append(u"__PAD__")
                    else:
                        w = self.embeddings.index2token.get(
                            x_iw[i], u"__UNKNOWN__")
                        words.append(w)
                sents.append(u" ".join(words))
            print u"{}] ".format(row) + u" ... ".join(sents)

    @classmethod
    def _get_command_sequence(self, tree, commands=None):
        # Initial case, convert tree to binary, and start recursion.
        if commands is None:
            commands = []
            tree = tree.copy(deep=True)
            tree.chomsky_normal_form(factor="left")
            word_position = 0
            for position in tree.treepositions():
                if not isinstance(tree[position], Tree):
                    tree[position] = word_position
                    word_position += 1
            cmds = []
            self._get_command_sequence(tree, commands=cmds)
            return cmds

        # Recursive case, with 3 cases:
        else:
            # Base case, this is a preterminal node, return 
            # child.
            if isinstance(tree, Tree) and not isinstance(tree[0], Tree):
                return (1, tree[0])
            # Recursive case with two children, add a command sequence.
            elif len(tree) == 2:
                left_source, left_position = self._get_command_sequence(
                    tree[0], commands=commands)
                right_source, right_position = self._get_command_sequence(
                    tree[1], commands=commands)
                commands.append([0, min(left_position, right_position),
                                 left_source, left_position,
                                 right_source, right_position])
                return [0, min(left_position, right_position)]
            # Recursive case with one child, propagate child up the tree.
            else:
                return self._get_command_sequence(tree[0], commands=commands)
