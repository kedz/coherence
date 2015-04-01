import numpy as np
#from collections import defaultdict

_legal_input = set([u'filename', 'document', "entitygrid"])

_unigrams = ['X', 'O', 'S', '-']

_bigrams = ['XX', 'XO', 'XS', 'X-',
            'OX', 'OO', 'OS', 'O-',
            'SX', 'SO', 'SS', 'S-',
            '-X', '-O', '-S', '--',
            '0X', '0O', '0S', '0-',
            'X1', 'O1', 'S1', '-1']

_trigrams = ['XXX', 'XXO', 'XXS', 'XX-',
             'XOX', 'XOO', 'XOS', 'XO-',
             'XSX', 'XSO', 'XSS', 'XS-',
             'X-X', 'X-O', 'X-S', 'X--',
             'OXX', 'OXO', 'OXS', 'OX-',
             'OOX', 'OOO', 'OOS', 'OO-',
             'OSX', 'OSO', 'OSS', 'OS-',
             'O-X', 'O-O', 'O-S', 'O--',
             'SXX', 'SXO', 'SXS', 'SX-',
             'SOX', 'SOO', 'SOS', 'SO-',
             'SSX', 'SSO', 'SSS', 'SS-',
             'S-X', 'S-O', 'S-S', 'S--',
             '-XX', '-XO', '-XS', '-X-',
             '-OX', '-OO', '-OS', '-O-',
             '-SX', '-SO', '-SS', '-S-',
             '--X', '--O', '--S', '---',
             '0XX', '0XO', '0XS', '0X-',
             '0OX', '0OO', '0OS', '0O-',
             '0SX', '0SO', '0SS', '0S-',
             '0-X', '0-O', '0-S', '0--',
             'XX1', 'XO1', 'XS1', 'X-1',
             'OX1', 'OO1', 'OS1', 'O-1',
             'SX1', 'SO1', 'SS1', 'S-1',
             '-X1', '-O1', '-S1', '--1']

class EntityGrid(object):
    def __init__(self, entities, ent2roles):
        self.entities = entities
        self.ent2roles = ent2roles
        self.is_salient = {}

        for ent in entities:
            nonzero = [role for role in ent2roles[ent] if role != u'-']
            self.is_salient[ent] = True if len(nonzero) > 2 else False

    def __unicode__(self):
        max_len = max([len(ent) for ent in self.entities])
        fmt_uni = u'{:' + unicode(max_len) + u's} {}'
        ent_list = list(self.entities)
        ent_list.sort()
        unis = []
        for ent in ent_list:
            role_uni = u' '.join(self.ent2roles[ent])
            unis.append(fmt_uni.format(ent.upper(), role_uni))
        return u'\n'.join(unis)

    def __str__(self):
        return unicode(self).encode(u'utf-8')

    def bg_counts(self, normalize=False):
        sal_counts = defaultdict(int)
        counts = defaultdict(int)
        for ent in self.entities:
            roles = self.ent2roles[ent]
            for i, role in enumerate(roles[:-1]):
                if self.is_salient[ent]:
                    trans = u'SAL:{}{}'.format(role, roles[i+1])
                    sal_counts[trans] += 1
                else:
                    trans = u'{}{}'.format(role, roles[i+1])
                    counts[trans] += 1
        
        if normalize is True:
            norm = sum(counts.itervalues())
            sal_norm = sum(sal_counts.itervalues())
            for trans, count in counts.iteritems():
                counts[trans] = count / float(norm)
            for trans, count in sal_counts.iteritems():
                sal_counts[trans] = count / float(sal_norm)
        
        return sal_counts, counts    



class EntityGridTransformer(object):
    
    _level2role = {3: u'S', 2: u'O', 1: u'X', 0: u'-'}
    _role2level = {u'S': 3, u'O': 2, u'X': 1, u'-': 0}
    _subj_types = set([u'nsubj', u'csubj'])
    _obj_types = set([u'csubjpass', u'dobj', u'nsubjpass'])

    def __init__(self, input=u"document", inplace=False):
        if input not in [u"document"]:
            raise NotImplementedError()
        self.input = input
        self.inplace = inplace

    def transform(self, X):
        if self.input == u"document":
            D = X
            return [self._from_document(d) for d in D]

    def transform_D_P(self, D_P):
        if self.input == u"document":
            convert = self._from_document
        E_P = []
        for d_Pd in D_P:
            e_Pe = {}
            e_Pe[u"gold"] = convert(d_Pd[u"gold"])
            e_Pe[u"perms"] = [convert(p) for p in d_Pd[u"perms"]]
            E_P.append(e_Pe)
        return E_P

    def _from_document(self, d):
        ents = set()
        s2ent_role = []
        for i, s in enumerate(d):
            ert = self._entity_role_tuples(s)
            s2ent_role.append(ert)
            for ent in ert.iterkeys():
                ents.add(ent)
        ent2roles = {}
        for ent in ents:
            roles = []
            for ert in s2ent_role:
                role = self._level2role[ert.get(ent, 0)]
                roles.append(role)
            ent2roles[ent] = roles 
        return EntityGrid(ents, ent2roles)

    def _entity_role_tuples(self, s):
        ent_roles = {}
        for t in s:
            if t.is_noun():
                lem = t.lem.lower()
                if ent_roles.get(lem, 0) < self._role2level[u'X']:
                    ent_roles[lem] = self._role2level[u'X']

            elif t.is_verb():                        
                if t.lem == u'be':
                    for ent, role in self._copula_roles(s, t):
                        ent_lem = ent.lem.lower()
                        if ent_roles.get(ent_lem, 0) < self._role2level[role]:
                            ent_roles[ent_lem] = self._role2level[role]
                    continue

                deps = s.gov2deps.get(t, None)
                
                # Check that this token is a verb phrase head  with dependents.
                if deps is None:
                    continue

                # Check for light verbs above clausal complement 
                has_xcomp = ('xcomp' in [rel for rel, dep in deps])
                
                for rel, dep in deps:
                    if dep.is_noun():
                        dlem = dep.lem.lower()
                        
                        if has_xcomp is True:
                            role = u'X'
                        elif rel in self._subj_types:
                            role = u'S'
                        elif rel in self._obj_types:
                            role = u'O'
                        else:
                            role = u'X'
                        
                        if ent_roles.get(dlem, 0) < self._role2level[role]:
                            ent_roles[dlem] = self._role2level[role]
                        for nn in self._noun_mods(s, dep):
                            nnl = nn.lem.lower()
                            if ent_roles.get(nnl, 0) < self._role2level[role]:
                                ent_roles[nnl] = self._role2level[role]
                         
        return ent_roles

    def _noun_mods(self, s, t):
        nmods = set()
        if t not in s.gov2deps:
            return nmods
        for rel, dep in s.gov2deps[t]:
            if dep.is_noun() and rel in ['nn', 'appos', 'poss']:
                nmods.add(dep)
                for nm in self._noun_mods(s, dep):
                    nmods.add(nm)
        return nmods

    def _copula_roles(self, s, to_be):
        ent_roles = set()
        cop_obj = None
        for rel, t in s.dep2govs[to_be]:
            if rel == 'cop' and t.is_noun():
                cop_obj = t
                ent_roles.add(tuple([cop_obj, 'X']))
                for nmod in self._noun_mods(s, cop_obj):
                    ent_roles.add(tuple([nmod, 'X']))
                break
        if cop_obj is not None:
            for rel, dep in s.gov2deps[cop_obj]:
                if rel == 'nsubj' and dep.is_noun():
                    ent_roles.add(tuple([dep, 'X']))
                    for nmod in self._noun_mods(s, dep):
                        ent_roles.add(tuple([nmod, 'X']))
        return ent_roles

class EntityGridVectorizer(object):
    def __init__(self,
                 unigrams=True, bigrams=True, trigrams=True,
                 salience_threshold=1, use_syntax=True):

        self.unigrams = unigrams
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.salience_threshold = salience_threshold
        self.use_syntax = use_syntax


        self._ready()


    def _ready(self):

        # Initialize feature indices depending on what features are active.
        self._feature_indices = {}
        index = 0        
        if self.unigrams is True:
            self._uni_start = index
            for unigram in _unigrams:
                if self.use_syntax is False:
                    if u'S' in unigram or 'O' in unigram:
                        continue
                self._feature_indices[unigram] = index
                index += 1 
            self._uni_stop = index

        if self.bigrams is True:
            self._bi_start = index
            for bigram in _bigrams:
                if self.use_syntax is False:
                    if u'S' in bigram or 'O' in bigram:
                        continue
                self._feature_indices[bigram] = index
                index += 1    
            self._bi_stop = index
        if self.trigrams is True:
            self._tri_start = index
            for trigram in _trigrams:
                if self.use_syntax is False:
                    if u'S' in trigram or 'O' in trigram:
                        continue
                self._feature_indices[trigram] = index
                index += 1    
            self._tri_stop = index

        if self.salience_threshold is None or self.salience_threshold < 1:
            self._salience_index_offset = 0
            self._n_dims = index
        else:
            self._salience_index_offset = index      
            self._n_dims = 2 * index        
    
    def transform(self, E):
        if isinstance(E, EntityGrid):
            x = np.zeros((1, self._n_dims), dtype=np.float64)
            self._get_unnormalized_row(x[0, :], E)
            self._normalize(x)
            return x
        else:
            n_grids = len(E)
            X = np.zeros((n_grids, self._n_dims), dtype=np.float64)
            for row, e in enumerate(E):
                self._get_unnormalized_row(X[row, :], e)
            self._normalize(X)
            return X        

    def pairwise_transform(self, E_P):

        n_pairs = np.sum([len(e_Pe["perms"]) for e_Pe in E_P])
        n_dims = self._n_dims

        X_gold = np.zeros((n_pairs, n_dims), dtype=np.float64)
        X_perm = np.zeros((n_pairs, n_dims), dtype=np.float64)

        row = 0
        for e_Ep in E_P: 
            Ep = e_Ep[u"perms"]
            n_perms = len(Ep)
            self._get_unnormalized_row(X_gold[row:row+n_perms, :], 
                                       e_Ep[u"gold"])
            for p, e_perm in enumerate(Ep):
                self._get_unnormalized_row(X_perm[row + p, :], e_perm)
            row += n_perms
            
        self._normalize(X_gold)
        self._normalize(X_perm)
        y = np.ones(n_pairs, dtype=np.int32) 
        X = X_gold - X_perm           
        for i in xrange(0, n_pairs, 2):
            y[i] = 0
            X[i] *= -1

        return X, y
#    def transform(self, X, y=None):
#        n_docs = len(X)
#        n_dims = 0
#        if self.unigrams is True:
#            n_dims += len(_unigrams)
#        if self.bigrams is True:
#            n_dims += len(_bigrams)
#        if self.trigrams is True:
#            n_dims += len(_trigrams)
#        if self._salience_index_offset > 0:
#            n_dims *= 2
#
#        vecs = np.zeros((n_docs, self.n_dims), dtype=np.float64)
#        for row, x in enumerate(X):
#            if self.input == u'filename':
#                doc = corenlp.read_xml(x)
#                grid = from_document(doc)
#            elif self.input == u'document':
#                grid = from_document(x)
#            elif self.input == u'entitygrid':
#                grid = x
#            else:
#                import sys
#                print "BAD TYPE!"
#                sys.exit()
#            
    def _normalize(self, X):            

        offset = self._salience_index_offset
        if self.unigrams:
            start = self._uni_start
            stop = self._uni_stop
            totals = np.sum(X[:,start:stop], axis=1, keepdims=True)
            X[:,start:stop] /= totals

            if offset > 0:            
                sal_start = start + offset
                sal_stop = stop + offset
                totals = np.sum(X[:,sal_start:sal_stop], 
                                axis=1, keepdims=True)
                X[:,sal_start:sal_stop] /= totals

        if self.bigrams:
            start = self._bi_start
            stop = self._bi_stop
            totals = np.sum(X[:,start:stop], axis=1, keepdims=True)
            X[:,start:stop] /= totals

            if offset > 0:
                sal_start = start + offset
                sal_stop = stop + offset
                totals = np.sum(X[:,sal_start:sal_stop],
                                axis=1, keepdims=True)
                X[:,sal_start:sal_stop] /= totals

        if self.trigrams:
            start = self._tri_start
            stop = self._tri_stop
            totals = np.sum(X[:,start:stop], axis=1, keepdims=True)
            X[:,start:stop] /= totals

            if offset > 0:
                sal_start = start + offset
                sal_stop = stop + offset
                totals = np.sum(X[:,sal_start:sal_stop],
                                axis=1, keepdims=True)
                X[:,sal_start:sal_stop] /= totals
        return X   

    def _get_unnormalized_row(self, row_vec, grid):
        if len(row_vec.shape) > 2:
            raise NotImplementedError()
        if len(row_vec.shape) == 1:
            row_vec = row_vec.reshape((1, row_vec.shape[0]))
        for ent in grid.entities:
            roles = grid.ent2roles[ent]
            if self.use_syntax is False:
                roles = [u'X' if role != u'-' else u'-'
                         for role in roles]
            ne = [role for role in roles if role != u'-']                
            roles = [u'0'] + roles + [u'1']
            n_roles = len(roles)

            salient = True if len(ne) > self.salience_threshold else False
            offset = 0 if not salient else self._salience_index_offset
            
            for i in range(n_roles):
                if self.unigrams and roles[i] != u'0' and roles[i] != u'1':
                    unigram = roles[i]                            
                    row_vec[:,self._feature_indices[unigram] + offset] += 1    
                if self.bigrams is True and i + 1 < n_roles:
                    bigram = u'{}{}'.format(roles[i], roles[i + 1])
                    row_vec[:,self._feature_indices[bigram] + offset] += 1  
                if self.trigrams is True and i + 2 < n_roles:
                    trigram = u'{}{}{}'.format(
                        roles[i], roles[i + 1], roles[i + 2])
                    row_vec[:,self._feature_indices[trigram] + offset] += 1

