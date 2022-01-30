import logging
import os
import pickle
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import gensim
import math
import csv
from tree_Operate import *

logger = logging.getLogger(__name__)


#词对象
class wordObj(object):
    def __init__(self,sen_id=-1,word_id_doc=-1,word_id_sen=-1,word='',pos='',parent_sen=-1,dep='',isPred='',argus=[]):
        self.sen_id = sen_id
        self.word_id_doc = word_id_doc
        self.word_id_sen = word_id_sen
        self.word = word
        self.pos = pos
        self.parent_sen = parent_sen
        self.parent = -1
        self.dep = dep
        self.isPred = isPred
        self.argus = argus
        self.label = 0

#event object
class eventObj(object):
    def __init__(self,id=-1):
        self.id = id
        self.sen_id = -1
        self.level = -1
        self.tokens = []
        self.tokens_ids = []
        self.noun_token = []
        self.verb_token = []
        self.dep_pred = -1
        self.token_level = [math.exp(-10),math.exp(-10),math.exp(-10)]
        self.deps = ['','','']
        self.pos = ['','','']
        self.is_sen_root = False
        self.tokens_index = [0,0,0]
        self.triple_tokens = ['','','']
        self.triple_token_ids = [0,0,0]

def get_dataset(args):
    train_name = './data/CoNLL2009-ST-Chinese-train-topic-event.csv'
    test_name = './data/CoNLL2009-ST-evaluation-Chinese-topic-event.csv'

    train_docs_file = os.path.join(args.output_dir, 'wordobj_cache/train_docs.pkl')
    test_docs_file = os.path.join(args.output_dir, 'wordobj_cache/test_docs.pkl')

    train = read_sentence(train_name)
    # logger.info('# Read %s Train set: document: %d, sentence: %d', dataset_name, len(train),train_num_sentence)
    logger.info('store train wordobjs')
    with open(train_docs_file, 'wb') as wf:
        pickle.dump(train, wf,-1)

    test = read_sentence(test_name)
    # logger.info("# Read %s Test set: document: %d, sentence: %d", dataset_name, len(test),test_num_sentence)
    logger.info('store test wordobjs')
    with open(test_docs_file, 'wb') as wf:
        pickle.dump(test, wf,-1)

    return train, test

def read_sentence(file_path):
    docs_tree = []
    sentences = []
    sentence = []
    words = ''
    word_id_doc = 0
    sen_id = 1
    pred_id_list = []
    with open(file_path, 'r',encoding='utf-8-sig') as f:
        lines = csv.reader(f)
        for line in lines:
            if (''.join(line).strip() == ''): # sentence end
                sen_id += 1
                if words == "（完）" or words == "完": # doc end
                    word_id_doc = 0
                    sen_id = 0
                    doc_tree = Doc_Tree()
                    doc_tree.build_tree(sentences)
                    doc_tree.remove_stop_word_nodes_tree()
                    docs_tree.append(doc_tree)
                    sentences = []
                    sentence = []
                    pred_id_list = []
                    words = ''
                    continue
                sentence = update_wordobj(sentence)
                sentences.append(sentence)
                sentence = []
                pred_id_list = []
                words = ''
                continue
            words += line[2]
            word_id_doc += 1
            wordobj = wordObj(sen_id,word_id_doc,int(line[0]),line[2],line[5],int(line[9]),line[11],line[13],line[15:])
            if line[13] == 'Y': # is an event
                pred_id_list.append(word_id_doc)
                if int(line[1]) == 1: # is topic event
                    wordobj.label = 1
            sentence.append(wordobj)

    return docs_tree

# update parent id from sen id to doc id
def update_wordobj(sentence):
    for wordobj in sentence:
        parent_id_sen = wordobj.parent_sen
        for object in sentence:
            if object.word_id_sen == parent_id_sen:
                wordobj.parent = object.word_id_doc
    return sentence


def load_datasets_and_vocabs(args):
    train_docs_file = os.path.join(args.output_dir, 'wordobj_cache/train_docs.pkl')
    test_docs_file = os.path.join(args.output_dir, 'wordobj_cache/test_docs.pkl')

    train_example_file = os.path.join(args.output_dir, 'example_cache/train_example_catch.pkl')
    test_example_file = os.path.join(args.output_dir, 'example_cache/test_example_catch.pkl')
    train_weight_file = os.path.join(args.output_dir, 'example_cache/train_weight_catch.pkl')
    test_weight_file = os.path.join(args.output_dir, 'example_cache/test_weight_catch.pkl')

    if os.path.exists(train_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_example = pickle.load(f)

        logger.info('Loading train_weight from %s', train_weight_file)
        with open(train_weight_file, 'rb') as f:
            train_labels_weight = pickle.load(f)

        logger.info('Loading train_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_example = pickle.load(f)

        logger.info('Loading test_weight from %s', train_weight_file)
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = pickle.load(f)
    else:
        if os.path.exists(train_docs_file) and os.path.exists(test_docs_file):
            with open(train_docs_file, 'rb') as rf:
                train = pickle.load(rf)
            with open(test_docs_file, 'rb') as rf:
                test = pickle.load(rf)
        else:
            train, test = get_dataset(args)

        # get examples of data
        train_example, train_labels_weight = create_example(train)
        test_example, test_labels_weight = create_example(test)

        logger.info('Creating train_example_cache')
        with open(train_example_file,'wb') as f:
            pickle.dump(train_example,f,-1)
        logger.info('Creating train_weight_cache')
        with open(train_weight_file,'wb') as wf:
            pickle.dump(train_labels_weight,wf,-1)

        logger.info('Creating test_example_cache')
        with open(test_example_file,'wb') as f:
            pickle.dump(test_example,f,-1)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file,'wb') as wf:
            pickle.dump(test_labels_weight,wf,-1)

    logger.info('Train set size: %s', len(train_example))
    logger.info('Test set size: %s,', len(test_example))

    # Build word vocabulary(dep_tag, part of speech) and save pickles.
    word_vecs,word_vocab,dep_tag_vocab, pos_tag_vocab, sen_id_tag_vocab = load_and_cache_vocabs(train_example+test_example, args)

    if args.embedding_type == 'word2vec':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.word2vec_embedding = embedding

    train_dataset = TEE_Dataset(
        train_example, args,word_vocab,dep_tag_vocab,pos_tag_vocab,sen_id_tag_vocab)
    test_dataset = TEE_Dataset(
        test_example, args,word_vocab,dep_tag_vocab,pos_tag_vocab,sen_id_tag_vocab)

    return train_dataset,torch.tensor(train_labels_weight), test_dataset,torch.tensor(test_labels_weight),word_vocab, dep_tag_vocab, pos_tag_vocab, sen_id_tag_vocab

def create_example(docs):
    examples = []
    all_labels = []
    for doc_tree in docs:
        example = {'t_ids':[],'token_level':[],'token_adj':[],'e_ids': [], 'events': [],'event_deps':[],'event_pos':[],'event_sen_ids':[],'event_level':[],
                   'share_e_adj':[],'org_e_adj':[],'labels': []}

        event_id = 1
        token_id = 1
        token_edges = []
        event_list = []
        child_nodes = doc_tree.dp_tree.all_nodes()
        child_nodes.sort(key=doc_tree.node_sort)

        for child_node in child_nodes:
            if child_node.identifier == DROOT:
                continue
            parent_node = doc_tree.dp_tree.parent(child_node.identifier)

            # create event
            if child_node.data.isPred == 'Y':
                eventobj = get_event(doc_tree,child_node)

                eventobj.id = event_id
                eventobj.sen_id = child_node.data.sen_id
                eventobj.level = math.exp(-child_node.data.level)

                parent_pred_id = get_dep_event(doc_tree,child_node)
                if parent_pred_id != -1:
                    doc_tree.dp_tree.nodes[child_node.identifier].data.dep_pred = parent_pred_id
                    eventobj.dep_pred = parent_pred_id

                # for organization edges
                if parent_node.identifier == DROOT:
                    eventobj.is_sen_root = True

                # add token information
                token_edges.append([token_id, token_id])
                token_edges.append([token_id + 1, token_id + 1])
                token_edges.append([token_id + 2, token_id + 2])
                example['t_ids'].append(token_id)
                example['t_ids'].append(token_id+1)
                example['t_ids'].append(token_id+2)
                sub = eventobj.triple_tokens[0]
                obj = eventobj.triple_tokens[2]
                if not (sub == '_'):
                    token_edges.append([token_id,token_id+1])
                    token_edges.append([token_id+1,token_id])
                if not (obj == '_'):
                    token_edges.append([token_id+2, token_id + 1])
                    token_edges.append([token_id + 1, token_id+2])
                token_id += 3
                example['token_level'].append(eventobj.token_level)

                # add event information
                example['e_ids'].append(eventobj.id)
                example['events'].append(eventobj.triple_tokens)
                example['event_deps'].append(eventobj.deps)
                example['event_pos'].append(eventobj.pos)
                example['event_sen_ids'].append(eventobj.sen_id)
                example['event_level'].append(eventobj.level)
                example['labels'].append(child_node.data.label)

                event_list.append(eventobj)
                event_id += 1

        token_edges = remove_repetion(token_edges)
        token_adj = build_adj(token_edges,example,'token')

        dep_e_edges,share_e_edges,org_e_edges = build_e_edges(event_list)
        dep_e_adj = build_adj(dep_e_edges,example,'event')
        share_e_adj = build_adj(share_e_edges,example,'event')
        org_e_adj = build_adj(org_e_edges,example,'event')

        example['token_adj'] = token_adj.numpy().tolist()
        example['dep_e_adj'] = dep_e_adj.numpy().tolist()
        example['share_e_adj'] = share_e_adj.numpy().tolist()
        example['org_e_adj'] = org_e_adj.numpy().tolist()
        examples.append(example)

        all_labels += example['labels']

    weights = get_labels_weight(all_labels)

    return examples, weights

def build_token_edges(doc_tree,cur_node):
    cnodes = []
    token_edges = []
    doc_tree.get_all_node(cur_node,cnodes)
    for cnode in cnodes:
        token_edges.append([cnode.identifier,cur_node.identifier])
        token_edges.append([cur_node.identifier,cnode.identifier])
    return token_edges

def get_event(doc_tree,cur_node):
    eventobj = eventObj()

    child_nodes = []
    doc_tree.get_all_node(cur_node,child_nodes)
    if cur_node.data.pos == 'VV' or cur_node.data.pos == 'VE':
        eventobj.verb_token.append(cur_node.tag)
    child_nodes.sort(key=doc_tree.node_sort)
    for child_node in child_nodes:
        # get event information
        eventobj.tokens_ids.append(child_node.identifier)
        eventobj.tokens.append(child_node)
        if (child_node.data.pos == 'NN' or child_node.data.pos == 'NR') and child_node.data.dep == 'NMOD':
            eventobj.noun_token.append(child_node.tag)

    # get triple element of event (sub, pred, obj)
    dir_child_nodes = doc_tree.dp_tree.children(cur_node.identifier)
    has_sub = False
    has_obj = False
    has_comp = False
    # get predicate information
    eventobj.token_level[1] = math.exp(-cur_node.data.level)
    eventobj.triple_tokens[1] = cur_node.tag
    eventobj.triple_token_ids[1] = cur_node.identifier
    eventobj.deps[1] = cur_node.data.dep
    eventobj.pos[1] = cur_node.data.pos

    for child_node in dir_child_nodes:
        if child_node.data.dep == 'SBJ' and not has_sub:
            eventobj.token_level[0] = math.exp(-child_node.data.level)
            eventobj.triple_tokens[0] = child_node.tag
            eventobj.triple_token_ids[0] = child_node.identifier
            eventobj.deps[0] = child_node.data.dep
            eventobj.pos[0] = child_node.data.pos
            has_sub = True
        if child_node.data.dep == 'OBJ' and not has_obj:
            eventobj.token_level[2] = math.exp(-child_node.data.level)
            eventobj.triple_tokens[2] = child_node.tag
            eventobj.triple_token_ids[2] = child_node.identifier
            eventobj.deps[2] = child_node.data.dep
            eventobj.pos[2] = child_node.data.pos
            has_obj = True
        # take obj preferentially
        if child_node.data.dep == 'COMP' and not has_comp and not has_obj:
            eventobj.token_level[2] = math.exp(-child_node.data.level)
            eventobj.triple_tokens[2] = child_node.tag
            eventobj.triple_token_ids[2] = child_node.identifier
            eventobj.deps[2] = child_node.data.dep
            eventobj.pos[2] = child_node.data.pos
            has_comp = True

    return eventobj

def get_dep_event(doc_tree,cur_node):
    parent_node = doc_tree.dp_tree.parent(cur_node.identifier)
    if parent_node.identifier == DROOT:
        return -1
    if parent_node.data.isPred == 'Y':
        return parent_node.identifier
    return get_dep_event(doc_tree,parent_node)


def build_e_edges(eventobjs):
    dep_e_edges = []
    share_e_edges = []
    org_e_edges = []
    for i,eventobj in enumerate(eventobjs):
        # self join
        share_e_edges.append([eventobj.id,eventobj.id])
        org_e_edges.append([eventobj.id,eventobj.id])

        for j, obj in enumerate(eventobjs):
            # dep edges
            if eventobj.dep_pred == obj.triple_token_ids[1]:
                dep_e_edges.append([eventobj.id,obj.id])
        
            # share edges
            if j>i and set(eventobj.noun_token) & set(obj.noun_token):
                share_e_edges.append([eventobj.id,obj.id])
                share_e_edges.append([obj.id,eventobj.id])
            if j > i and set(eventobj.verb_token) & set(obj.verb_token):
                share_e_edges.append([eventobj.id,obj.id])
                share_e_edges.append([obj.id,eventobj.id])
            
            # org edges
            if eventobj.is_sen_root and obj.is_sen_root and ((eventobj.sen_id - obj.sen_id) == 1):
                org_e_edges.append([eventobj.id,obj.id])
            if eventobj.is_sen_root and (not obj.is_sen_root) and obj.dep_pred == -1 and ((eventobj.sen_id - obj.sen_id) == 1):
                org_e_edges.append([eventobj.id,obj.id])
            if (not eventobj.is_sen_root) and eventobj.dep_pred == -1 and obj.is_sen_root and ((eventobj.sen_id - obj.sen_id) == 1):
                org_e_edges.append([eventobj.id,obj.id])

    dep_e_edges = remove_repetion(dep_e_edges)
    share_e_edges = remove_repetion(share_e_edges)
    org_e_edges = remove_repetion(org_e_edges)

    return dep_e_edges,share_e_edges,org_e_edges

def remove_repetion(llist):
    new_list = []
    for li in llist:
        if li not in new_list:
            new_list.append(li)
    return new_list

def build_adj(sour_edges,data,data_type):
    if data_type == 'token':
        ids = np.array(data['t_ids'], dtype=np.int32)
        matrix_shape = np.array(data['t_ids']).shape[0]
    else:
        ids = np.array(data['e_ids'], dtype=np.int32)
        matrix_shape = np.array(data['labels']).shape[0]

    idx_map = {j: i for i, j in enumerate(ids)}
    edges = []
    for i,edge in enumerate(sour_edges):
        edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    edges = np.array(edges, dtype=np.int32).reshape(np.array(sour_edges).shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(matrix_shape, matrix_shape),dtype=np.float32)

    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj

def get_labels_weight(labels):
    label_ids = labels
    nums_labels = Counter(labels)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weights = []
    label_value = [0,1]
    for value_id in label_value:
        if value_id not in label_ids:
            weights.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weights.append(median/label[1])
                    break

    return weights

def load_and_cache_vocabs(examples, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'embedding_cache')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and word2vec embeddings.
    if args.embedding_type == 'word2vec':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',args.dataset_name)
            word_vocab = build_text_vocab(examples)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', cached_word_vocab_file)
            word_vecs = load_word2vec_embedding(word_vocab['itos'], args,0.25)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(examples, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of pos tags from %s',cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of pos tags.')
        pos_tag_vocab = build_pos_tag_vocab(examples, min_freq=0)
        logger.info('Saving pos tags  vocab, size: %s, to file %s',pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    # Build vocab of sen id tags.
    cached_sen_id_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_sen_id_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_sen_id_tag_vocab_file):
        logger.info('Loading vocab of sen id tags from %s', cached_sen_id_tag_vocab_file)
        with open(cached_sen_id_tag_vocab_file, 'rb') as f:
            sen_id_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of sen id tags.')
        sen_id_tag_vocab = build_sen_id_tag_vocab(examples, min_freq=0)
        logger.info('Saving sen id tags  vocab, size: %s, to file %s', sen_id_tag_vocab['len'], cached_sen_id_tag_vocab_file)
        with open(cached_sen_id_tag_vocab_file, 'wb') as f:
            pickle.dump(sen_id_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab, sen_id_tag_vocab


def load_word2vec_embedding(words,args,uniform_scale):

    path = os.path.join(args.embedding_dir,'baike_26g_news_13g_novel_229g.model')
    w2v_model = gensim.models.Word2Vec.load(path)

    w2v_vocabs = [word for word, Vocab in w2v_model.wv.vocab.items()]

    word_vectors = []
    for word in words:
        if word in w2v_vocabs:
            word_vectors.append(w2v_model.wv[word])
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, w2v_model.vector_size))

    return word_vectors


def _default_unk_index():
    return 1

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        tokens_event = example['events']
        for tokens in tokens_event:
            counter.update(tokens)

    itos = []
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_dep_tag_vocab(examples, vocab_size=1000, min_freq=0):
    counter = Counter()
    for example in examples:
        event_deps = example['event_deps']
        for dep in event_deps:
            counter.update(dep)

    itos = []
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_pos_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for example in examples:
        event_pos = example['event_pos']
        for pos in event_pos:
            counter.update(pos)

    itos = []
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_sen_id_tag_vocab(examples, vocab_size=1000, min_freq=0):
    """
    sentence id tags vocab.
    """
    counter = Counter()
    for example in examples:
        sen_id = example['event_sen_ids']
        counter.update(sen_id)

    itos = []
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class TEE_Dataset(Dataset):
    def __init__(self, examples, args, word_vocab,dep_tag_vocab, pos_tag_vocab,sen_id_tag_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.sen_id_tag_vocab = sen_id_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['token_level'],e['token_adj'],e['event_ids'],e['event_dep_ids'],e['event_pos_ids'],e['event_sen_ids'],\
                e['event_level'],e['dep_e_adj'],e['share_e_adj'],e['org_e_adj'],e['labels']

        items_tensor = tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''

        for i in range(len(self.examples)):
            self.examples[i]['event_ids'] = []
            self.examples[i]['event_dep_ids'] = []
            self.examples[i]['event_pos_ids'] = []

            # event information
            for event in self.examples[i]['events']:
                self.examples[i]['event_ids'].append([self.word_vocab['stoi'][w] for w in event])
            for event in self.examples[i]['event_deps']:
                self.examples[i]['event_dep_ids'].append([self.dep_tag_vocab['stoi'][dep] for dep in event])
            for event in self.examples[i]['event_pos']:
                self.examples[i]['event_pos_ids'].append([self.pos_tag_vocab['stoi'][pos] for pos in event])
            self.examples[i]['event_len'] = len(self.examples[i]['events'])
            self.examples[i]['event_sen_ids'] = [self.sen_id_tag_vocab['stoi'][sen_id] for sen_id in self.examples[i]['event_sen_ids']]


def my_collate(batch):
    '''
    Pad event in a batch.
    Sort the events based on length.
    Turn all into tensors.
    '''
    # from Dataset.__getitem__()
    token_level,token_adj,event_ids,event_dep_ids,event_pos_ids,event_sen_ids,event_level,dep_e_adj,share_e_adj,org_e_adj,labels = zip(*batch)

    # batch size is 1,remove the dim = 0
    # convert to tensor
    token_level = token_level[0].clone().detach()
    token_adj = token_adj[0].clone().detach()
    event_ids = event_ids[0].clone().detach()
    event_dep_ids = event_dep_ids[0].clone().detach()
    event_pos_ids = event_pos_ids[0].clone().detach()
    event_sen_ids = event_sen_ids[0].clone().detach()
    event_level = event_level[0].clone().detach()
    dep_e_adj = dep_e_adj[0].clone().detach()
    share_e_adj = share_e_adj[0].clone().detach()
    org_str_e_adj = org_e_adj[0].clone().detach()
    labels = labels[0].clone().detach()


    return token_level,token_adj,event_ids,event_dep_ids,event_pos_ids,event_sen_ids,event_level,dep_e_adj,share_e_adj,org_str_e_adj,labels
