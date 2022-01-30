from treelib import Tree

Symbol_list = [',','，']
DROOT = 0

class Tree_Node(object):
    def __init__(self,sen_id=-1,word_id_sen=-1,dep='',pos='',isPred='',sen_root=-1,level=-1,label = 0):
        self.sen_id = sen_id
        self.word_id_sen = word_id_sen
        self.dep = dep
        self.pos = pos
        self.isPred = isPred
        self.sen_root = sen_root
        self.level = level
        self.dep_pred = -1
        self.label = label

class Doc_Tree(object):
    def __init__(self):
        self.dp_tree = Tree()
        self.dp_tree.create_node('DROOT', DROOT, data=Tree_Node(level=-1))
    def build_tree(self,sentences):
        for sentence in sentences:
            for wordobj in sentence:
                # 是否为句子根结点
                if wordobj.dep == 'ROOT':
                    self.dp_tree.create_node(wordobj.word, wordobj.word_id_doc, parent=DROOT,
                                                 data=Tree_Node(wordobj.sen_id, wordobj.word_id_sen,wordobj.dep,wordobj.pos,
                                                                wordobj.isPred,0,0,wordobj.label))
                    self.build_dp_tree(sentence,wordobj.word_id_doc,wordobj.word_id_doc)
                    break

    def node_sort(self,node):
        return node.identifier

    def complete_ee(self,par_node):
        child_nodes = self.dp_tree.children(par_node.identifier)
        left_info = ''
        right_info = ''
        if not child_nodes:
            return par_node.tag

        child_nodes.sort(key= self.node_sort)

        for child_node in child_nodes:
            if child_node.identifier < par_node.identifier:
                left_info += self.complete_ee(child_node)
            elif child_node.identifier > par_node.identifier:
                right_info += self.complete_ee(child_node)

        return left_info + par_node.tag + right_info

    def get_all_node(self,cur_node,cnode_list):
        if not cur_node:
            return cnode_list
        child_nodes = self.dp_tree.children(cur_node.identifier)
        for child_node in child_nodes:
            self.get_all_node(child_node,cnode_list)
            cnode_list.append(child_node)

    def build_dp_tree(self,sentence,pnode_id,root_id):
        pnode = self.dp_tree.get_node(pnode_id)
        for wordobj in sentence:
            if wordobj.parent_sen == pnode.data.word_id_sen:
                self.dp_tree.create_node(wordobj.word,wordobj.word_id_doc,parent=pnode.identifier,data=Tree_Node(wordobj.sen_id,wordobj.word_id_sen,wordobj.dep,wordobj.pos,wordobj.isPred,root_id,pnode.data.level+1,wordobj.label))
                self.build_dp_tree(sentence,wordobj.word_id_doc,root_id)

    def remove_stop_word_nodes_tree(self):
        nodes = self.dp_tree.all_nodes()
        for node in nodes:
            if node.data.pos == 'PU' and node.tag in Symbol_list:
                # self.update_node_id(node, node.data.sen_root)

                cnodes = []
                self.get_all_node(node,cnodes)
                for cnode in cnodes:
                    self.dp_tree.nodes[cnode.identifier].data.level = cnode.data.level + 1
                self.dp_tree.link_past_node(node.identifier)
