from multilayer_net import MultiLayerNetwork
from multi_timer import MultiTimer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import music21 as ms
import tqdm

class PrefixTree:
    def __init__(self, input_file=None, depth=0, parent=None, node=None, last_visit=-1000):
        self.weight = 0
        self.effective_weight = 0
        self.last_visit = last_visit
        self.depth = depth
        self.parent = parent
        self.node = node
        self.children = {}
        self.transition_weight = {}
        if input_file is not None:
            with open(input_file, 'r') as openfile:
                data = json.load(openfile)
                self.load(data)
    
    def is_root(self):
        return self.parent is None
    
    def reset_last_visit(self, value=-1000):
        self.last_visit = value
        for child in self.children.values():
            child.reset_last_visit(value=value)

    def load(self, data):
        self.weight = data["weight"]
        self.transition_weight = data["transition_weight"]
        for node, child in data["children"].items():
            tree = PrefixTree(depth=self.depth+1, parent=self, node=node)
            tree.load(child)
            self.children[node] = tree

    def to_dict(self):
        data = {}
        data["weight"] = self.weight
        data["transition_weight"] = self.transition_weight
        data["children"] = {}
        for node, child in self.children.items():
            data["children"][node] = child.to_dict()
        return data
    
    def save(self, output_file):
        with open(output_file, "w") as outfile:
            json.dump(self.to_dict(), outfile)
    
    def next_proba(self, node):
        return self.transition_weight[node]/self.weight if node in self.transition_weight else 0

    def entropy(self):
        return -sum([weight*np.log2(weight) for weight in self.transition_weight.values()])/self.weight + np.log2(self.weight)

    def relative_entropy(self, child):
        return sum([child.next_proba(node)*np.log2(child.next_proba(node)/self.next_proba(node)) for node in child.transition_weight.keys()])
    
    def get_tree(self, lst_nodes, copy_node_lst=True):
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if lst_nodes:
            node = lst_nodes.pop()
            if node in self.children:
                return self.children[node].get_tree(lst_nodes, copy_node_lst=False)
        return self
    
    def proba(self, lst_nodes, node, copy_node_lst=True):
        self.get_tree(lst_nodes, copy_node_lst).next_proba(node)

    def pattern_depth(self, lst_nodes, copy_node_lst=True):
        return self.get_tree(lst_nodes, copy_node_lst).depth
    
    def add_all_prefix_children(self, lst_nodes, next_node, idx, copy_node_lst=True):
        self.weight += 1
        if idx >= self.last_visit + self.depth:
            self.last_visit = idx
            self.effective_weight += 1
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if next_node in self.transition_weight:
            self.transition_weight[next_node] += 1
        else :
            self.transition_weight[next_node] = 1
        if lst_nodes:
            node = lst_nodes.pop()
            if node not in self.children:
                self.children[node] = PrefixTree(depth=self.depth+1, parent=self, node=node)
            self.children[node].add_all_prefix_children(lst_nodes, next_node, idx, copy_node_lst=False)

    def weight_pruning(self, t):
        nodes_to_remove = []
        for node in self.children.keys():
            if self.children[node].weight < t:
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.children.pop(node)
        for child in self.children.values():
            child.weight_pruning(t)
    
    def similarity_pruning(self, K=0.04):
        nodes_to_remove = []
        for child in self.children.values():
            child.similarity_pruning(K)
        for node, child in self.children.items():
            if child.children: continue
            info_gain = self.relative_entropy(child)*child.weight
            if info_gain < K:
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.children.pop(node)

    def stringify(self):
        output = "root:" + str(self.weight) + "\n" if self.depth==0 else ""
        for node, weight in sorted(self.transition_weight.items(), key=lambda x:-x[1]):
            output += "|  "*self.depth + "[" + str(round(weight/self.weight,3)) + "]:" + node + "\n" 
        for node,child in sorted(self.children.items(), key=lambda x:-x[1].weight):
            output += "|  "*self.depth + "├──" + str(child.weight) + " " + node + "\n"
            output += child.stringify()
        return output

    def __str__(self):
        return self.stringify()
    
    def nb_nodes(self):
        return 1 + sum([child.nb_nodes() for child in self.children.values()])
    
    def max_depth(self):
        if not self.children: 
            return 0
        else:
            return 1 + max([child.max_depth() for child in self.children.values()])
    
    def random_next_node(self, lst_nodes, alpha=1.,copy_node_lst=True):
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if lst_nodes:
            node = lst_nodes.pop()
            if node in self.children:
                return self.children[node].random_next_node(lst_nodes, copy_node_lst=False)
        return random.choices(list(self.transition_weight.keys()), [elt**alpha for elt in self.transition_weight.values()])[0]
    
    def generate_sequence(self, size, alpha=1.):
        # Can be optimized to copy only the last "max_order" elements of the list 
        nodes = []
        for i in range(size):
            nodes.append(self.random_next_node(nodes, alpha))
        return nodes

    def add_nodes(self, nodes_lst, max_order, edge_mode = False):
        indices = range(1,len(nodes_lst)+1,2) if edge_mode else range(len(nodes_lst)+1)
        if edge_mode:
            max_order = 2*max_order - 1
        for i in indices:
            self.add_all_prefix_children(nodes_lst[max(0,i-max_order):i], 
                nodes_lst[i] if i<len(nodes_lst) else "end", i, False)
    
    def compact(self):
        nodes_to_compact = []
        for node, child in self.children.items():
            child.compact()
            if len(child.children) == 1:
                nodes_to_compact.append(node)
        for node in nodes_to_compact:
            child = self.children.pop(node)
            for key, value in child.children.items():
                subnode, subchild = key, value
            new_node = node + "," + subnode
            self.children[new_node] = subchild

    def compute_scores(self, entry_cost=1, word_length_cost=1, sequence_length_cost=1, edge_mode=False):
        for child in self.children.values():
            child.compute_scores(entry_cost, word_length_cost, sequence_length_cost, edge_mode)
        if edge_mode and self.depth%2==0:
            self.score = -1000
        else:
            self.score = (self.depth-1)*self.effective_weight*sequence_length_cost - entry_cost - self.depth*word_length_cost

    def get_best_tree(self):
        best_score = self.score
        best_tree = self
        for child in self.children.values():
            tree, score = child.get_best_tree()
            if score > best_score:
                best_score = score
                best_tree = tree
        return best_tree, best_score

    def get_tree_list(self):
        if self.children:
            return sum([child.get_tree_list() for child in self.children.values()], start=[]) + [self]
        else:
            return [self]

    def rebuild_sequence(self, seq=None):
        if seq is None:
            seq = []
        if self.node is None:
            return seq
        else :
            seq.append(self.node)
            return self.parent.rebuild_sequence(seq)

class SequenceSimplifier:

    def __init__(self, sequence, entry_cost=1, word_length_cost=1, sequence_length_cost=1):
        self.entry_cost = entry_cost
        self.word_length_cost = word_length_cost
        self.sequence_length_cost = sequence_length_cost
        unique_words = set(sequence)
        self.base_alphabet_size = len(unique_words)
        self.id_to_word = {}
        word_to_id = {}
        for i,word in enumerate(unique_words):
            self.id_to_word[i] = word
            word_to_id[word] = i
        self.sequence = [word_to_id[word] for word in sequence]
    
    def rebuild_elt(self, id):
        word = self.id_to_word[id]
        if type(word) != tuple:
            return word
        else:
            return tuple([self.rebuild_elt(sub_word) for sub_word in word])
    
    def rebuild_sequence(self):
        return [self.rebuild_elt(id) for id in self.sequence]

    def create_new_word(self, new_word):
        new_id = len(self.id_to_word)
        pattern = list(new_word)
        words_to_modify = []
        for id,word in self.id_to_word.items():
            if type(word)==tuple and len(word) > len(new_word):
                replaced_word = tuple(replace(list(word), pattern, new_id))
                words_to_modify.append((id, replaced_word))
        for id,word in words_to_modify:
            self.id_to_word[id] = word

        self.id_to_word[new_id] = new_word
        self.sequence = replace(self.sequence, pattern, new_id)

    def alphabet_size(self):
        return sum([len(word) if type(word)==tuple else 1 for word in self.id_to_word.values()])
    
    def nb_entries_alphabet(self):
        return len(self.id_to_word)

    def sequence_len(self):
        return len(self.sequence)
    
    def total_score(self):
        return self.word_length_cost*self.nb_entries_alphabet() \
            + self.entry_cost*self.alphabet_size() \
            + self.sequence_length_cost * self.sequence_len()

    def best_sequence(self, t, L, base_alphabet_only=False, edge_mode=False):
        tree = PrefixTree()
        if base_alphabet_only:
            if edge_mode:
                raise Exception("Edge mode not implemented with base alphabet only")
            sub_seq = []
            for elt in self.sequence:
                if elt >= self.base_alphabet_size and sub_seq:
                    tree.reset_last_visit()
                    tree.add_nodes(sub_seq, L)
                    sub_seq = []
                else:
                    sub_seq.append(elt)
            if sub_seq:
                tree.reset_last_visit()
                tree.add_nodes(sub_seq, L)
        else :
            tree.add_nodes(self.sequence, L, edge_mode)
            for node in self.id_to_word.values():
                if type(node) != tuple: continue
                tree.reset_last_visit()
                tree.add_nodes(list(node), L, edge_mode)
        tree.weight_pruning(t=t)
        tree.compute_scores(self.entry_cost, self.word_length_cost, self.sequence_length_cost, edge_mode=edge_mode)
        best_tree, score = tree.get_best_tree()
        return tuple(best_tree.rebuild_sequence()), score

    def print_score(self):
        alphabet_size_score = self.alphabet_size()
        nb_entries = self.nb_entries_alphabet()
        length_score = self.sequence_len()
        print(f"score: {length_score}(length) + {nb_entries}(nb_entries) + {alphabet_size_score}(voc_size) = {nb_entries + alphabet_size_score + length_score}")


def extract_node_info(node):
    return json.loads(node)

def apply_diatonic_interval(pitch, interval, sharp_number=0):
    sharp_order = ["F","C","G","D","A","E","B"]
    basic_scale = ["C","D","E","F","G","A","B"]
    offset = basic_scale.index(pitch.name[0])
    idx = (interval+basic_scale)%len(basic_scale)
    return basic_scale[idx]

def replace(sequence, pattern, new_elt):
    '''
    Returns a list where each occurence of a pattern in the input list 
    is replaced by a single element representing this pattern
    '''
    new_sequence = []
    size = len(pattern)
    i=0
    while i < len(sequence)-size:
        if sequence[i:i+size] == pattern:
            new_sequence.append(new_elt)
            i += size
        else:
            new_sequence.append(sequence[i])
            i += 1
    new_sequence += sequence[i:]
    return new_sequence

def elt_to_str(elt):
    if type(elt) != tuple:
        return str(elt)
    return "(" + "".join([elt_to_str(sub_elt) for sub_elt in elt]) + ")"

def add_node(node, node_set):
    if node in node_set: return
    node_set.add(node)
    if type(node) != tuple: return
    for sub_node in node:
        add_node(node, node_set)

def get_alphabet(sequence):
    node_set = set()
    for node in sequence:
        add_node(node, node_set)
    return node_set

def sequence_score(sequence):
    node_set = get_alphabet(sequence)
    alphabet_score = sum([len(elt) if type(elt)==tuple else 1 for elt in node_set])
    length_score = len(sequence)
    total_score = alphabet_score + length_score
    print(f"score : {alphabet_score}(alphabet) + {length_score}(length) = {total_score}")
    return total_score

def is_int(value, margin=0.01):
    return abs(round(value)-value) <= margin

def duration_to_note(duration, rest=False):
    if duration == 0:
        assert(not rest)
        return chr(119188)
    if not is_int(np.log2(duration)):
        if not is_int(np.log2(duration/1.5)):
            return ("r" if rest else "") + str(duration)
        else:
            extension = "."
            value = round(2-np.log2(duration/1.5))
    else:
        value = round(2-np.log2(duration))
        extension = ""
    if rest:
        return chr(119099+value) + extension
    else:
        return chr(119133+value) + extension
    
def new_element_appearance(sequence):
    elt_set = set()
    nb_new_elts = []
    for elt in sequence:
        elt_set.add(elt)
        nb_new_elts.append(len(elt_set))
    return nb_new_elts

if __name__ == "__main__":

    t=5
    K=1
    L=50
    timer = MultiTimer()

    # directory = "datasets\\mozart_sonatas\\"
    # directory = "midis\\invent_bach\\"
    # midi_files = [directory + f for f in os.listdir(directory)]
    # midi_files = [str(elt) for elt in ms.corpus.getComposer('bach', fileExtensions=('xml',))]
    # midi_files = ["midis\\invent_bach\\invent14.mid"]
    midi_files = ["datasets\\mozart_sonatas\\K545-3.xml"]
    # midi_files = ["datasets\\mozart_sonatas\\K331-1.xml"]
    # midi_files = ["midis\\schubert_quartet.mid"]
    # output_file = "trees\\bach_invent1_duration_interval.json"
    timer.start("load_midi")
    net = MultiLayerNetwork(use_gui=False, enharmony=False, pitch=False, structure="multilayer", diatonic_interval=True, duration=True, rest=True, music_files=midi_files, offset=False, offset_period=1.)
    net.load_new_file(midi_files[0])
    timer.end("load_midi")
    nodes_lst = net.nodes_lists[0]
    nodes_lst = [json.loads(elt) for elt in nodes_lst]
    nodes_lst = [duration_to_note(elt["duration"],elt["rest"]) for elt in nodes_lst]
    edges_lst = net.edges_lists[0]
    edges_lst = [elt[22:-1] for elt in edges_lst]
    all_nodes = []
    for i in range(len(edges_lst)):
        all_nodes.append(nodes_lst[i])
        all_nodes.append(edges_lst[i])
    all_nodes.append(nodes_lst[-1])
    sequence_simplifier = SequenceSimplifier(all_nodes)
    tree = PrefixTree()
    sequence_simplifier.print_score()
    data = [sequence_simplifier.total_score()]
    while True:
        timer.start("computing best sequence")
        best_seq, score = sequence_simplifier.best_sequence(t, L, base_alphabet_only=False, edge_mode=True)
        timer.end("computing best sequence")
        if score <= 0: 
            break
        # print(f"expected score gain = {score}")
        timer.start("creating new word")
        sequence_simplifier.create_new_word(tuple(best_seq))
        timer.end("creating new word")
        print(f"simplifying {sequence_simplifier.rebuild_elt(sequence_simplifier.nb_entries_alphabet()-1)} with score {score}")
        sequence_simplifier.print_score()
        data.append(sequence_simplifier.total_score())
    # print("\n".join([f"{key}:{value}" for key,value in sequence_simplifier.id_to_word.items()]))
    final_sequence = sequence_simplifier.rebuild_sequence()
    print("\n".join([elt_to_str(elt) for elt in final_sequence]))
    timer.print_times()
    plt.plot(data)
    plt.ylim(0,data[0]*1.05)
    plt.show()