from multilayer_net import MultiLayerNetwork
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import music21 as ms
import tqdm

class SuffixTree:
    def __init__(self, input_file=None):
        self.weight = 0
        self.children = {}
        self.transition_weight = {}
        if input_file is not None:
            with open(input_file, 'r') as openfile:
                data = json.load(openfile)
                self.load(data)

    def load(self, data):
        self.weight = data["weight"]
        self.transition_weight = data["transition_weight"]
        for node, child in data["children"].items():
            tree = SuffixTree()
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
    
    def pattern_depth(self, lst_nodes, copy_node_lst=True):
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if lst_nodes:
            node = lst_nodes.pop()
            if node in self.children:
                return 1 + self.children[node].pattern_depth(lst_nodes, copy_node_lst=False)
        return 0
    
    def proba(self, lst_nodes, node, copy_node_lst=True):
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if lst_nodes:
            last_node = lst_nodes.pop()
            if last_node in self.children:
                return self.children[last_node].proba(lst_nodes, node, copy_node_lst=False)
        return self.next_proba(node)
    
    def get_tree(self, lst_nodes, copy_node_lst=True):
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if lst_nodes:
            node = lst_nodes.pop()
            if node in self.children:
                return self.children[node].get_tree(lst_nodes, copy_node_lst=False)
        return self
    
    def add_all_prefix_children(self, lst_nodes, next_node, copy_node_lst=True):
        self.weight += 1
        if copy_node_lst:
            lst_nodes = [node for node in lst_nodes]
        if next_node in self.transition_weight:
            self.transition_weight[next_node] += 1
        else :
            self.transition_weight[next_node] = 1
        if lst_nodes:
            node = lst_nodes.pop()
            if node not in self.children:
                self.children[node] = SuffixTree()
            self.children[node].add_all_prefix_children(lst_nodes, next_node, copy_node_lst=False)

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

    def stringify(self, depth=0):
        output = "root:" + str(self.weight) + "\n" if depth==0 else ""
        for node, weight in sorted(self.transition_weight.items(), key=lambda x:-x[1]):
            output += "|  "*depth + "[" + str(round(weight/self.weight,3)) + ":" + node + "\n" 
        for node,child in sorted(self.children.items(), key=lambda x:-x[1].weight):
            output += "|  "*depth + "├──" + str(child.weight) + " " + node + "\n"
            output += child.stringify(depth+1)
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

    def add_nodes(self, nodes_lst, max_order):
        for i in range(len(nodes_lst)):
            self.add_all_prefix_children(nodes_lst[max(0,i-max_order):i], nodes_lst[i], False)

def extract_node_info(node):
    return json.loads(node)

def apply_diatonic_interval(pitch, interval, sharp_number=0):
    sharp_order = ["F","C","G","D","A","E","B"]
    basic_scale = ["C","D","E","F","G","A","B"]
    offset = basic_scale.index(pitch.name[0])
    idx = (interval+basic_scale)%len(basic_scale)
    return basic_scale[idx]

if __name__ == "__main__":

    t=5
    K=1
    L=1000

    # directory = "datasets\\mozart_sonatas\\"
    # directory = "midis\\invent_bach\\"
    # midi_files = [directory + f for f in os.listdir(directory)]
    # midi_files = [str(elt) for elt in ms.corpus.getComposer('bach', fileExtensions=('xml',))]
    midi_files = ["midis\\invent_bach\\invent1.mid"]
    output_file = "trees\\bach_invent1_interval.json"
    net = MultiLayerNetwork(use_gui=False, enharmony=False, pitch=False, structure="monolayer", diatonic_interval=True, duration=False, midi_files=midi_files)
    tree = SuffixTree()
    loading_bar = tqdm.tqdm(total=len(midi_files))
    for midi_file in midi_files:
        net.load_new_midi(midi_file)
        lst_nodes = net._get_nodes_list()
        for i in range(net.nb_layers):
            tree.add_nodes(net._get_nodes_list(i),L)
        loading_bar.update(1)
    
    tree.weight_pruning(t=t)
    # tree.similarity_pruning(K=K)
    # print(tree)

    # Test saving and loading
    tree.save(output_file)
    # loaded_tree = SuffixTree(output_file)
    # print(loaded_tree.nb_nodes())
    # print(loaded_tree.max_depth())
    # print(loaded_tree.weight)
    # print(loaded_tree)