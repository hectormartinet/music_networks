import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rd

class Node:
    def __init__(self, note) -> None:
        self.pitch = note.pitch
        self.pitch_class = note.pitch
        self.duration = note.duration.quarterLength


    def to_note(self):
        pass

def music_to_graph(noteList, G):
    for i in range(len(noteList)-2):
        prev_node, next_node = Node(noteList[i]), Node(noteList[i+1])
        if prev_node in G and next_node in G[prev_node]:
            G[prev_node][next_node]['weight'] += 1
        else :
            G.add_edge(prev_node, next_node, weight=1)
    return G