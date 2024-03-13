import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, element) -> None:
        if element.isRest :
            self.pitch = None
        else:
            if element.isChord :
                self.pitch = element[-1].pitch
            else:
                self.pitch = element.pitch
        self.duration = element.duration.quarterLength

    def data(self):
        return (self.pitch, self.duration)

    def to_note(self):
        if self.pitch is None :
            note = ms.note.NotRest
        else:
            note = ms.note.Note(self.pitch)
        note.duration.quarterLength = self.duration
        return note

def node_entropy(graph, node):
    entropy = 0
    total_out_weight = sum([graph[node][adj_node]['weight'] for adj_node in graph[node]])
    for adj_node in graph[node]:
        p = graph[node][adj_node]['weight']/total_out_weight
        entropy -= p*np.log(p)
    return entropy

file = ms.converter.parse('schubert_quartet.mid')
graphs = []
for part in file[1:5]:
    G = nx.DiGraph()
    elements = [elt for elt in part.stripTies().recurse().notesAndRests]
    for part_idx in range(len(elements)-1):
        previous_node = Node(elements[part_idx]).data()
        next_node = Node(elements[part_idx+1]).data()
        if part_idx>0 and next_node in G[previous_node]:
            G[previous_node][next_node]['weight'] += 1
        else :
            G.add_edge(previous_node, next_node, weight=1)
    graphs.append(G)
    # print(G)
    # nx.draw(G)
    # plt.show()
instruments = ["Violin I", "Violin II", "Viola", "Violoncello"]
for part_idx in range(4):
    part = file[part_idx+1]
    elements = [elt for elt in part.stripTies().recurse().notesAndRests]
    total_offset = 0
    x, y = [],[]
    for note_idx in range(len(elements)):
        if total_offset > 50 : break
        node = Node(elements[note_idx]).data()
        entropy = node_entropy(graphs[part_idx],node)
        x.append(total_offset)
        y.append(entropy)
        total_offset += elements[note_idx].duration.quarterLength/4
        x.append(total_offset)
        y.append(entropy)
    plt.plot(x, y, label = instruments[part_idx])
    plt.legend()
plt.show()
