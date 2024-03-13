import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rd

# Get Bach prelude in C
s = ms.corpus.parse('bwv846')
notes = [elt for elt in s.stripTies().recurse().notes if not elt.isChord]
# Reduce to one voice by sorting
notes.sort(key = lambda elt: elt.measureNumber + 0.25*elt.offset)

order = 1

# build graph
G = nx.DiGraph()
for i in range(len(notes)-order-1):
    previous_node = []
    next_node = []
    for k in range(order):
        previous_node.append(notes[i+k].pitch)
        next_node.append(notes[i+k+1].pitch)
    previous_node = tuple(previous_node)
    next_node = tuple(next_node)
    if i>0 and next_node in G[previous_node]:
        G[previous_node][next_node]['weight'] += 1
    else :
        G.add_edge(previous_node,next_node, weight=1)

nx.draw(G)
print(G)
plt.show()
        
# random walk to generate melody
L = 7*8+1
first_node = [n for n in G][rd.randint(0,len(G))]
nodes = [first_node]
melody = [note for note in first_node]


for i in range(L-2):
    total_out_weight = sum([G[nodes[-1]][node]['weight'] for node in G[nodes[-1]]])
    if total_out_weight == 0: break
    p = rd.randint(0,total_out_weight)
    acc = 0
    for node in G[nodes[-1]]:
        acc += G[nodes[-1]][node]['weight']
        if acc > p:
            nodes.append(node)
            melody.append(node[-1])
            break

stream = ms.stream.Stream()
for pitch in melody :
    note = ms.note.Note(str(pitch))
    note.duration.quarterLength = 0.5
    stream.append(note)
stream.show()