import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rd

# Get Bach prelude in C
s = ms.corpus.parse('bwv846')
notes = [elt for elt in s.stripTies().recurse().notes if not elt.isChord]
# Reduce to one voice by sorting
notes.sort(key = lambda elt: elt.measureNumber + 0.25*elt.offset)

# build graph
G = nx.DiGraph()

for i in range(len(notes)-2):
    if notes[i].pitch in G and notes[i+1].pitch in G[notes[i].pitch]:
        G[notes[i].pitch][notes[i+1].pitch]['weight'] += 1
    else :
        G.add_edge(notes[i].pitch,notes[i+1].pitch, weight=1)

# nx.draw(G, with_labels=True)
# plt.show()
        
# random walk to generate melody
L = 7*8+1
firstNote = [n for n in G][rd.randint(0,len(G))]
melody = [firstNote]
# melody = [ms.pitch.Pitch('C4')]

for i in range(L-1):
    total_out_weight = sum([G[melody[-1]][note]['weight'] for note in G[melody[-1]]])
    p = rd.randint(0,total_out_weight)
    acc = 0
    for note in G[melody[-1]]:
        acc += G[melody[-1]][note]['weight']
        if acc > p:
            melody.append(note)
            break

stream = ms.stream.Stream()
for pitch in melody :
    note = ms.note.Note(str(pitch))
    note.duration.quarterLength = 0.5
    stream.append(note)
stream.show()