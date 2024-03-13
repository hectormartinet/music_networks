import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rd
import random

T = 8 # periode of the ""beat checking""

# Get Bach prelude in C
s = ms.corpus.parse('bwv846')
notes = [elt for elt in s.stripTies().recurse().notes if not elt.isChord]
# Reduce to one voice by sorting
notes.sort(key = lambda elt: elt.measureNumber + 0.25*elt.offset)

# build graph
G = nx.DiGraph()
# On beat = 0, other 1
for i in range(len(notes)-2):
    if (notes[i].pitch, i%T) in G and (notes[i+1].pitch, (i+1)%T) in G[(notes[i].pitch,i%T)]:
        G[(notes[i].pitch, i%T)][(notes[i+1].pitch,(i+1)%T)]['weight'] += 1
    else :
        G.add_edge((notes[i].pitch,i%T),(notes[i+1].pitch,(i+1)%T), weight=1)

# nx.draw(G, with_labels=True)
# plt.show()
print(G)
print(len(G))
        
# random walk to generate melody
L = 15*8+1
firstNote = random.choice([n for n in G if n[1]==0])[0]
melody = [firstNote]
# melody = [ms.pitch.Pitch('C4')]

for i in range(1,L):
    total_out_weight = sum([G[(melody[-1], (i-1)%T)][note]['weight'] for note in G[(melody[-1],(i-1)%T)]])
    p = rd.randint(0,total_out_weight)
    acc = 0
    for note in G[(melody[-1], (i-1)%T)]:
        acc += G[(melody[-1], (i-1)%T)][note]['weight']
        if acc > p:
            melody.append(note[0])
            break

stream = ms.stream.Stream()
for pitch in melody :
    note = ms.note.Note(str(pitch))
    note.duration.quarterLength = 0.5
    stream.append(note)
stream.show()