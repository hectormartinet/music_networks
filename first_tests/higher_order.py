import music21 as ms
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rd

# Get Bach prelude in C
s = ms.corpus.parse('bwv846')
notes = [elt for elt in s.stripTies().recurse().notes if not elt.isChord]
# Reduce to one voice by sorting
notes.sort(key = lambda elt: elt.measureNumber + 0.25*elt.offset)
print(notes[0])


# build graph
G = nx.DiGraph()

for i in range(len(notes)-3):
    if (notes[i].pitch,notes[i+1].pitch) in G and (notes[i+1].pitch,notes[i+2].pitch) in G[(notes[i].pitch,notes[i+1].pitch)]:
        G[(notes[i].pitch,notes[i+1].pitch)][(notes[i+1].pitch,notes[i+2].pitch)]['weight'] += 1
    else :
        G.add_edge((notes[i].pitch,notes[i+1].pitch),(notes[i+1].pitch,notes[i+2].pitch), weight=1)

# nx.draw(G, with_labels=True)
# print(G)
# plt.show()
        
# random walk to generate melody
L = 7*8+1
firstNote, secondNote = [n for n in G][rd.randint(0,len(G))]
melody = [firstNote, secondNote]
# melody = [ms.pitch.Pitch('C4')]

for i in range(L-2):
    total_out_weight = sum([G[(melody[-2],melody[-1])][note]['weight'] for note in G[(melody[-2],melody[-1])]])
    p = rd.randint(0,total_out_weight)
    acc = 0
    for note in G[(melody[-2],melody[-1])]:
        acc += G[(melody[-2],melody[-1])][note]['weight']
        if acc > p:
            melody.append(note[1])
            break

stream = ms.stream.Stream()
for pitch in melody :
    print(pitch)
    note = ms.note.Note(str(pitch))
    note.duration.quarterLength = 0.5
    stream.append(note)
stream.show()