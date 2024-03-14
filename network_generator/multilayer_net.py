import music21 as ms
import networkx as nx
# import pandas as pd
# import numpy as np
import os
# import matplotlib.pyplot as plt
# import collections
from tqdm import tqdm
# import pyautogui
import math
# import random
import json

class MultiLayerNetwork:
    def __init__(self, midifilename, outfolder, **kwargs):
        self.rest = self.get_param_or_default(kwargs, "rest", False)
        self.octave = self.get_param_or_default(kwargs, "octave", False)
        self.pitch = self.get_param_or_default(kwargs, "pitch", True)
        self.duration = self.get_param_or_default(kwargs, "duration", False)
        self.offset = self.get_param_or_default(kwargs, "offset", False)
        self.offset_period = self.get_param_or_default(kwargs, "offset_period", 1)
        self.Net=nx.DiGraph()
        self.subNet = []
        self.outfolder = outfolder
        self.stream_list=[]; self.negative_stream_list=[]
        self.instruments=[]
        self.intergraph = None
        self.name = midifilename
        self.midi_file = midifilename
        whole_piece = ms.converter.parse(midifilename)
        for part in whole_piece.parts: # loads each channel/instrument into stream list            
            self.stream_list.append(part)
        for el in whole_piece.recurse():
            if 'Instrument' in el.classes:
                self.instruments.append(str(el))

    def get_param_or_default(self, param_dict, param, default):
        if param_dict.get(param) is None : return default
        return param_dict[param]

    def is_buildable(self, elt):
        if elt.isRest and not self.rest :
            return False
        return True

    def build_node(self, elt, i):
        assert(self.is_buildable(elt))
        node = {"layer":i}
        if elt.isRest:
            node["rest"] = True
        if self.pitch:
            if elt.isNote:
                if self.octave:
                    node["pitch"] = str(elt.pitch)
                else:
                    node["pitch"] = elt.pitch.name
            if elt.isChord:
                if self.octave:
                    node["pitch"] = " ".join(list(set([str(pitch) for pitch in elt.pitches])))
                else:
                    node["pitch"] = " ".join(list(set([pitch.name for pitch in elt.pitches])))
        if self.duration:
            node["duration"] = elt.duration.quarterLength
        if self.offset:
            node["offset"] = elt.offset - self.offset_period*math.floor(elt.offset/self.offset_period)
        return str(node)

    def node_infos(self, node):
        return json.loads(node)
    
    def stream_to_network(self):
        s_len = len(self.stream_list)
        print("[+] Creating network - Intra-layer processing")
        pbar = tqdm(total=s_len)

        for i in range(s_len):  # For each instrument
            last_note = self.process_intra_layer(i)
            pbar.update(1)
        print("[+] Creating network - Inter-layer processing")
        self.process_inter_layer()
        return self.Net

    def process_intra_layer(self, i, previous_node=None):
        s_flat = self.stream_list[i].flatten()
        for elt in s_flat.notesAndRests:
            if not self.is_buildable(elt): continue
            node = self.build_node(elt, i)
            timestamp = float(elt.offset)
            self.add_or_update_node(node, i)
            self.add_or_update_edge(previous_node, node)
            previous_node = node
    
    def process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [(elt.offset, elt.quarterLength, idx, self.build_node(elt,idx)) for idx in range(s_len) 
            for elt in self.stream_list[idx].flatten().notesAndRests if self.is_buildable(elt)]
        all_nodes_infos.sort(key=lambda x: x[0])
        nb_notes = len(all_nodes_infos)
        for i in range(nb_notes):
            offset, duration, idx, node = all_nodes_infos[i]
            j = i+1
            while  j<nb_notes and all_nodes_infos[j][0] < offset + duration:
                offset2, duration2, idx2, node2 = all_nodes_infos[j]
                if idx != idx2:
                    # add undirected edge
                    self.add_or_update_edge(node, node2)
                    self.add_or_update_edge(node2, node)
                j += 1

    def add_or_update_node(self, node, i):
        self.Net.add_node(node, l=i)
    
    def add_or_update_edge(self, from_node, to_node):
        if from_node is None: return
        if self.Net.has_edge(from_node, to_node):
            self.Net[from_node][to_node]["weight"] += 1
        else:
            self.Net.add_edge(from_node, to_node, weight=1)

    def export_net(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        print("[+] Writing main graphml file to : " + filename)
        nx.write_graphml(self.Net, filename)

    def create_net(self):
        """Create the main network
        """
        print("[+] Converting MIDI file to network")
        self.stream_to_network()
    
    def get_net(self):
        """Getter for the network

        Returns:
            NetwrokX: The main network
        """
        return self.Net
    
    

if __name__ == "__main__" :
    input_file_path = 'midis/schubert_quartet.mid'  # Replace with your MIDI file path
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(input_file_path, output_folder, pitch=True, duration=False, offset=False, offset_period=4)

    # Call createNet function
    net1.create_net()

    # Derive the output filename from the input MIDI filename
    output_filename = os.path.splitext(os.path.basename(input_file_path))[0] + '.graphml'

    # Extract the name without extension
    name_without_extension = os.path.splitext(output_filename)[0]

    # Construct the path for the new subfolder
    subfolder_path = os.path.join(output_folder, name_without_extension)

    # Check if the subfolder exists; if not, create it
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Export the multilayer network
    net1.export_net(os.path.join(subfolder_path, output_filename))

ms.stream.iterator
