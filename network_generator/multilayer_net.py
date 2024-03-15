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
        self.net=nx.DiGraph()
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

    def build_node(self, infos):
        node = {}
        node["layer"] = infos["layer"]
        if self.rest:
            node["rest"] = infos["rest"]
        if self.pitch:
            node["pitch"] = infos["pitch"]
        if self.duration:
            node["duration"] = infos["duration"]
        if self.offset:
            node["offset"] = infos["offset"]
        return str(node)
    
    def parse_elt(self, elt, i):
        assert(self.is_buildable(elt))
        infos = {}
        infos["layer"] = i
        infos["rest"] = elt.isRest
        infos["duration"] = elt.duration.quarterLength
        infos["offset"] = elt.offset - self.offset_period*math.floor(elt.offset/self.offset_period)
        infos["pitch"] = self.parse_pitch(elt)
        return infos
    
    def parse_pitch(self,elt):
        if elt.isNote:
            if self.octave:
                return str(elt.pitch)
            else:
                return elt.pitch.name
        if elt.isChord:
            if self.octave:
                unique_notes = list(set([str(pitch) for pitch in elt.pitches]))
                unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
                return " ".join(unique_notes)
            else:
                unique_notes = list(set([str(pitch.name) for pitch in elt.pitches]))
                unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
                return " ".join(unique_notes)
        if elt.isRest:
            return "rest"
        assert(False)

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
        return self.net

    def process_intra_layer(self, i, previous_node=None):
        s_flat = self.stream_list[i].flatten()
        for elt in s_flat.notesAndRests:
            if not self.is_buildable(elt): continue
            infos = self.parse_elt(elt, i)
            node = self.build_node(infos)
            timestamp = float(elt.offset)
            self.add_or_update_node(node, i)
            self.add_or_update_edge(previous_node, node)
            previous_node = node
    
    def process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [(elt.offset, elt.quarterLength, idx, self.build_node(self.parse_elt(elt, idx))) for idx in range(s_len) 
            for elt in self.stream_list[idx].flatten().notesAndRests if self.is_buildable(elt)]
        all_nodes_infos.sort(key=lambda x: x[0])
        nb_notes = len(all_nodes_infos)
        for i in range(nb_notes):
            offset, duration, idx, node = all_nodes_infos[i]
            j = i+1
            while  j<nb_notes and all_nodes_infos[j][0] < offset + duration:
                _, _, idx2, node2 = all_nodes_infos[j]
                if idx != idx2:
                    # add undirected edge
                    self.add_or_update_edge(node, node2)
                    self.add_or_update_edge(node2, node)
                j += 1

    def add_or_update_node(self, node, i):
        self.net.add_node(node, l=i)
    
    def add_or_update_edge(self, from_node, to_node):
        if from_node is None: return
        if self.net.has_edge(from_node, to_node):
            self.net[from_node][to_node]["weight"] += 1
        else:
            self.net.add_edge(from_node, to_node, weight=1)

    def export_net(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        print("[+] Writing main graphml file to : " + filename)
        nx.write_graphml(self.net, filename)

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
        return self.net
    
    def get_sub_net(self, layer=None):
        """Return the list of subnetworks

        Args:
            layer (Number of the layer, optional): Specify the layer to return. Defaults to None.

        Returns:
            List NetworkX: The list of subnetworks
        """
        #list of subnets (maybe to change function in multilayer class)
        # s_len=len(self.stream_list)
        # self.subNet =[]
        # for i in range (0,s_len):
        #     self.subNet.append(nx.subgraph_view(self.net, lambda : ))
        # t=self.filter_edges(n)
        # H = n.edge_subgraph(t)
        # self.intergraph = H.to_undirected()
        # return self.subNet, self.intergraph
        pass
    
    

if __name__ == "__main__" :
    input_file_path = 'midis/bwv772.mid'  # Replace with your MIDI file path
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(input_file_path, output_folder, pitch=True, duration=False, offset=False, offset_period=4, octave=True)

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
