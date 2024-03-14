import music21 as ms
import networkx as nx
# import pandas as pd
# import numpy as np
import os
# import matplotlib.pyplot as plt
# import collections
from tqdm import tqdm
# import pyautogui
# import math
# import random
import json

class MultiLayerNetwork:
    def __init__(self, midifilename, outfolder, options=None, **kwargs):
        self.rest = self.get_param_or_default(kwargs, "rest", False)
        self.octave = self.get_param_or_default(kwargs, "octave", False)
        self.pitch = self.get_param_or_default(kwargs, "pitch", True)
        self.duration = self.get_param_or_default(kwargs, "duration", False)
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
    
    def process_inter_layer(self, ):
        pass

    def add_or_update_node(self, node, i):
        self.Net.add_node(node, l=i)
    
    def add_or_update_edge(self, previous_node, node):
        if previous_node is not None:
            self.Net.add_edge(previous_node, node)

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
    
    

if __name__ == "__main__" :
    input_file_path = 'midis/schubert_quartet.mid'  # Replace with your MIDI file path
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(input_file_path, output_folder)

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

