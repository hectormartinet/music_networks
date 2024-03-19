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
        self.get_params(**kwargs)
        self.net=nx.DiGraph()
        self.sub_net = []
        self.outfolder = outfolder
        self.stream_list=[]; self.negative_stream_list=[]
        self.instruments=[]
        self.intergraph = None
        self.name = midifilename
        self.midi_file = midifilename
        whole_piece = ms.converter.parse(midifilename)
        for part in whole_piece.parts: # loads each channel/instrument into stream list
            if self.transpose:
                self.stream_list.append(self.stream_to_C(part))
            else :  
                self.stream_list.append(part)
        for el in whole_piece.recurse():
            if 'Instrument' in el.classes:
                self.instruments.append(str(el))

    def get_params(self, **kwargs):
        self.rest = self.get_param_or_default(kwargs, "rest", False)
        self.stop_at_unwanted = self.get_param_or_default(kwargs, "stop_at_unwanted", False)
        self.octave = self.get_param_or_default(kwargs, "octave", False)
        self.pitch = self.get_param_or_default(kwargs, "pitch", True)
        self.duration = self.get_param_or_default(kwargs, "duration", False)
        self.offset = self.get_param_or_default(kwargs, "offset", False)
        self.offset_period = self.get_param_or_default(kwargs, "offset_period", 1)
        assert(self.offset_period > 0)
        self.transpose = self.get_param_or_default(kwargs, "transpose", False)
        self.strict_link = self.get_param_or_default(kwargs, "strict_link", False)
        self.layer = self.get_param_or_default(kwargs, "layer", True)
        self.interval = self.get_param_or_default(kwargs, "interval", False)
        self.diatonic = self.get_param_or_default(kwargs, "diatonic", False)
        self.chromatic = not self.diatonic

    def stream_to_C(self, part):
        k = part.flatten().analyze('key')
        i = ms.interval.Interval(k.tonic, ms.pitch.Pitch('C'))
        part_transposed = part.transpose(i)
        return part_transposed

    def get_param_or_default(self, param_dict, param, default):
        if param_dict.get(param) is None : return default
        print("[+] Parameter " + str(param) + " set to "+ str(param_dict[param]))
        return param_dict[param]

    def is_buildable(self, elt):
        if elt.isRest and not self.rest :
            return False
        return True

    def parse_elt(self, elt, i):
        assert(self.is_buildable(elt))
        infos = {}
        infos["layer"] = i
        infos["rest"] = elt.isRest
        infos["duration"] = elt.duration.quarterLength
        infos["offset"] = elt.offset - self.offset_period*math.floor(elt.offset/self.offset_period)
        infos["timestamp"] = elt.offset
        infos["pitch"] = self.parse_pitch(elt)
        infos["pitch_class"] = self.parse_pitch_class(elt)
        return infos

    def build_node(self, infos):
        node = {}
        if self.rest:
            node["rest"] = infos["rest"]
        if self.pitch:
            if self.octave:
                node["pitch"] = infos["pitch"]
            else:
                node["pitch"] = infos["pitch_class"]
        if self.duration:
            node["duration"] = infos["duration"]
        if self.offset:
            node["offset"] = infos["offset"]
        if self.layer:
            return (infos["layer"],str(node))
        return str(node)
    
    def parse_pitch(self,elt):
        if elt.isNote:
            return str(elt.pitch)
        if elt.isChord:
            unique_notes = list(set([str(pitch) for pitch in elt.pitches]))
            unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
            return " ".join(unique_notes)
        if elt.isRest:
            return "rest"
        assert(False)

    def parse_pitch_class(self,elt):
        if elt.isNote:
            return elt.pitch.name
        if elt.isChord:
            unique_notes = list(set([str(pitch.name) for pitch in elt.pitches]))
            unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
            return " ".join(unique_notes)
        if elt.isRest:
            return "rest"
        assert(False)
    
    def stream_to_network(self):
        s_len = len(self.stream_list)
        print("[+] Creating network - Intra-layer processing")
        pbar = tqdm(total=s_len)

        for i in range(s_len):  # For each instrument
            self.process_intra_layer(i if self.layer else 0)
            pbar.update(1)
        if self.layer:
            print("[+] Creating network - Inter-layer processing")
            self.process_inter_layer()
        return self.net

    def process_intra_layer(self, i, previous_node=None):
        s_flat = self.stream_list[i].flatten()
        for elt in s_flat.notesAndRests:
            if not self.is_buildable(elt):
                if self.stop_at_unwanted:
                    previous_node = None
                continue
            infos = self.parse_elt(elt, i)
            node = self.build_node(infos)
            self.add_or_update_node(node, infos)
            self.add_or_update_edge(previous_node, node, inter=False)
            previous_node = node
    
    def process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [self.parse_elt(elt, idx) for idx in range(s_len) 
            for elt in self.stream_list[idx].flatten().notesAndRests if self.is_buildable(elt)]
        all_nodes_infos.sort(key=lambda x: x["timestamp"])
        nb_notes = len(all_nodes_infos)
        for i in range(nb_notes):
            timestamp = all_nodes_infos[i]["timestamp"]
            duration = all_nodes_infos[i]["duration"]
            layer = all_nodes_infos[i]["layer"]
            node = self.build_node(all_nodes_infos[i])
            for j in range(i+1,nb_notes):
                timestamp2 = all_nodes_infos[j]["timestamp"]
                if (self.strict_link and timestamp2 > timestamp) or timestamp2 >= timestamp + duration:
                    break
                layer2 = all_nodes_infos[j]["layer"]
                node2 = self.build_node(all_nodes_infos[j])
                if layer != layer2:
                    # add undirected edge
                    self.add_or_update_edge(node, node2, inter=True)
                    self.add_or_update_edge(node2, node, inter=True)
                j += 1

    def add_or_update_node(self, node, infos):
        if not self.net.has_node(node):
            self.net.add_node(node, 
                weight=1, 
                layer = infos["layer"] if self.layer else [infos["layer"]], 
                pitch = infos["pitch"] if self.pitch and self.octave else [infos["pitch"]],
                pitch_class = infos["pitch_class"] if self.pitch else [infos["pitch_class"]],
                duration = float(infos["duration"]) if self.duration else [float(infos["duration"])],
                offset = float(infos["offset"]) if self.offset else [float(infos["offset"])],
                timestamps =[float(infos["timestamp"])],
                rest = infos["rest"] if self.rest else [infos["rest"]],
            )
        else :
            self.net.nodes[node]["weight"] += 1
            if not self.layer:
                self.net.nodes[node]["layer"].append(infos["layer"])
            if not (self.pitch and self.octave):
                self.net.nodes[node]["pitch"].append(infos["pitch"])
            if not self.pitch:
                self.net.nodes[node]["pitch_class"].append(infos["pitch_class"])
            if not self.duration:
                self.net.nodes[node]["duration"].append(float(infos["duration"]))
            if not self.offset:
                self.net.nodes[node]["offset"].append(float(infos["offset"]))
            self.net.nodes[node]["timestamps"].append(float(infos["timestamp"]))
            if not self.rest:
                self.net.nodes[node]["rest"].append(infos["rest"])
    
    def add_or_update_edge(self, from_node, to_node, inter):
        if from_node is None or to_node is None: return
        if self.net.has_edge(from_node, to_node):
            self.net[from_node][to_node]["weight"] += 1
        else:
            self.net.add_edge(from_node, to_node, weight=1, inter=inter)

    def convert_attributes_to_str(self):
        for node in self.net.nodes:
            if not self.layer:
                self.net.nodes[node]["layer"] = str(self.net.nodes[node]["layer"])
            if not (self.pitch and self.octave):
                self.net.nodes[node]["pitch"] = str(self.net.nodes[node]["pitch"])
            if not self.pitch:
                self.net.nodes[node]["pitch_class"] = str(self.net.nodes[node]["pitch_class"])
            if not self.duration:
                self.net.nodes[node]["duration"] = str(self.net.nodes[node]["duration"])
            if not self.offset:
                self.net.nodes[node]["offset"] = str(self.net.nodes[node]["offset"])
            self.net.nodes[node]["timestamps"] = str(self.net.nodes[node]["timestamps"])
            if not self.rest:
                self.net.nodes[node]["rest"] = str(self.net.nodes[node]["rest"])

    def export_net(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        print("[+] Writing main graphml file to : " + filename)
        nx.write_graphml(self.net, filename)

    def export_sub_net(self, folder,filename2):
        """Export the subnet to a graphml file

        Args:
            folder (string): Output folder
        """
        try:
            os.mkdir(folder)
        except:
            pass
        print("[+] Writing " + str(len(self.stream_list)) + " graphml subnet files to : " + folder)
        filename = folder +filename2+ "l"
        for i in range(0,len(self.sub_net)):
            cur_out = filename + "_" + str(i) + ".graphml"
            nx.write_graphml(self.sub_net[i], cur_out)

    def create_net(self):
        """Create the main network
        """
        print("[+] Converting MIDI file to network")
        self.stream_to_network()
        self.convert_attributes_to_str()
    
    def get_net(self):
        """Getter for the network

        Returns:
            NetwrokX: The main network
        """
        return self.net
    
    def get_sub_net(self):
        """Return the list of subnetworks

        Args:
            layer (Number of the layer, optional): Specify the layer to return. Defaults to None.

        Returns:
            List NetworkX: The list of subnetworks
        """
        if self.layer:
            return self.net
        s_len=len(self.stream_list)
        self.sub_net =[]
        for i in range(s_len):
            def filter(node, layer=i): return node[0]==layer # use default arg to avoid dependancy on i
            self.sub_net.append(nx.subgraph_view(self.net, filter_node=filter))
        # self.intergraph = nx.subgraph_view(self.net, filter_edge=lambda edge: edge.inter).to_undirected()
        return self.sub_net, self.intergraph
    
    def list_to_string(self,my_list):
        return ','.join(str(x) for x in my_list)
    

if __name__ == "__main__" :
    input_file_path = 'midis/invent1.mid'  # Replace with your MIDI file path
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(input_file_path, output_folder, pitch=True, duration=True, offset=True, offset_period=1, octave=True, rest=False, stop_at_unwanted=True, layer=True)

    # Call createNet function
    net1.create_net()

    # Get the subnet and intergraph
    net1.get_sub_net()

    # Derive the output filename from the input MIDI filename
    output_filename = os.path.splitext(os.path.basename(input_file_path))[0] + '.graphml'

    # Extract the name without extension
    name_without_extension = os.path.splitext(output_filename)[0]

    # Construct the path for the new subfolder
    subfolder_path = os.path.join(output_folder, name_without_extension)

    # Check if the subfolder exists; if not, create it
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    net1.convert_attributes_to_str()

    # Export the multilayer network
    net1.export_net(os.path.join(subfolder_path, output_filename))

    # Export subnets
    net1.export_sub_net(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file_path))[0]) + os.path.sep, os.path.splitext(os.path.basename(input_file_path))[0])

