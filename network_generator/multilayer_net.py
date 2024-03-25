import music21 as ms
import networkx as nx
import parameter_picker as par_pick
# import pandas as pd
# import numpy as np
import os
# import matplotlib.pyplot as plt
# import collections
from tqdm import tqdm
# import pyautogui
import math
# import random

class MultiLayerNetwork:
    def __init__(self, use_gui=True, **kwargs):
        params = self.get_params(**kwargs)
        if use_gui:
            params = self.pick_parameters(**params)
        self.parse_params(**params)
        self.net=nx.DiGraph()
        self.nodes_lists = []

    def get_files(self, file_or_folder, extension='.mid'):
        if file_or_folder.endswith("/"):
            return [file_or_folder + file for file in os.listdir(file_or_folder) if os.path.splitext(file)[1]==extension]
        else:
            return [file_or_folder]

    def get_params(self, **kwargs):
        default_params = {
            "rest":False,
            "stop_at_ignored":True,
            "octave":False,
            "pitch":True,
            "duration":False,
            "offset":False,
            "offset_period":1.,
            "transpose":False,
            "strict_link":False,
            "layer":True,
            "diatonic_interval":False,
            "chromatic_interval":False,
            "midi_folder_or_file":"midis/",
            "outfolder":"results/"
        }
        for key in default_params.keys():
            if key in kwargs:
                default_params[key] = kwargs[key]
        return default_params

    def pick_parameters(self, **params):
        return par_pick.get_parameters(**params)

    def load_new_midi(self, midifilename):
        self.sub_net = []
        self.stream_list = []
        self.parsed_stream_list = []
        self.instruments = []
        self.intergraph = None
        self.name = midifilename
        self.midi_file = midifilename
        whole_piece = ms.converter.parse(midifilename, quarterLengthDivisors = (16,))
        for part in whole_piece.parts: # loads each channel/instrument into stream list
            if self.transpose:
                self.stream_list.append(self.stream_to_C(part))
            else :  
                self.stream_list.append(part)
            self.parsed_stream_list = [self.build_parsed_list(part, i) for i,part in enumerate(self.stream_list)]
        for elt in whole_piece.recurse():
            if 'Instrument' in elt.classes:
                self.instruments.append(str(elt))

    def parse_params(self, **params):
        self.rest = params["rest"]
        self.stop_at_ignored = params["stop_at_ignored"]
        self.octave = params["octave"]
        self.pitch = params["pitch"]
        self.duration = params["duration"]
        self.offset = params["offset"]
        self.offset_period = params["offset_period"]
        assert(self.offset_period > 0)
        self.transpose = params["transpose"]
        self.strict_link = params["strict_link"]
        self.layer = params["layer"]
        self.diatonic_interval = params["diatonic_interval"]
        self.chromatic_interval = params["chromatic_interval"]
        self.midi_files = self.get_files(params["midi_folder_or_file"])
        self.outfolder = params["outfolder"]
    
    @property
    def interval(self): return self.diatonic_interval or self.chromatic_interval

    def stream_to_C(self, part):
        k = part.flatten().analyze('key')
        i = ms.interval.Interval(k.tonic, ms.pitch.Pitch('C'))
        part_transposed = part.transpose(i)
        return part_transposed

    def get_param_or_default(self, param_dict, param, default):
        if param_dict.get(param) is None : return default
        print("[+] Parameter " + str(param) + " set to "+ str(param_dict[param]))
        return param_dict[param]

    def is_ignored(self, parsed_elt):
        if not self.rest and parsed_elt["rest"]:
            return True
        if self.interval and parsed_elt["chord"]:
            return True
        return False

    def parse_elt(self, elt, i):
        infos = {}
        infos["layer"] = i
        infos["rest"] = elt.isRest
        infos["chord"] = elt.isChord
        infos["duration"] = elt.duration.quarterLength
        infos["offset"] = elt.offset - self.offset_period*math.floor(elt.offset/self.offset_period)
        infos["timestamp"] = elt.offset
        infos["pitch"] = self.parse_pitch(elt)
        infos["pitch_class"] = self.parse_pitch_class(elt)
        return infos

    def build_parsed_list(self, part, i):
        lst = [self.parse_elt(elt,i) for elt in part.flatten().notesAndRests]
        if not self.stop_at_ignored :
            lst = [elt for elt in lst if not self.is_ignored(elt)]
        for i in range(len(lst)-1):
            if lst[i]["rest"] or lst[i]["chord"] or lst[i+1]["rest"] or lst[i+1]["chord"]:
                lst[i]["chromatic_interval"] = 0
                lst[i]["diatonic_interval"] = 0
                continue
            interval = ms.interval.Interval(ms.pitch.Pitch(lst[i]["pitch"]), ms.pitch.Pitch(lst[i+1]["pitch"]))
            lst[i]["diatonic_interval"] = interval.diatonic.generic.value
            lst[i]["chromatic_interval"] = interval.chromatic.semitones
        lst[len(lst)-1]["chromatic_interval"] = 0
        lst[len(lst)-1]["diatonic_interval"] = 0
        return lst


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
        if self.diatonic_interval:
            node["diatonic_interval"] = infos["diatonic_interval"]
        if self.chromatic_interval:
            node["chromatic_interval"] = infos["chromatic_interval"]
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
        for parsed_elt in self.parsed_stream_list[i]:
            if self.is_ignored(parsed_elt):
                if self.stop_at_ignored:
                    previous_node = None
                continue
            node = self.build_node(parsed_elt)
            self.add_or_update_node(node, parsed_elt)
            self.add_or_update_edge(previous_node, node, inter=False)
            previous_node = node
    
    def process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [elt for lst in self.parsed_stream_list for elt in lst if not self.is_ignored(elt)]
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
            def conditional_list(elt, no_list):
                return elt if no_list else [elt]
            self.net.add_node(node, 
                weight=1, 
                layer = conditional_list(infos["layer"],self.layer), 
                pitch = conditional_list(infos["pitch"], self.pitch and self.octave),
                pitch_class = conditional_list(infos["pitch_class"], self.pitch),
                chromatic_interval = conditional_list(infos["chromatic_interval"], self.chromatic_interval),
                diatonic_interval = conditional_list(infos["diatonic_interval"], self.diatonic_interval),
                duration = conditional_list(float(infos["duration"]), self.duration),
                offset = conditional_list(float(infos["offset"]), self.offset),
                timestamps =[float(infos["timestamp"])],
                rest = conditional_list(infos["rest"], self.rest),
            )
        else :
            self.net.nodes[node]["weight"] += 1
            def append_if_list(attribute, elt_to_add=None):
                if type(self.net.nodes[node][attribute]) == list:
                    if elt_to_add is None:
                        self.net.nodes[node][attribute].append(infos[attribute])
                    else:
                        self.net.nodes[node][attribute].append(elt_to_add)
            append_if_list("layer")
            append_if_list("pitch")
            append_if_list("pitch_class")
            append_if_list("chromatic_interval")
            append_if_list("diatonic_interval")
            append_if_list("duration",float(infos["duration"]))
            append_if_list("offset", float(infos["offset"]))
            append_if_list("rest")
            self.net.nodes[node]["timestamps"].append(float(infos["timestamp"]))
    
    def add_or_update_edge(self, from_node, to_node, inter):
        if from_node is None or to_node is None: return
        if self.net.has_edge(from_node, to_node):
            self.net[from_node][to_node]["weight"] += 1
        else:
            self.net.add_edge(from_node, to_node, weight=1, inter=inter)

    def convert_attributes_to_str(self):
        def convert_if_list(attribute, node):
            if type(self.net.nodes[node][attribute]) == list :
                self.net.nodes[node][attribute] = self.list_to_string(self.net.nodes[node][attribute])
        for node in self.net.nodes:
            for attribute in self.net.nodes[node].keys():
                convert_if_list(attribute, node)

    def export_net(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        print("[+] Writing main graphml file to : " + filename)
        nx.write_graphml(self.net, filename)

    def export_sub_net(self, folder, filename):
        """Export the subnet to a graphml file

        Args:
            folder (string): Output folder
        """
        try:
            os.mkdir(folder)
        except:
            pass
        print("[+] Writing " + str(len(self.stream_list)) + " graphml subnet files to : " + folder)
        filename = folder + filename + "l"
        for i in range(0,len(self.sub_net)):
            cur_out = filename + "_" + str(i) + ".graphml"
            nx.write_graphml(self.sub_net[i], cur_out)

    def create_net(self):
        """Create the main network
        """
        print("[+] Converting MIDI file to network")
        for midi in self.midi_files:
            self.load_new_midi(midi)
            self.stream_to_network()

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
        if not self.layer:
            return self.net
        s_len = len(self.stream_list)
        self.sub_net =[]
        for i in range(s_len):
            def filter(node, layer=i): return node[0]==layer # use default arg to avoid dependancy on i
            self.sub_net.append(nx.subgraph_view(self.net, filter_node=filter))
        # self.intergraph = nx.subgraph_view(self.net, filter_edge=lambda edge: edge.inter).to_undirected()
        return self.sub_net, self.intergraph
    
    def get_nodes_list(self, layer):
        return [self.build_node(elt) for elt in self.parsed_stream_list[layer]]
    
    def list_to_string(self,my_list):
        return ','.join(str(x) for x in my_list)
    

if __name__ == "__main__" :
    input_file_path = 'midis/invent_bach/'  # Replace with your MIDI file path
    output_folder = 'results/'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(use_gui=True, output_folder = output_folder, midi_folder_or_file = input_file_path)

    # Call createNet function
    net1.create_net()

    # Get the subnet and intergraph
    net1.get_sub_net()

    # Derive the output filename from the input MIDI filename
    if input_file_path.endswith("/"):
        output_filename = os.path.dirname(input_file_path).split("/")[-1] + '.graphml'
    else:
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
    net1.export_sub_net(os.path.join(output_folder, name_without_extension) + os.path.sep, output_filename)

