import music21 as ms
import networkx as nx
import parameter_picker as par_pick
from preset_params import get_preset_params
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
    def __init__(self, use_gui=True, verbosity=1, preset_param=None, name=None, **kwargs):
        if preset_param is not None:
            params = get_preset_params(preset_param)
        else:
            params = self.get_params(**kwargs)
        if use_gui:
            params = self.pick_parameters(params)
        self.name = name
        self.parse_params(**params)
        self.verbosity = verbosity
        self.net = nx.DiGraph()
        self.nodes_lists = []

    def print_if_useful(self, message, verbosity_level):
        if verbosity_level <= self.verbosity:
            print(message)

    def get_params(self, **kwargs):
        default_params = {
            "pitch":True,
            "octave":False,
            "duration":False,
            "rest":False,
            "offset":False,
            "offset_period":1.,
            "enharmony":True,
            "diatonic_interval":False,
            "chromatic_interval":False,
            "chord_function":False,
            "transpose":False,
            "flatten":False,
            "layer":True,
            "strict_link":False,
            "max_link_time_diff":4.,
            "group_by_beat":False,
            "midi_files":["midis/invent_bach/invent1.mid"],
            "outfolder":"results/"
        }
        for key in default_params.keys():
            if key in kwargs:
                default_params[key] = kwargs[key]
        return default_params

    def pick_parameters(self, params):
        return par_pick.get_parameters(params)

    def load_new_midi(self, midifilename):
        self.sub_net = []
        self.stream_list = []
        self.parsed_stream_list = []
        self.instruments = []
        self.intergraph = None
        self.midi_file = midifilename
        self.print_if_useful("Loading new midi : " + midifilename, 2)
        whole_piece = ms.converter.parse(midifilename, quarterLengthDivisors = (16,))
        self.original_key = whole_piece.flatten().analyze('key')
        self.print_if_useful("Analysed key : " + str(self.original_key), 3)
        self.key = self.original_key
        if self.flatten:
            whole_piece = whole_piece.chordify()
            if self.transpose:
                self.stream_list.append(self.stream_to_C(whole_piece))
            else:
                self.stream_list.append(whole_piece)
        else:
            for part in whole_piece.parts: # loads each channel/instrument into stream list
                if self.transpose:
                    self.stream_list.append(self.stream_to_C(part))
                else:
                    self.stream_list.append(part)
            for elt in whole_piece.recurse():
                if 'Instrument' in elt.classes:
                    self.instruments.append(str(elt))
        if self.group_by_beat:
            self.group_notes_by_beat()
        self.parsed_stream_list = [self.build_parsed_list(part, i) for i,part in enumerate(self.stream_list)]

    def parse_params(self, **params):
        self.rest = params["rest"]
        self.pitch = params["pitch"]
        self.octave = params["octave"]
        self.enharmony = params["enharmony"]
        self.duration = params["duration"]
        self.offset = params["offset"]
        self.offset_period = params["offset_period"]
        assert(self.offset_period > 0)
        self.transpose = params["transpose"]
        self.strict_link = params["strict_link"]
        self.max_link_time_diff = params["max_link_time_diff"]
        self.flatten = params["flatten"]
        self.layer = params["layer"] and not self.flatten
        self.diatonic_interval = params["diatonic_interval"] and not self.enharmony
        self.chromatic_interval = params["chromatic_interval"]
        self.chord_function = params["chord_function"]
        self.group_by_beat = params["group_by_beat"]
        self.midi_files = params["midi_files"]
        for file_name in self.midi_files:
            assert(os.path.splitext(file_name)[1] in [".mid", ".mscz"])
        self.outfolder = params["outfolder"]
    
    @property
    def interval(self): return self.diatonic_interval or self.chromatic_interval

    def stream_to_C(self, part):
        if self.original_key.mode == "major":
            i = ms.interval.Interval(self.original_key.tonic, ms.pitch.Pitch('C'))
            self.key = ms.key.Key("C")
        elif self.original_key.mode == "minor":
            i = ms.interval.Interval(self.original_key.tonic, ms.pitch.Pitch('A'))
            self.key = ms.key.Key("a")
        else:
            assert(False)
        return part.transpose(i)

    def is_ignored(self, elt):
        if not self.rest and elt.isRest:
            return True
        if self.interval and elt.isChord:
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
        infos["pitch"] = self.parse_pitch(elt, octave=True)
        infos["pitch_class"] = self.parse_pitch(elt, octave=False)
        infos["chord_function"] = ms.roman.romanNumeralFromChord(elt,self.key).romanNumeral if elt.isChord else "N/A"
        # infos["chord_function"] = ms.roman.romanNumeralFromChord(elt,self.key).figure if elt.isChord else "N/A"
        return infos

    def build_parsed_list(self, part, i):
        lst = [self.parse_elt(elt,i) for elt in part.flatten().notesAndRests if not self.is_ignored(elt)]
        if not lst : return lst
        for i in range(len(lst)-1):
            if lst[i]["rest"] or lst[i]["chord"] or lst[i+1]["rest"] or lst[i+1]["chord"]:
                lst[i]["chromatic_interval"] = 0
                lst[i]["diatonic_interval"] = 0
                continue
            interval = ms.interval.Interval(ms.pitch.Pitch(lst[i]["pitch"]), ms.pitch.Pitch(lst[i+1]["pitch"]))
            lst[i]["diatonic_interval"] = interval.diatonic.generic.value
            lst[i]["chromatic_interval"] = interval.chromatic.semitones
        lst[len(lst)-1]["chromatic_interval"] = 0 # TODO Change default values...
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
            node["layer"] = infos["layer"]
        if self.chord_function:
            node["chord_function"] = infos["chord_function"]
        return json.dumps(node)
    
    def parse_pitch(self, elt, octave):
        if elt.isNote:
            return self.pitch_to_str(elt.pitch, octave)
        if elt.isChord:
            unique_notes = list(set([self.pitch_to_str(pitch, octave) for pitch in elt.pitches]))
            unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
            return " ".join(unique_notes)
        if elt.isRest:
            return "rest"
        assert(False)

    def pitch_to_str(self, pitch, octave):
        if self.enharmony:
            pitch = ms.pitch.Pitch(pitch.midi)
        if octave:
            return str(pitch)
        else:
            return pitch.name 
    
    def stream_to_network(self):
        s_len = len(self.stream_list)
        self.print_if_useful("[+] Creating network - Intra-layer processing", 2)

        for i in range(s_len):  # For each instrument
            self.process_intra_layer(i if self.layer else 0)
        if self.layer and s_len > 1:
            self.print_if_useful("[+] Creating network - Inter-layer processing", 2)
            self.process_inter_layer()
        return self.net

    def process_intra_layer(self, i, prev_elt=None):
        prev_node = self.build_node(prev_elt) if prev_elt is not None else None
        for elt in self.parsed_stream_list[i]:
            node = self.build_node(elt)
            self.add_or_update_node(node, elt)
            if prev_elt is not None:
                time_diff = elt["timestamp"] - prev_elt["timestamp"] - prev_elt["duration"]
                if time_diff <= self.max_link_time_diff:
                    self.add_or_update_edge(prev_node, node, inter=False)
            prev_node = node
            prev_elt = elt
    
    def process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [elt for lst in self.parsed_stream_list for elt in lst]
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
                chord_function = conditional_list(infos["chord_function"], self.chord_function)
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
            append_if_list("chord_function")
            self.net.nodes[node]["timestamps"].append(float(infos["timestamp"]))
    
    def add_or_update_edge(self, from_node, to_node, inter):
        if from_node is None or to_node is None: return
        if self.net.has_edge(from_node, to_node):
            self.net[from_node][to_node]["weight"] += 1
        else:
            self.net.add_edge(from_node, to_node, weight=1, inter=inter)

    def convert_attributes_to_str(self):
        for node in self.net.nodes:
            for attribute in self.net.nodes[node].keys():
                if type(self.net.nodes[node][attribute]) == list :
                    self.net.nodes[node][attribute] = self.list_to_string(self.net.nodes[node][attribute])

    def export_net(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        self.print_if_useful("[+] Writing main graphml file to : " + filename, 1)
        nx.write_graphml(self.net, filename)

    def export_sub_net(self, folder, filename):
        """Export the subnet to a graphml file

        Args:
            folder (string): Output folder
            filename (string): Output filename
        """
        try:
            os.mkdir(folder)
        except:
            pass
        self.print_if_useful("[+] Writing " + str(len(self.stream_list)) + " graphml subnet files to : " + folder, 1)
        filename = folder + filename
        for i in range(0,len(self.sub_net)):
            cur_out = filename + "l_" + str(i) + ".graphml"
            nx.write_graphml(self.sub_net[i], cur_out)
        cur_out = filename + "_intergraph.graphml"
        nx.write_graphml(self.intergraph, cur_out)

    def create_net(self):
        """Create the main network
        """
        nb_files = len(self.midi_files)
        self.print_if_useful("[+] Converting " + str(nb_files) + " MIDI file(s) to network", 1)
        pbar = tqdm(total=nb_files)
        for midi in self.midi_files:
            self.load_new_midi(midi)
            self.stream_to_network()
            pbar.update(1)

    def get_net(self):
        """Getter for the network

        Returns:
            NetworkX: The main network
        """
        return self.net
    
    def get_sub_net(self):
        """Return the list of subnetworks

        Returns:
            List NetworkX: The list of subnetworks
        """
        if not self.layer:
            return self.net, nx.DiGraph()
        s_len = len(self.stream_list)
        self.sub_net =[]
        for i in range(s_len):
            def filter(node, layer=i): return self.net.nodes[node]["layer"]==layer # use default arg to avoid dependancy on i
            self.sub_net.append(nx.subgraph_view(self.net, filter_node=filter))
        # TODO make intergraph work
        def filter(node1,node2): return self.net[node1][node2]["inter"]
        self.intergraph = nx.subgraph_view(self.net, filter_edge=filter)#.to_undirected()
        return self.sub_net, self.intergraph
    
    def get_nodes_list(self, layer=0):
        return [self.build_node(elt) for elt in self.parsed_stream_list[layer]]
    
    def export_nodes_list(self, file, layer=0):
        """Export the list of nodes in the order they are played in the song

        Args:
            file (str): Name of the output file
            layer (int): index of the layer to export

        """
        lst = self.get_nodes_list(layer)
        open(file,"w").write("\n".join(lst))

    
    def list_to_string(self,my_list):
        return ','.join(str(x) for x in my_list)
    
    def group_notes_by_beat(self):
        stream_list_copy, self.stream_list = self.stream_list, []
        for stream in stream_list_copy:
            notes_and_rests = stream.flatten().notesAndRests
            new_stream = ms.stream.Stream()
            idx = 0
            first_beat = math.floor(notes_and_rests[0].offset)
            if len(notes_and_rests) == 0: continue
            last_beat = math.floor(max([elt.offset + elt.quarterLength for elt in notes_and_rests]))
            elts_in_beat = []
            loop = True
            for current_beat in range(first_beat, last_beat):
                elts_in_beat = [elt for elt in elts_in_beat if elt.offset + elt.quarterLength > current_beat]
                for i in range(idx, len(notes_and_rests)):
                    if notes_and_rests[i].offset >= current_beat + 1:
                        idx = i
                        break
                    elts_in_beat.append(notes_and_rests[i])
                pitches_in_beat = [pitch for note in elts_in_beat if not note.isRest for pitch in note.pitches]
                if pitches_in_beat:
                    new_note = ms.chord.Chord(pitches_in_beat)
                else:
                    new_note = ms.note.Rest()
                new_note.offset = current_beat
                new_stream.append(new_note)
                current_beat += 1
            self.stream_list.append(new_stream)

    

if __name__ == "__main__" :
    input_file_path = 'midis/test_pitches.mid'  # Replace with your MIDI file path
    output_folder = 'results/'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(use_gui=True, output_folder=output_folder, name="test")

    # Call createNet function
    net1.create_net()

    # Get the subnet and intergraph
    net1.get_sub_net()

    if net1.name is not None:
        output_filename = net1.name + '.graphml'
    # Derive the output filename from the input MIDI filename
    elif os.path.isdir(input_file_path):
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
    net1.export_sub_net(os.path.join(output_folder, name_without_extension) + os.path.sep, name_without_extension)
    net1.export_nodes_list("test.txt",0)