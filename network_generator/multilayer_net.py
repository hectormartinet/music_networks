import music21 as ms
import networkx as nx
import numpy as np
import parameter_picker as par_pick
from preset_params import get_preset_params
from multiprocessing import Pool
import multi_timer
import os
from tqdm import tqdm
import math
import json
import copy

class MultiLayerNetwork:
    # TODO keep documentation up to date
    def __init__(self, use_gui=True, verbosity=1, preset_param=None, name="auto", **kwargs):
        """
        Class to create a network from music files (midi, xml, ...)

        Args:
            - use_gui (bool, optional) : Launch a small UI to choose the parameters of the network
            - verbosity (int, optional): Amount of information printed when building the network
            - preset_params(str, optional): Use some preset parameters to replicate networks of some papers:
                "liu"
                "ferreti"
                "gomez"
                "serra"
                "nardelli"
                "kulkarni"
                "rolla"
                "perkins"
                "frottier"
                "mrad_melodic"
                "mrad_harmonic"
            - name(str, optional): Name of the network. Is used to generate the name of the output files when needed.\
                by default is on 'auto' which tries to figure out a name based on music_file and/or preset parameter
            
            -[**kwargs]: all network parameters(all optional with a default value)
            - Node attributes: Booleans to know which values are hold by the nodes.
                - pitch(bool): Pitch of the note disregarding octave. Defaults to True.
                - octave(bool): Octave of the note. Defaults to False.
                - duration(bool): Duration of the note in quarter notes. Defaults to False.
                - offset(bool): The offset of the note modulo the beat_duration (float parameter). Defaults to False, 1.
                - chord_function(bool): Give the corresponding degree of the chord
            - Edge attributes: edge type information that will be hold by the nodes for graphs of order 2 or more
                - chromatic_interval(bool): Chromatic interval between two consecutive notes. Defaults to False.
                - diatonic_interval(bool): Diatonic interval between two consecutive notes. Defaults to False.
            
            - Edge parameters: To know when two nodes are linked
                - strict_link(bool): Only link nodes from different layers when the corresponding notes play at the same time \
                    otherwise link nodes if the corresponding note play together at some point. Defaults to False.
                - max_link_time_diff(float): Maximal time difference in quarter notes between two consecutive notes \
                    (i.e. just separated by rests) for them to be linked. Defaults to 4.

            - General structure parameters: Describe the general behaviour of the network
                - structure(str): Gives the struture of the outputed net. If the song has only one part, this parameter does not matter \
                    Here is each value possible for this parameter:
                    - 'multilayer': Process each part separately and output a net with as much layers as they are parts in the input file.
                    - 'monolayer': Process each part separately but aggregate the values in a net with one layer
                    - 'chordify': Flatten all the parts into one part
                - transpose(bool): Transpose the song in C major/ A minor depending on the mode based on the starting tonality. Defaults to False.
                - enharmony(bool): Treat equivalent notes (e.g. C# and Db) as the same note. Defaults to True.
                - chordify_by_beat(bool): For each beat, regroup all the notes of every part and replace it by a chord with the same notes (and a duration of 1). Defaults to False.
                - order(int): Number of consecutive notes contained in one node. For example C D E F with order=2 will give the nodes (and edges) : (C,D)->(D,E)->(E,F).
                - group_by_beat(bool): For each part, group the notes per beat. TODO detail
                - beat_duration(float): Duration of a beat in quarter notes. Used for the parameters offset, chordify_by_beat, group_by_beat.\
                    Should be set at any meaningful duration unit of the song. e.g. in 6/8 signature, 0.5, 1.5 and 3 are good, whereas in 4/4, you might want to set it to 1,2 or 4.

            - Input/Output parameters:
                - music_files(list[str]): list of files to use. Convert automatically to a list of size one if the input is not a list.
                - outfolder(str): Output folder for all the graphs.
        """
        # Default Parameters
        params = {
            "pitch":True,
            "octave":False,
            "duration":False,
            "rest":False,
            "offset":False,
            "beat_duration":1.,
            "enharmony":True,
            "diatonic_interval":False,
            "chromatic_interval":False,
            "chord_function":False,
            "transpose":False,
            "structure":"multilayer",
            "order":1,
            "strict_link":False,
            "max_link_time_diff":4.,
            "chordify_by_beat":False,
            "group_by_beat":False,
            "split_chords":False,
            "duration_weighted_intergraph":True,
            "analyze_key":False,
            "keep_extra":False,
            "music_files":["midis/invent_bach/invent1.mid"],
            "outfolder":"results"
        }
        if kwargs["music_files"] is not None and type(kwargs["music_files"]) != list:
            kwargs["music_files"] = [kwargs["music_files"]]
        if preset_param is not None:
            self._overwrite_params(params, **get_preset_params(preset_param))
        self._overwrite_params(params, **kwargs)
        if use_gui:
            params = self._pick_parameters(params)
        self._parse_params(**params)
        self.name = name 
        if self.name == "auto":
            self.auto_naming = True
            self.name = self._auto_name(preset_param)
        self.verbosity = verbosity
        self.net = nx.DiGraph()
        """NetworkX: Current working net"""

        self.separated_nets = {}
        """dict{str:NetworkX}: All networks generated from each file by name"""

        self.separated_sub_nets = {}

        self.separated_intergraphs = {}

        self.aggregated_net = nx.DiGraph()
        """NetworkX: Aggregated network from every file"""

        self.aggregated_sub_nets = []
        self.aggregated_intergraph = None

        self.last_visit_id = {}
        try:
            os.makedirs(self.outfolder)
        except:
            pass
        
        self.timer = multi_timer.MultiTimer()


    def _auto_name(self, preset_param=None):
        name = ""
        if len(self.music_files) == 1:
            name += os.path.splitext(os.path.basename(self.music_files[0]))[0]
        if preset_param is not None:
            if name != "": name += "_"
            name += preset_param
  
        if name == "":
            return "no_name"
        return name


    def _print_if_useful(self, message, verbosity_level):
        if verbosity_level <= self.verbosity:
            print(message)

    def stream_to_C(part, key):
        if key.mode == "major":
            i = ms.interval.Interval(key.tonic, ms.pitch.Pitch('C'))
            new_key = ms.key.Key("C")
        elif key.mode == "minor":
            i = ms.interval.Interval(key.tonic, ms.pitch.Pitch('A'))
            new_key = ms.key.Key("a")
        else:
            assert(False)
        return part.transpose(i), new_key

    def load_whole_piece(file_name, analyze_key=False, chordify=False, transpose=False):
        """
            Parse a song and do all 
        """
        whole_piece = ms.converter.parse(file_name, quarterLengthDivisors = (16,))
        original_key = None
        if analyze_key:
            original_key = whole_piece.flatten().analyze('key')
        key = original_key
        if chordify and len(whole_piece.parts)>1:
            whole_piece = whole_piece.chordify()
        if transpose:
            whole_piece, key = MultiLayerNetwork.stream_to_C(whole_piece, original_key)
        return whole_piece, key, original_key

    def _get_flatten_stream(self, layer):
        if not self.split_chords:
            return self.parsed_nodes[layer]
        return [elt for sub_lst in self.parsed_nodes[layer] for elt in sub_lst]

    def _overwrite_params(self, current_params, **new_params):
        for key in current_params.keys():
            if key in new_params:
                current_params[key] = new_params[key]
        return current_params

    def _pick_parameters(self, params):
        return par_pick.get_parameters(params)
        

    def load_new_file(self, file_name, whole_piece=None, key=None, original_key=None):
        self.stream_list = []
        self.parsed_nodes = []
        self.instruments = []
        self.original_key = original_key
        self.key = key
        self.music_file = file_name
        self._print_if_useful("Loading new file : " + file_name, 2)
        if whole_piece is None:
            self.timer.start("load file")
            whole_piece, self.key, self.original_key = MultiLayerNetwork.load_whole_piece(self.music_file, self.analyze_key, self.chordify, self.transpose)
            self.timer.end("load file")
        self.timer.start("parsing notes")
        if self.chordify:
            self.stream_list.append(whole_piece)
        else:
            for part in whole_piece.parts: # loads each channel/instrument into stream list
                self.stream_list.append(part.chordify())
            for elt in whole_piece.recurse():
                if 'Instrument' in elt.classes:
                    self.instruments.append(str(elt))
        if self.chordify_by_beat:
            self.chordify_notes_by_beat()
        self.parsed_nodes = [self._build_parsed_list(part, i) for i,part in enumerate(self.stream_list)]
        self.parsed_edges = [self._build_edges_list(parsed_nodes,i) for i,parsed_nodes in enumerate(self.parsed_nodes)]
        self.nodes_lists = [self._get_nodes_list(i) for i in range(self.nb_layers)]
        self.edges_lists = [self._get_edges_list(i) for i in range(self.nb_layers)]
        self.timer.end("parsing notes")

    def duration_to_nice_notation(self, duration, rest=False):
        def is_int(value, margin=0.01):
            return abs(round(value)-value) <= margin
        if duration == 0:
            assert(not rest)
            return chr(119188)
        if not is_int(np.log2(duration)):
            if not is_int(np.log2(duration/1.5)):
                return ("r" if rest else "") + str(duration)
            else:
                extension = "."
                value = round(2-np.log2(duration/1.5))
        else:
            value = round(2-np.log2(duration))
            extension = ""
        if rest:
            return chr(119099+value) + extension
        else:
            return chr(119133+value) + extension

    def _parse_params(self, **params):
        self.order = int(params["order"])
        assert(self.order >= 1)
        self.rest = params["rest"]
        self.pitch = params["pitch"]
        self.octave = params["octave"]
        self.enharmony = params["enharmony"]
        self.duration = params["duration"]
        self.offset = params["offset"]
        self.beat_duration = params["beat_duration"]
        assert(self.beat_duration > 0)
        self.transpose = params["transpose"]
        self.strict_link = params["strict_link"]
        self.max_link_time_diff = params["max_link_time_diff"]
        self.structure = params["structure"]
        self.diatonic_interval = params["diatonic_interval"] and not self.enharmony
        self.chromatic_interval = params["chromatic_interval"]
        self.chord_function = params["chord_function"]
        self.chordify_by_beat = params["chordify_by_beat"]
        self.group_by_beat = params["group_by_beat"]
        self.music_files = params["music_files"]
        if type(self.music_files) != list:
            self.music_files = [self.music_files]
        self.duration_weighted_intergraph = params["duration_weighted_intergraph"]
        self.keep_extra = params["keep_extra"] and self.order == 1
        self.split_chords = params["split_chords"] and self.order == 1
        self.analyze_key = params["analyze_key"] or self.transpose or self.chord_function
        for file_name in self.music_files:
            if not os.path.splitext(file_name)[1] in [".mid", ".musicxml", ".mxl", ".xml"]:
                raise Exception(f"{os.path.splitext(file_name)[1]} is not a correct file format")
        self.outfolder = params["outfolder"]
        if not self.outfolder.endswith(os.path.sep):
            self.outfolder += os.path.sep
    
    @property
    def interval(self): return self.diatonic_interval or self.chromatic_interval

    @property
    def nb_layers(self): return len(self.stream_list)

    @property
    def chordify(self): return self.structure == "chordify"

    @property
    def multilayer(self): return self.structure == "multilayer" and self.nb_layers > 1

    @property
    def interlayering(self): return self.multilayer and not self.group_by_beat and not self.split_chords

    def _is_ignored(self, elt):
        if not self.rest and elt.isRest:
            return True
        if isinstance(elt, ms.harmony.NoChord):
            return True
        return False
    
    def _parse_chord_function(self, elt):
        if not self.analyze_key or not elt.isChord:
            return "N/A"
        return ms.roman.romanNumeralFromChord(elt,self.key).romanNumeral

    def parse_elt(self, elt, layer=0):
        infos = {}
        infos["layer"] = layer
        infos["rest"] = elt.isRest
        infos["chord"] = elt.isChord
        infos["duration"] = float(elt.duration.quarterLength)
        infos["duration_nice_notation"] = self.duration_to_nice_notation(infos["duration"], infos["rest"])
        infos["offset"] = float(elt.offset % self.beat_duration)
        infos["timestamp"] = float(elt.offset)
        infos["pitch"] = self._parse_pitch(elt, octave=True)
        infos["pitch_class"] = self._parse_pitch(elt, octave=False)
        infos["chord_function"] = self._parse_chord_function(elt)
        if not self.split_chords:
            return infos
        if not elt.isChord:
            return [infos]
        info_lst = []
        for note in elt:
            info_copy = copy.deepcopy(infos)
            info_copy["pitch"] = self._parse_pitch(note, octave=True)
            info_copy["pitch_class"] = self._parse_pitch(note, octave=False)
            info_copy["chord"] = False
            info_lst.append(info_copy)
        return info_lst

    def _build_parsed_list(self, part, i):
        return [self.parse_elt(elt,i) for elt in part.flatten().notesAndRests if not self._is_ignored(elt)]
    
    def _build_edges_list(self, lst_nodes, i):
        if not self.split_chords:
            return [self._parse_interval(lst_nodes[i], lst_nodes[i+1]) for i in range(len(lst_nodes)-1)]
        lst_edges = []
        for i in range(len(lst_nodes)-1):
            if self.split_chords:
                for prev_elt in lst_nodes[i]:
                    for next_elt in lst_nodes[i+1]:
                        self._parse_interval(prev_elt, next_elt)
        return lst_edges

    def _get_high_note_pitch(self, elt):
        return ms.pitch.Pitch(elt["pitch"].split(" ")[-1])

    def _get_pitch_list(self, elt):
        return [ms.pitch.Pitch(pitch) for pitch in elt["pitch"].split(" ")]

    def _parse_interval(self, prev_elt=None, next_elt=None):
        elt = {}
        if prev_elt is None or next_elt is None or next_elt["rest"] or prev_elt["rest"]:
            elt["chromatic_interval"] = "N/A"
            elt["diatonic_interval"] = "N/A"
            return elt
        next_pitches = self._get_pitch_list(next_elt)
        prev_pitches = self._get_pitch_list(prev_elt)
        intervals = [ms.interval.Interval(prev_pitches[i], next_pitches[i]) for i in range(min(len(prev_pitches),len(next_pitches)))]
        # Use all intervals ?? (= remove the [0] from the two next lines)
        elt["diatonic_interval"] = [interval.diatonic.generic.value for interval in intervals][0]
        elt["chromatic_interval"] = [interval.chromatic.semitones for interval in intervals][0]
        return elt

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
        if self.multilayer:
            node["layer"] = infos["layer"]
        if self.chord_function:
            node["chord_function"] = infos["chord_function"]
        return json.dumps(node)
    
    def build_edge(self, infos):
        edge = {}
        if self.diatonic_interval:
            edge["diatonic_interval"] = infos["diatonic_interval"]
        if self.chromatic_interval:
            edge["chromatic_interval"] = infos["chromatic_interval"]
        return json.dumps(edge)
        
    def _parse_pitch(self, elt, octave):
        if elt.isNote:
            return self._pitch_to_str(elt.pitch, octave)
        if elt.isChord:
            unique_notes = list(set([self._pitch_to_str(pitch, octave) for pitch in elt.pitches]))
            unique_notes.sort(key=lambda elt : ms.pitch.Pitch(elt).midi)
            return " ".join(unique_notes)
        if elt.isRest:
            return "rest"
        assert(False)

    def _pitch_to_str(self, pitch, octave):
        if self.enharmony:
            pitch = ms.pitch.Pitch(pitch.midi)
        if octave:
            return str(pitch)
        else:
            return pitch.name 
    
    def _stream_to_network(self):
        self.timer.start("stream_to_network")
        self._print_if_useful("[+] Creating network - Intra-layer processing", 2)

        for i in range(self.nb_layers):  # For each instrument
            self._process_intra_layer(i)
        if self.interlayering:
            self._print_if_useful("[+] Creating network - Inter-layer processing", 2)
            self._process_inter_layer()
        self.timer.end("stream_to_network")
        return self.net

    def _process_intra_layer(self, layer, prev_node=None, end_time=None):
        self.last_visit_id = {}
        if self.split_chords: return self._process_intra_layer_splited(layer, prev_node, end_time)
        if self.group_by_beat: return self._process_intra_layer_grouped(layer, prev_node)
        for i in range(len(self.parsed_nodes[layer])+1-self.order):
            node_info = self.parsed_nodes[layer][i:i+self.order]
            edge_info = self.parsed_edges[layer][i:i+self.order-1]
            node = self.nodes_lists[layer][i] if self.order==1 else ",".join(self.nodes_lists[layer][i:i+self.order]+self.edges_lists[layer][i:i+self.order-1])
            self._add_or_update_node(node, node_info, layer, i, edge_info)
            if end_time is not None:
                time_diff = 0 if self.order>1 else node_info[0]["timestamp"] - end_time # TODO handle correctly when order > 2
                if time_diff <= self.max_link_time_diff: # TODO decide if the comparison should be strict (and change test accordingly)
                        self._add_or_update_edge(prev_node, node, inter=False)
            prev_node = node
            end_time = node_info[-1]["timestamp"] + node_info[-1]["duration"]
    
    def _process_intra_layer_grouped(self, layer, prev_node=None):
        time = math.floor(self.parsed_nodes[layer][0]["timestamp"])
        start_idx = 0
        while start_idx < len(self.parsed_nodes[layer]):
            end_idx = start_idx
            while end_idx < len(self.parsed_nodes[layer]) and self.parsed_nodes[layer][end_idx]["timestamp"] < time+self.beat_duration:
                end_idx += 1
            node_info = self.parsed_nodes[layer][start_idx:end_idx]
            edge_info = self.parsed_edges[layer][start_idx:end_idx-1]
            node = ",".join(self.nodes_lists[layer][start_idx:end_idx]+self.edges_lists[layer][start_idx:end_idx-1])
            if node=="":
                node = "empty"
            self._add_or_update_node(node, node_info, layer, edge_info=edge_info)
            if prev_node is not None:
                self._add_or_update_edge(prev_node, node, inter=False)
            prev_node = node
            start_idx = end_idx
            time += self.beat_duration

    def _process_intra_layer_splited(self, layer, prev_nodes=None, end_time=None):
        assert(self.order == 1)
        for elts in self.parsed_nodes[layer]:
            nodes = [self.build_node(elt) for elt in elts]
            for node, elt in zip(nodes, elts):
                self._add_or_update_node(node, [elt], layer)
            if end_time is not None:
                time_diff = elts[0]["timestamp"] - end_time
                if time_diff <= self.max_link_time_diff:
                    for node in nodes:
                        for prev_node in prev_nodes:
                            self._add_or_update_edge(prev_node, node, inter=False)
            prev_nodes = nodes
            end_time = elts[0]["timestamp"] + elts[0]["duration"]
    
    def _process_inter_layer(self):
        layer_and_idx = [(layer,idx)for layer in range(self.nb_layers) for idx in range(len(self.parsed_nodes[layer])-self.order+1)]
        layer_and_idx.sort(key=lambda x: self.parsed_nodes[x[0]][x[1]]["timestamp"])
        nb_notes = len(layer_and_idx)
        for i in range(nb_notes):
            layer, idx = layer_and_idx[i]
            nodes_info = self.parsed_nodes[layer][idx:idx + self.order]
            timestamp = nodes_info[0]["timestamp"]
            duration = sum([info["duration"] for info in nodes_info])
            node = self.nodes_lists[layer][idx] if self.order==1 else ",".join(self.nodes_lists[layer][idx:idx+self.order]+self.edges_lists[layer][idx:idx+self.order-1])
            for j in range(i+1,nb_notes):
                layer2, idx2 = layer_and_idx[j]
                nodes_info2 = self.parsed_nodes[layer2][idx2:idx2 + self.order]
                timestamp2 = nodes_info2[0]["timestamp"]
                if (self.strict_link and timestamp2 > timestamp) or timestamp2 >= timestamp + duration:
                    break
                duration2 = sum([info["duration"] for info in nodes_info2])
                node2 = self.nodes_lists[layer2][idx2] if self.order==1 else ",".join(self.nodes_lists[layer2][idx2:idx2+self.order]+self.edges_lists[layer2][idx2:idx2+self.order-1])
                weight = min(timestamp+duration,timestamp2+duration2)-max(timestamp,timestamp2) if self.duration_weighted_intergraph else 1
                if layer != layer2:
                    # add undirected edge
                    self._add_or_update_edge(node, node2, inter=True, weight=weight)
                    self._add_or_update_edge(node2, node, inter=True, weight=weight)
                j += 1

    def _add_or_update_node(self, node, node_info, layer, idx=None, edge_info=None):
        if edge_info is None:
            edge_info = {}
        total_duration = float(sum([info["duration"] for info in node_info]))
        if not self.net.has_node(node):
            self.net.add_node(
                node, 
                weight=1, 
                duration_weight=total_duration, 
                non_overlapping_weight=1, 
                non_overlapping_duration_weight=total_duration
            )
            if idx is not None: self.last_visit_id[node] = idx
            def add_attribute(param, param_used, elt=None, is_node_info=True):
                if not is_node_info and self.order==1 and not self.group_by_beat: return
                if elt is None:
                    if is_node_info:
                        elt = [info[param] for info in node_info]
                    else:
                        elt = [info[param] for info in edge_info]
                if type(elt)==list and len(elt)==1 and not self.group_by_beat:
                    elt = elt[0]
                if param_used:
                    self.net.nodes[node][param] = elt
                elif self.keep_extra:
                    self.net.nodes[node][param] = [elt]
            if self.order==1 or "layer" not in self.net.nodes[node]:
                add_attribute("layer", self.multilayer, elt=layer)
            add_attribute("pitch", self.pitch and self.octave)
            if not self.octave : add_attribute("pitch_class", self.pitch)
            add_attribute("duration", self.duration)
            add_attribute("duration_nice_notation", self.duration, "".join([info["duration_nice_notation"] for info in node_info]))
            add_attribute("offset", self.offset)
            add_attribute("timestamps", False, [info["timestamp"] for info in node_info])
            add_attribute("rest", self.rest)
            add_attribute("chord_function", self.chord_function)
            add_attribute("chromatic_interval", self.chromatic_interval, is_node_info=False)
            add_attribute("diatonic_interval", self.diatonic_interval, is_node_info=False)
        else :
            self.net.nodes[node]["weight"] += 1
            self.net.nodes[node]["duration_weight"] += total_duration
            if idx is None or (node not in self.last_visit_id or self.last_visit_id[node]+self.order <= idx): # no overlap condition
                self.net.nodes[node]["non_overlapping_weight"] += 1
                self.net.nodes[node]["non_overlapping_duration_weight"] += total_duration
                self.last_visit_id[node] = idx
            if self.keep_extra:
                def append_if_list(attribute, elt_to_add=None):
                    if type(self.net.nodes[node][attribute]) == list:
                        if elt_to_add is None:
                            self.net.nodes[node][attribute].append(node_info[attribute])
                        else:
                            self.net.nodes[node][attribute].append(elt_to_add)
                append_if_list("layer")
                append_if_list("pitch")
                append_if_list("duration",float(node_info["duration"]))
                append_if_list("offset", float(node_info["offset"]))
                append_if_list("rest")
                append_if_list("chord_function")
                self.net.nodes[node]["timestamps"].append(float(node_info["timestamp"]))
    
    def _add_or_update_edge(self, from_node, to_node, inter, weight=1):
        if from_node is None or to_node is None: return
        if self.net.has_edge(from_node, to_node):
            self.net[from_node][to_node]["weight"] += weight
        else:
            self.net.add_edge(from_node, to_node, weight=weight, inter=inter)

    def _prepare_for_export(self, net):
        """
        Necessary step before exporting graph.
        """
        if not self.keep_extra and not self.group_by_beat and self.order==1 :return
        for node in net.nodes:
            for attribute in net.nodes[node].keys():
                if type(net.nodes[node][attribute]) == list :
                    net.nodes[node][attribute] = self.list_to_string(net.nodes[node][attribute])

    def _get_file_path(self, folder=None, filename=None):
        if folder is None:
            folder = self.outfolder
        if filename is None:
            filename = self.name
        return folder + filename, folder, filename

    def _export_net(self, net, folder=None, filename=None):
        """Export the choosen network to a graphml file

        Args:
            folder (string): Output folder
            filename (string): Output filename
        """
        filepath,folder,filename = self._get_file_path(folder, filename)
        filepath += ".graphml"
        self._print_if_useful("[+] Writing main graphml file to : " + filepath, 1)
        nx.write_graphml(net, filepath)

    def _export_sub_nets(self, sub_nets, folder=None, filename=None):
        """Export the current working subnets to a graphml file

        Args:
            folder (string): Output folder
            filename (string): Output filename
        """
        filepath,folder,filename = self._get_file_path(folder, filename)
        self._print_if_useful("[+] Writing " + str(self.nb_layers) + " graphml subnet files to : " + folder, 1)
        for i in range(len(sub_nets)):
            cur_out = filepath + "l_" + str(i) + ".graphml"
            nx.write_graphml(sub_nets[i], cur_out)

    def _export_intergraph(self, intergraph, folder=None, filename=None):
        
        filepath,folder,filename = self._get_file_path(folder, filename)
        self._print_if_useful("[+] Writing graphml intergraph file to : " + folder, 1)
        cur_out = filepath + "_intergraph.graphml"
        nx.write_graphml(intergraph, cur_out)

    def _process_net(self, file_path, separate_graphs, output_txt, whole_piece=None, key=None, original_key=None):
        self.load_new_file(file_path, whole_piece, key, original_key)
        self._stream_to_network()
        name = os.path.splitext(os.path.basename(file_path))[0]
        if output_txt:
            for i in range(self.nb_layers):
                self.export_nodes_list(filename=name, layer=i)
        if separate_graphs:
            self.separated_nets[name] = self.net
            self.separated_sub_nets[name] = self._get_sub_nets(self.net)
            self.separated_intergraphs[name] = self._get_intergraph(self.net)

    def create_net(self, separate_graphs=False, output_txt=True, parallel=False, pool_size=100):
        """Create the main network
            Args:
                - separate_graphs(bool): If set to True, create a net for every music file, else create only one net where everything is aggregated
                - output_txt(bool): If set to True, output the nodes played in order for each layer of each net
                - parallel(bool): If set to True, load the different files in parallel. Recommended only for big files as it could slow down the process for smaller ones.
                - pool_size(int): Number of files loaded at the same time. Is ignored if parallel is False.
        """
        nb_files = len(self.music_files)
        if nb_files <= 1: separate_graphs = False

        self._print_if_useful("[+] Converting " + str(nb_files) + " file(s) to network", 1)
        pbar = tqdm(total=nb_files)
        self.net = nx.DiGraph()
        self.separated_nets = {}
        if parallel:
            for pool_idx in range(math.ceil(len(self.music_files)/pool_size)):
                low_idx = pool_idx*pool_size
                high_idx = min((pool_idx+1)*pool_size, len(self.music_files))
                pool_input = [(file, self.analyze_key, self.chordify, self.transpose) for file in self.music_files[low_idx:high_idx]]
                self.timer.start("load file")
                with Pool() as p:
                    parsed_pieces = p.starmap(MultiLayerNetwork.load_whole_piece, pool_input)
                self.timer.end("load file")
                for piece, file_path in zip(parsed_pieces, self.music_files[low_idx:high_idx]):
                    self._process_net(file_path, separate_graphs, output_txt, piece[0], piece[1], piece[2])
                    pbar.update(1)
        else:
            for file_path in self.music_files:
                self._process_net(file_path, separate_graphs, output_txt)
                pbar.update(1)
                if separate_graphs:
                    self.net = nx.DiGraph()
        if not separate_graphs:
            self.aggregated_net = self.net
            self.aggregated_sub_nets = self._get_sub_nets(self.net)
            self.aggregated_intergraph = self._get_intergraph(self.net)


    def get_aggregated_net(self):
        """Getter for the aggregated network

        Returns:
            NetworkX: The main network
        """
        return self.aggregated_net
    
    def get_separate_nets(self):
        """Getter for the separated networks

        Returns:
            dict{str: NetworkX}: all networks by name
        """
        return self.separated_nets
    
    def get_net(self):
        """Getter for the current working net

        Returns:
            NetworkX: Current working net
        """
        return self.net

    def _get_sub_nets(self, net):
        """Return the list of subnetworks from the input net

        Returns:
            list[NetworkX]: The list of subnetworks
        """
        if not self.multilayer:
            return [net]
        sub_nets =[]
        for i in range(self.nb_layers):
            def filter(node, layer=i): return net.nodes[node]["layer"]==layer # use default arg to avoid dependancy on i
            sub_nets.append(nx.subgraph_view(net, filter_node=filter))
        return sub_nets
    
    def _get_intergraph(self, net):
        """Build and return the intergraph of the input net

        Returns:
            list[NetworkX]: The list of subnetworks
        """
        if not self.multilayer:
            return nx.DiGraph()
        def filter(node1,node2): return net[node1][node2]["inter"]
        return nx.subgraph_view(net, filter_edge=filter)
    
    def _get_nodes_list(self, layer=0):
        if self.split_chords:
            return [[self.build_node(elt) for elt in elt_list] for elt_list in self.parsed_nodes[layer]]
        return [self.build_node(elt) for elt in self.parsed_nodes[layer]]
    
    def _get_edges_list(self, layer=0):
        if self.split_chords:
            return [[self.build_edge(elt) for elt in elt_list] for elt_list in self.parsed_edges[layer]]
        return [self.build_edge(elt) for elt in self.parsed_edges[layer]]

    def export_nodes_list(self, folder=None, filename=None, layer=0):
        """Export the list of nodes in the order they are played in the song

        Args:
            file (str): Name of the output file
            layer (int): index of the layer to export

        """
        file_path, _, _ = self._get_file_path(folder, filename)
        if self.nb_layers > 1:
            file_path += f"_nodesl_{layer}.txt"
        else:
            file_path +="_nodes.txt"
        if self.split_chords:
            open(file_path,"w").write("\n".join([str(elt) for elt in self.nodes_lists[layer]]))
        elif self.order==1:
            open(file_path,"w").write("\n".join(self.nodes_lists[layer]))
        else:
            open(file_path,"w").write("\n".join([",".join(self.nodes_lists[layer][i:i+self.order]) for i in range(len(self.nodes_lists[layer])+1-self.order)]))

    
    def list_to_string(self,my_list):
        return ','.join(str(x) for x in my_list)
    
    def chordify_notes_by_beat(self):
        stream_list_copy, self.stream_list = self.stream_list, []
        for stream in stream_list_copy:
            notes_and_rests = stream.flatten().notesAndRests
            new_stream = ms.stream.Stream()
            starting_idx = 0
            first_timestamp = math.floor(notes_and_rests[0].offset)
            if len(notes_and_rests) == 0: continue
            last_timestamp = math.floor(max([elt.offset + elt.quarterLength for elt in notes_and_rests]))
            elts_in_beat = []
            loop = True
            current_timestamp = first_timestamp
            while current_timestamp < last_timestamp:
                print(current_timestamp)
                # Get elements from last beat that are still playing in the current one
                elts_in_beat = [elt for elt in elts_in_beat if elt.offset + elt.quarterLength > current_timestamp]
                # Add elements which starts in the current beat
                for i in range(starting_idx, len(notes_and_rests)):
                    if notes_and_rests[i].offset >= current_timestamp + self.beat_duration:
                        starting_idx = i
                        break
                    elts_in_beat.append(notes_and_rests[i])
                pitches_in_beat = [pitch for note in elts_in_beat if not note.isRest for pitch in note.pitches]
                if pitches_in_beat:
                    new_note = ms.chord.Chord(pitches_in_beat)
                else:
                    new_note = ms.note.Rest()
                new_note.offset = current_timestamp
                new_stream.append(new_note)
                current_timestamp += self.beat_duration
            self.stream_list.append(new_stream)

    def export_nets(self, types=["main_net","sub_net"]):
        """
        Export previously created nets.

        Args:
            list[str]: types of nets ('main_net','sub_net','intergraph') or the shortcut 'all' to pick them all
                - main_net: The net in its entirety
                - sub_net: Layers of the main net. As many sub_nets as the number of layers. Is not exported if there is only one layer (would be the same as the main net)
                - intergraph: Graph of the interlayer edges between the layers. Only one intergraph per net. Is not exported if there is only one layer (would be empty)
        """
        self.timer.start("exporting nets")
        if types == "all":
            types = ["main_net","sub_net","intergraph"]
        
        if not self.multilayer:
            types = ["main_net"]

        if self.aggregated_net.number_of_nodes() > 0:
            self._prepare_for_export(self.aggregated_net)
            if "main_net" in types:
                self._export_net(self.aggregated_net)
            if "sub_net" in types:
                self._export_sub_nets(self.aggregated_sub_nets)
            if "intergraph" in types:
                self._export_intergraph(self.aggregated_intergraph)
        
        for name in self.separated_nets.keys():
            self._prepare_for_export(self.separated_nets[name])
            if "main_net" in types:
                self._export_net(self.separated_nets[name], filename=name)
            if "sub_net" in types:
                self._export_sub_nets(self.separated_sub_nets[name], filename=name)
            if "intergraph" in types:
                self._export_intergraph(self.separated_intergraphs[name], filename=name)
        self.timer.end("exporting nets")
        

    

if __name__ == "__main__" :

    directory = "midis\\invent_bach\\"
    music_files = [directory + f for f in os.listdir(directory)]
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net = MultiLayerNetwork(use_gui=True, outfolder=output_folder, music_files=music_files, preset_param="liu")

    # Build net
    net.create_net(separate_graphs=True, output_txt=True, parallel=False)
    
    # Export nets ('all' = main nets + subnets + intergraphs)
    net.export_nets()

    # Output processing times
    net.timer.print_times()
