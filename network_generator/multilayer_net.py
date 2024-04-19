import music21 as ms
import networkx as nx
import parameter_picker as par_pick
from preset_params import get_preset_params
import os
from tqdm import tqdm
import math
import json
import copy

class MultiLayerNetwork:
    # TODO keep documentation up to date
    def __init__(self, use_gui=True, verbosity=1, preset_param=None, name="auto", **kwargs):
        """
        Class to create a network from midi files

        Args:
            use_gui (bool, optional) : Launch a small UI to choose the parameters of the network
            verbosity (int, optional): Amount of information printed when building the network
            preset_params(str, optional): Use some preset parameters to replicate networks of some papers:
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
            name(str, optional): Name of the network. Is used to generate the name of the output files when needed.\
                by default is on 'auto' which tries to figure out a name based on midi_file and/or preset parameter
            
            [**kwargs]: all network parameters(all optional with a default value)
            Node parameters: Booleans to know which values are hold by the nodes.
                - pitch(bool): Pitch of the note disregarding octave. Defaults to True.
                - octave(bool): Octave of the note. Defaults to False.
                - duration(bool): Duration of the note in quarter notes. Defaults to False.
                - offset(bool), offset_period(float): The offset of the note modulo the offset_period. Defaults to False, 1.
                - chromatic_interval(bool): Chromatic interval with the next note. Defaults to False.
                - diatonic_interval(bool): Diatonic interval with the next note. Defaults to False.
                - chord_function(bool): Give the corresponding degree of the chord

            Edge parameters: To know when two nodes are linked
                - strict_link(bool): Only link nodes from different layers when the corresponding notes play at the same time \
                    otherwise link nodes if the corresponding note play together at some point. Defaults to False.
                - max_link_time_diff(float): Maximal time difference in quarter notes between two consecutive notes \
                    (i.e. just separated by rests) for them to be linked. Defaults to 4.

            General structure parameters: Describe the general behaviour of the network
                - layer(bool): Create a network with a layer for every part. Defaults to True.
                - transpose(bool): Transpose the song in C major/ A minor depending on the mode based on the starting tonality. Defaults to False.
                - flatten(bool): Flatten the song and treat everything as one part.
                - enharmony(bool): Treat equivalent notes (e.g. C# and Db) as the same note. Defaults to True.
                - group_by_beat(bool): Group the notes of each part by beat and set durations to 1. Defaults to False.

            Input/Output parameters:
                - midi_files(list[str]): 
                  of midis to use.
                - outfolder(str): Output folder for all the graphs.
        """
        # Default Parameters
        params = {
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
            "split_chords":False,
            "duration_weighted_intergraph":True,
            "analyze_key":True,
            "midi_files":["midis/invent_bach/invent1.mid"],
            "outfolder":"results"
        }
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
        """dict{str:NetworkX}: All networks generated from each midi file by name"""

        self.separated_sub_nets = {}

        self.separated_intergraphs = {}

        self.aggregated_net = nx.DiGraph()
        """NetworkX: Agregated network from every midi file"""

        self.aggregated_sub_nets = []
        self.aggregated_intergraph = None

        self.nodes_lists = []

    def _auto_name(self, preset_param=None):
        name = ""
        if len(self.midi_files) == 1:
            name += os.path.splitext(os.path.basename(self.midi_files[0]))[0]
        if preset_param is not None:
            if name != "": name += "_"
            name += preset_param
  
        if name == "":
            return "no_name"
        return name


    def _print_if_useful(self, message, verbosity_level):
        if verbosity_level <= self.verbosity:
            print(message)

    def _get_flatten_stream(self, layer):
        if not self.split_chords:
            return self.parsed_stream_list[layer]
        return [elt for sub_lst in self.parsed_stream_list[layer] for elt in sub_lst]

    def _overwrite_params(self, current_params, **new_params):
        for key in current_params.keys():
            if key in new_params:
                current_params[key] = new_params[key]
        return current_params

    def _pick_parameters(self, params):
        return par_pick.get_parameters(params)

    def load_new_midi(self, midifilename):
        self.stream_list = []
        self.parsed_stream_list = []
        self.instruments = []
        self.midi_file = midifilename
        self._print_if_useful("Loading new midi : " + midifilename, 2)
        whole_piece = ms.converter.parse(midifilename, quarterLengthDivisors = (16,))
        if self.analyze_key:
            self.original_key = whole_piece.flatten().analyze('key')
            self._print_if_useful("analyzed key : " + str(self.original_key), 3)
            self.key = self.original_key
        if self.flatten:
            whole_piece = whole_piece.chordify()
            if self.transpose:
                self.stream_list.append(self._stream_to_C(whole_piece))
            else:
                self.stream_list.append(whole_piece)
        else:
            for part in whole_piece.parts: # loads each channel/instrument into stream list
                if self.transpose:
                    self.stream_list.append(self._stream_to_C(part))
                else:
                    self.stream_list.append(part)
            for elt in whole_piece.recurse():
                if 'Instrument' in elt.classes:
                    self.instruments.append(str(elt))
        if self.group_by_beat:
            self.group_notes_by_beat()
        self.parsed_stream_list = [self._build_parsed_list(part, i) for i,part in enumerate(self.stream_list)]

    def _parse_params(self, **params):
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
        if type(self.midi_files) != list:
            self.midi_files = [self.midi_files]
        self.duration_weighted_intergraph = params["duration_weighted_intergraph"]
        self.keep_extra = True # TODO keep supplementary info or not (for large dataset, you don't want to have this)
        self.split_chords = params["split_chords"]
        self.analyze_key = params["analyze_key"] or self.transpose or self.chord_function
        for file_name in self.midi_files:
            assert(os.path.splitext(file_name)[1] in [".mid", ".musicxml"])
        self.outfolder = params["outfolder"]
        if not self.outfolder.endswith(os.path.sep):
            self.outfolder += os.path.sep
    
    @property
    def interval(self): return self.diatonic_interval or self.chromatic_interval

    def _stream_to_C(self, part):
        if self.original_key.mode == "major":
            i = ms.interval.Interval(self.original_key.tonic, ms.pitch.Pitch('C'))
            self.key = ms.key.Key("C")
        elif self.original_key.mode == "minor":
            i = ms.interval.Interval(self.original_key.tonic, ms.pitch.Pitch('A'))
            self.key = ms.key.Key("a")
        else:
            assert(False)
        return part.transpose(i)

    def _is_ignored(self, elt):
        if not self.rest and elt.isRest:
            return True
        if self.interval and elt.isChord:
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
        infos["duration"] = elt.duration.quarterLength
        infos["offset"] = elt.offset - self.offset_period*math.floor(elt.offset/self.offset_period)
        infos["timestamp"] = elt.offset
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
        lst = [self.parse_elt(elt,i) for elt in part.flatten().notesAndRests if not self._is_ignored(elt)]
        if not lst : return lst
        def build_lst(elt): return elt if type(elt)==list else [elt]
        for i in range(len(lst)-1):
            if self.split_chords:
                for prev_elt in lst[i]:
                    for next_elt in lst[i+1]:
                        self._parse_interval(prev_elt, next_elt)
            else:
                self._parse_interval(lst[i], lst[i+1])
        if self.split_chords:
            for elt in lst[len(lst)-1]:
                self._parse_interval(elt)
        else:
            self._parse_interval(lst[len(lst)-1])
        return lst
    
    def _parse_interval(self, prev_elt, next_elt=None):
        if next_elt is None or prev_elt["rest"] or prev_elt["chord"] or next_elt["rest"] or next_elt["chord"]:
            prev_elt["chromatic_interval"] = 0
            prev_elt["diatonic_interval"] = 0
            return
        interval = ms.interval.Interval(ms.pitch.Pitch(prev_elt["pitch"]), ms.pitch.Pitch(next_elt["pitch"]))
        prev_elt["diatonic_interval"] = interval.diatonic.generic.value
        prev_elt["chromatic_interval"] = interval.chromatic.semitones

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
        s_len = len(self.stream_list)
        self._print_if_useful("[+] Creating network - Intra-layer processing", 2)

        for i in range(s_len):  # For each instrument
            self._process_intra_layer(i if self.layer else 0)
        if self.layer and s_len > 1:
            self._print_if_useful("[+] Creating network - Inter-layer processing", 2)
            self._process_inter_layer()
        return self.net

    def _process_intra_layer(self, layer, prev_elt=None):
        if self.split_chords: return self._process_intra_layer_splited(layer, prev_elt)
        prev_node = self.build_node(prev_elt) if prev_elt is not None else None
        for elt in self.parsed_stream_list[layer]:
            node = self.build_node(elt)
            self._add_or_update_node(node, elt)
            if prev_elt is not None:
                time_diff = elt["timestamp"] - prev_elt["timestamp"] - prev_elt["duration"]
                if time_diff <= self.max_link_time_diff:
                    self._add_or_update_edge(prev_node, node, inter=False)
            prev_node = node
            prev_elt = elt

    def _process_intra_layer_splited(self, layer, prev_elts=None):
        prev_nodes = [self.build_node(elt) for elt in prev_elts] if prev_elts is not None else None
        for elts in self.parsed_stream_list[layer]:
            nodes = [self.build_node(elt) for elt in elts]
            for node, elt in zip(nodes, elts):
                self._add_or_update_node(node, elt)
            if prev_elts is not None:
                time_diff = elts[0]["timestamp"] - prev_elts[0]["timestamp"] - prev_elts[0]["duration"]
                if time_diff <= self.max_link_time_diff:
                    for node in nodes:
                        for prev_node in prev_nodes:
                            self._add_or_update_edge(prev_node, node, inter=False)
            prev_nodes = nodes
            prev_elts = elts
    
    def _process_inter_layer(self):
        s_len = len(self.stream_list)
        all_nodes_infos = [elt for layer in range(len(self.stream_list)) for elt in self._get_flatten_stream(layer)]
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
                duration2 = all_nodes_infos[j]["duration"]

                node2 = self.build_node(all_nodes_infos[j])
                weight = min(timestamp+duration,timestamp2+duration2)-max(timestamp,timestamp2) if self.duration_weighted_intergraph else 1
                if layer != layer2:
                    # add undirected edge
                    self._add_or_update_edge(node, node2, inter=True, weight=weight)
                    self._add_or_update_edge(node2, node, inter=True, weight=weight)
                j += 1

    def _add_or_update_node(self, node, infos):
        if not self.net.has_node(node):
            def conditional_list(elt, elt_param):
                return elt if elt_param else [elt]
            self.net.add_node(node, 
                weight=1, 
                layer = conditional_list(infos["layer"], self.layer), 
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
        for node in net.nodes:
            for attribute in net.nodes[node].keys():
                if type(net.nodes[node][attribute]) == list :
                    net.nodes[node][attribute] = self.list_to_string(net.nodes[node][attribute])

    def _get_file_path(self, folder=None, filename=None):
        if folder is None:
            folder = self.outfolder
        if filename is None:
            filename = self.name
        try:
            os.mkdir(folder)
        except:
            pass
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
        self._print_if_useful("[+] Writing " + str(len(self.stream_list)) + " graphml subnet files to : " + folder, 1)
        for i in range(0,len(sub_nets)):
            cur_out = filepath + "l_" + str(i) + ".graphml"
            nx.write_graphml(sub_nets[i], cur_out)

    def _export_intergraph(self, intergraph, folder=None, filename=None):
        
        filepath,folder,filename = self._get_file_path(folder, filename)
        self._print_if_useful("[+] Writing graphml intergraph file to : " + folder, 1)
        cur_out = filepath + "_intergraph.graphml"
        nx.write_graphml(intergraph, cur_out)

    def create_net(self, separate_graphs=False):
        """Create the main network
            Args:
                - separate_graphs(bool): If set to True, create a net for every midi file, else create only one net where everything is aggregated
        """
        nb_files = len(self.midi_files)
        if nb_files <= 1: separate_graphs = False

        self._print_if_useful("[+] Converting " + str(nb_files) + " MIDI file(s) to network", 1)
        pbar = tqdm(total=nb_files)
        self.net = nx.DiGraph()
        self.separated_nets = {}
        for midi_file_path in self.midi_files:
            self.load_new_midi(midi_file_path)
            self._stream_to_network()
            if separate_graphs:
                name = os.path.splitext(os.path.basename(midi_file_path))[0]
                self.separated_nets[name] = self.net
                self.separated_sub_nets[name] = self._get_sub_nets(self.net)
                self.separated_intergraphs[name] = self._get_intergraph(self.net)
                self.net = nx.DiGraph()
            pbar.update(1)
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
        if not self.layer:
            return [net]
        s_len = len(self.stream_list)
        sub_nets =[]
        for i in range(s_len):
            def filter(node, layer=i): return net.nodes[node]["layer"]==layer # use default arg to avoid dependancy on i
            sub_nets.append(nx.subgraph_view(net, filter_node=filter))
        return sub_nets
    
    def _get_intergraph(self, net):
        """Build and return the intergraph of the current working network

        Returns:
            list[NetworkX]: The list of subnetworks
        """
        if not self.layer:
            return nx.DiGraph()
        def filter(node1,node2): return net[node1][node2]["inter"]
        return nx.subgraph_view(net, filter_edge=filter)
    
    def _get_nodes_list(self, layer=0):
        return [self.build_node(elt) for elt in self._get_flatten_stream(layer)]
    
    def export_nodes_list(self, file_path=None, layer=0):
        """Export the list of nodes in the order they are played in the song

        Args:
            file (str): Name of the output file
            layer (int): index of the layer to export

        """
        if file_path is None:
            file_path = self.default_filepath(".txt")
        lst = self._get_nodes_list(layer)
        open(file_path,"w").write("\n".join(lst))

    
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

    def export_nets(self, types=["main_net"]):
        """
        Export previously created nets.

        Args:
            list[str]: types of nets ('main_net','sub_net','intergraph') or the shortcut 'all' to pick them all
        """
        if types == "all":
            types = ["main_net","sub_net","intergraph"]

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

    

if __name__ == "__main__" :

    directory = "midis\\invent_bach\\"
    midi_files = [directory + f for f in os.listdir(directory)]
    output_folder = 'results'  # Replace with your desired output folder
    
    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = MultiLayerNetwork(use_gui=True, outfolder=output_folder, midi_files=midi_files)

    # Build net
    net1.create_net(separate_graphs=True)
    
    # Export nets ('all' = main nets + subnets + intergraphs)
    net1.export_nets(types='all')