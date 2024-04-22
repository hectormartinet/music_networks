from multilayer_net import *
import networkx as nx
import music21 as ms
import os

class MultilayerNetworkTester:
    
    def __init__(self):
        """
        Class to test the MultilayerNetwork class
        """
        self.test_folder = "tests_temp/"
        self.current_test = "None"

    def assert_graph_match(self, graph, exp_graph):
        """
        Check that two graphs (graph, exp_graph) match except that we ignore extra attributes from graph
        """
        if graph.number_of_nodes() != exp_graph.number_of_nodes():
            raise Exception(f"The two graphs don't have the same number of nodes: found {graph.number_of_nodes()}, expected {exp_graph.number_of_nodes()}")
        if graph.number_of_edges() != exp_graph.number_of_edges():
            raise Exception(f"The two graphs don't have the same number of edges: found {graph.number_of_edges()}, expected {exp_graph.number_of_edges()}")
        for node, data in exp_graph.nodes(data=True):
            if not graph.has_node(node):
                raise Exception(f"The graph does not have the node : {node}")
            for key, value in data.items():
                if not key in graph.nodes[node]:
                    raise Exception(f"The node {node} does not have the attribute {key}")
                if graph.nodes[node][key] != value:
                    raise Exception(f"Expected value for node {node} on attribute {key} : {value}. Value found : {graph.nodes[node][key]}")
        for node1, node2, data in exp_graph.edges(data=True):
            if not graph.has_edge(node1, node2):
                raise Exception(f"The graph does not have the edge : {(node1,node2)}")
            for key, value in data.items():
                if graph[node1][node2][key] != value:
                    raise Exception(f"Expected value for edge {node1,node2}  on attribute {key} : {value}. Value found : {graph[node1][node2][key]}")

    def run_unit_tests(self):
        self.make_test_folder()
        tests = [
            self.test_enharmony,
            self.test_chord_function,
            self.test_group_by_beat,
            self.test_pitch,
            self.test_octave,
            self.test_duration,
            self.test_rest,
            self.test_offset,
            self.test_transpose,
            self.test_max_link_time_diff,
            self.test_split_chords,
            self.test_flatten,
            self.test_layer,
            self.test_strict_link,
            self.test_analyze_key,
            self.test_chromatic_interval,
            self.test_diatonic_interval,
            self.test_duration_weighted_intergraph,
            self.test_max_link_time_diff,
        ]
        nb_test_failed = 0
        for test in tests:
            try:
                test()
            except Exception as err:
                print(f"Unit tests '{self.current_test}' failed, error :")
                print(err)
                nb_test_failed += 1
        if nb_test_failed == 0:
            print(f"All {len(tests)} unit tests passed successfully")
        else:
            print(f"{len(tests) - nb_test_failed}/{len(tests)} test passed.")
        self.remove_test_folder() 

    def test_run(self, **params):
        """
        Run a net build (and some other functions) to test if it doesn't crash
        """

        net = MultiLayerNetwork(use_gui=False,verbosity=0,**params)
        net.create_net()
        net.get_net()
        net._get_sub_nets()
        net._get_nodes_list()
        net1.convert_attributes_to_str()
    
    def make_test_folder(self):
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

    def remove_test_folder(self):
        for file in os.listdir(self.test_folder):
            path = self.test_folder + file
            os.remove(path)
        os.removedirs(self.test_folder)

    def test_enharmony(self, **params):
        """
        Test enharmony parameter: 
            - If enharmony is True: E##, F# end G- should be treated as the same node
            - If enharmony is False: E##, F# end G- should be treated as 3 different nodes
            - musicxml file is used because midi does not handle enharmonics
        """
        self.current_test = "enharmony"

        # Create musicxml file
        stream = ms.converter.parse("tinyNotation: E## F# G-")
        file_path = self.test_folder + self.current_test + ".musicxml"
        stream.write("musicxml",file_path)

        # With enharmony
        self.current_test = "enharmony(True)"
        
        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, enharmony=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        exp_graph = nx.DiGraph()
        note = ms.note.Note("F#")
        node = net.build_node(net.parse_elt(note))
        exp_graph.add_node(node, weight=3)
        exp_graph.add_edge(node, node, weight=2)
        self.assert_graph_match(graph, exp_graph)

        # Without enharmony
        self.current_test = "enharmony(False)"

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, enharmony=False)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        exp_graph = nx.DiGraph()
        notes =[ms.note.Note("E##"), ms.note.Note("F#"), ms.note.Note("G-")]
        nodes = [net.build_node(net.parse_elt(note)) for note in notes]
        for node in nodes:
            exp_graph.add_node(node, weight=1)
        for i in range(2):
            exp_graph.add_edge(nodes[i], nodes[i+1], weight=1)
        self.assert_graph_match(graph, exp_graph)
        
    def test_chord_function(self):
        """
        Test chord_function parameter (set to True):
            - Chord inversions of the same chord should be treated as the same node (C3 E3 G3 = E3 G3 C4)
            - Checking degrees of some chords (I,V)
        """
        self.current_test = "chord_function"

        # Create midi file
        stream = ms.stream.Stream()
        chord1 = ms.chord.Chord("C3 E3 G3")
        chord2 = ms.chord.Chord("D2 G3 B3")
        chord3 = ms.chord.Chord("E3 G3 C4")
        stream.append(chord1)
        stream.append(chord2)
        stream.append(chord3)
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi",file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, pitch=False, chord_function=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        exp_graph = nx.DiGraph()
        node1 = net.build_node(net.parse_elt(chord1))
        node2 = net.build_node(net.parse_elt(chord2))
        exp_graph.add_node(node1, weight=2, chord_function="I")
        exp_graph.add_node(node2, weight=1, chord_function="V")
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node1, weight=1)
        self.assert_graph_match(graph, exp_graph)
    
    def test_group_by_beat(self):
        """
        Test group_by_beat parameter:
            - Notes from the same beat should end up in the same chord
            - Within a beat, different order of notes should give the same node
        """
        self.current_test = "group_by_beat"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: c8 d8 e4 d8 c8")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, group_by_beat=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        chord = ms.chord.Chord("C D")
        note = ms.note.Note("E")
        node1 = net.build_node(net.parse_elt(chord))
        node2 = net.build_node(net.parse_elt(note))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2)
        exp_graph.add_node(node2, weight=1)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node1, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_pitch(self):
        """
        Test pitch parameter:
            - Notes with same pitch but different octaves, duration... should give the same node
        """
        self.current_test = "pitch"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: c8 C4 E8 e2")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("E")
        node1 = net.build_node(net.parse_elt(note1))
        node2 = net.build_node(net.parse_elt(note2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C")
        exp_graph.add_node(node2, weight=2, pitch_class="E")
        exp_graph.add_edge(node1, node1, weight=1)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node2, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_octave(self):
        """
        Test octave parameter:
            - Notes with same pitch but different octaves should give different nodes
        """
        self.current_test = "octave"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: C8 c4 c8")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, octave=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C3")
        note2 = ms.note.Note("C4")
        node1 = net.build_node(net.parse_elt(note1))
        node2 = net.build_node(net.parse_elt(note2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=1, pitch_class="C", pitch="C3")
        exp_graph.add_node(node2, weight=2, pitch_class="C", pitch="C4")
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node2, weight=1)
        self.assert_graph_match(graph, exp_graph)
    
    def test_duration(self):
        """
        Test duration parameter:
            - Notes with different duration should give different nodes
            - (TODO) test tuplets (weird behaviour with triplets)
        """
        self.current_test = "duration"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: C8 D8 E4")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, duration=True, pitch=False)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note(quarterLength=0.5)
        note2 = ms.note.Note(quarterLength=1)
        node1 = net.build_node(net.parse_elt(note1))
        node2 = net.build_node(net.parse_elt(note2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, duration=0.5)
        exp_graph.add_node(node2, weight=1, duration=1)
        exp_graph.add_edge(node1, node1, weight=1)
        exp_graph.add_edge(node1, node2, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_rest(self):
        """
        Test rest parameter:
            - Rests should be taken into account iff the 'rest' parameter is True
        """
        self.current_test = "rest"

        # Create midi file
        stream = ms.converter.parse("tinyNotation:3/4 C r C")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # With rests
        self.current_test = "rest(True)"

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, rest=True)
        net.create_net()
        graph = net.get_net()
        
        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Rest()
        node1 = net.build_node(net.parse_elt(note1))
        node2 = net.build_node(net.parse_elt(note2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C", rest=False)
        exp_graph.add_node(node2, weight=1, pitch_class="rest", rest=True)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node1, weight=1)
        self.assert_graph_match(graph, exp_graph)

        # Without rests
        self.current_test = "rest(False)"

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, rest=False)
        net.create_net()
        graph = net.get_net()
        
        # Create expected graph
        note1 = ms.note.Note("C")
        node1 = net.build_node(net.parse_elt(note1))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C")
        exp_graph.add_edge(node1, node1, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_offset(self):
        """
        Test offset and offset_period parameters:
            - Notes with the same offset should give the same node
        """
        self.current_test = "offset"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: C8 D8 C4 D4")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, offset=True, offset_period=1)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("D")
        parsed_elt1 = net.parse_elt(note1)
        parsed_elt1["offset"] = 0.
        parsed_elt2 = net.parse_elt(note2)
        parsed_elt2["offset"] = 0.5
        parsed_elt3 = net.parse_elt(note2)
        parsed_elt3 ["offset"] = 0.
        node1 = net.build_node(parsed_elt1)
        node2 = net.build_node(parsed_elt2)
        node3 = net.build_node(parsed_elt3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C", offset=0.)
        exp_graph.add_node(node2, weight=1, pitch_class="D", offset=0.5)
        exp_graph.add_node(node3, weight=1, pitch_class="D", offset=0.)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node1, weight=1)
        exp_graph.add_edge(node1, node3, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_transpose(self):
        """
        Test transpose parameter:
            - Everything should be transposed in C Major/ A minor depending on the mode
            - Test with G major
        """
        self.current_test = "transpose"

        # Create midi file
        stream = ms.stream.Stream()
        chord1 = ms.chord.Chord("G B D")
        chord2 = ms.chord.Chord("F# A D")
        chord3 = ms.chord.Chord("G B D") # same as the first one but need to duplicate for music21
        stream.append(chord1)
        stream.append(chord2)
        stream.append(chord3)
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, transpose=True)
        net.create_net()
        if net.original_key != ms.key.Key("G",mode="major"):
            raise Exception(f"Key analysis failed, expected tonality : G major, found : {net.original_key}")
        graph = net.get_net()

        # Create expected graph
        chord1 = ms.chord.Chord("C E G")
        chord2 = ms.chord.Chord("B D G")
        node1 = net.build_node(net.parse_elt(chord1))
        node2 = net.build_node(net.parse_elt(chord2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C E G")
        exp_graph.add_node(node2, weight=1, pitch_class="D G B")
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node1, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_strict_link(self):
        """
        Test strict link parameter:
            - When strict_link is True, two nodes from different layers link iff the nodes play exactly at the same time.
        """
        self.current_test = "strict_link"

        # Create midi file, tinyNotation does not work when using multiple parts
        part1 = ms.stream.Part()
        part1.append(ms.note.Note("E"))
        part1.append(ms.note.Note("C"))
        part2 = ms.stream.Part()
        part2.append(ms.note.Note("C", quarterLength=0.5))
        part2.append(ms.note.Note("D", quarterLength=0.5))
        part2.append(ms.note.Note("E"))
        stream = ms.stream.Stream([part1, part2])
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, layer=True, duration_weighted_intergraph=False, strict_link=True)
        net.create_net()
        graph = net._get_intergraph(net.net)

        # Create expected graph
        parsed_elt0_1 = net.parse_elt(ms.note.Note("E"))
        parsed_elt0_1["layer"] = 0
        parsed_elt0_2 = net.parse_elt(ms.note.Note("C"))
        parsed_elt0_2["layer"] = 0
        parsed_elt1_1 = net.parse_elt(ms.note.Note("C"))
        parsed_elt1_1["layer"] = 1
        parsed_elt1_2 = net.parse_elt(ms.note.Note("D"))
        parsed_elt1_2["layer"] = 1
        parsed_elt1_3 = net.parse_elt(ms.note.Note("E"))
        parsed_elt1_3["layer"] = 1
        node0_1 = net.build_node(parsed_elt0_1)
        node0_2 = net.build_node(parsed_elt0_2)
        node1_1 = net.build_node(parsed_elt1_1)
        node1_2 = net.build_node(parsed_elt1_2)
        node1_3 = net.build_node(parsed_elt1_3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node0_1, weight=1, pitch_class="E")
        exp_graph.add_node(node0_2, weight=1, pitch_class="C")
        exp_graph.add_node(node1_1, weight=1, pitch_class="C")
        exp_graph.add_node(node1_2, weight=1, pitch_class="D")
        exp_graph.add_node(node1_3, weight=1, pitch_class="E")
        exp_graph.add_edge(node0_1, node1_1, weight=1)
        exp_graph.add_edge(node1_1, node0_1, weight=1)
        exp_graph.add_edge(node0_2, node1_3, weight=1)
        exp_graph.add_edge(node1_3, node0_2, weight=1)

        self.assert_graph_match(graph, exp_graph)

    def test_max_link_time_diff(self):
        """
        Test max link time diff parameter:
            - Should not connect nodes that are too far appart
        """
        self.current_test = "max_link_time_diff"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: C1 r1 C2 r2 D1")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, max_link_time_diff=2)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("D")
        node1 = net.build_node(net.parse_elt(note1))
        node2 = net.build_node(net.parse_elt(note2))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=2, pitch_class="C")
        exp_graph.add_node(node2, weight=1, pitch_class="D")
        exp_graph.add_edge(node1, node2, weight=1)
        self.assert_graph_match(graph, exp_graph)

    def test_layer(self):
        self.current_test = "layer"
        
        # Create midi file, tinyNotation does not work when using multiple parts
        part1 = ms.stream.Part()
        part1.append(ms.note.Note("E"))
        part1.append(ms.note.Note("C"))
        part2 = ms.stream.Part()
        part2.append(ms.note.Note("C", quarterLength=0.5))
        part2.append(ms.note.Note("D", quarterLength=0.5))
        part2.append(ms.note.Note("E"))
        stream = ms.stream.Stream([part1, part2])
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, layer=True, duration_weighted_intergraph = False)
        net.create_net()
        graph = net._get_intergraph(net.net)

        # Create expected graph
        parsed_elt0_1 = net.parse_elt(ms.note.Note("E"))
        parsed_elt0_1["layer"] = 0
        parsed_elt0_2 = net.parse_elt(ms.note.Note("C"))
        parsed_elt0_2["layer"] = 0
        parsed_elt1_1 = net.parse_elt(ms.note.Note("C"))
        parsed_elt1_1["layer"] = 1
        parsed_elt1_2 = net.parse_elt(ms.note.Note("D"))
        parsed_elt1_2["layer"] = 1
        parsed_elt1_3 = net.parse_elt(ms.note.Note("E"))
        parsed_elt1_3["layer"] = 1
        node0_1 = net.build_node(parsed_elt0_1)
        node0_2 = net.build_node(parsed_elt0_2)
        node1_1 = net.build_node(parsed_elt1_1)
        node1_2 = net.build_node(parsed_elt1_2)
        node1_3 = net.build_node(parsed_elt1_3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node0_1, weight=1, pitch_class="E")
        exp_graph.add_node(node0_2, weight=1, pitch_class="C")
        exp_graph.add_node(node1_1, weight=1, pitch_class="C")
        exp_graph.add_node(node1_2, weight=1, pitch_class="D")
        exp_graph.add_node(node1_3, weight=1, pitch_class="E")
        exp_graph.add_edge(node0_1, node1_1, weight=1)
        exp_graph.add_edge(node1_1, node0_1, weight=1)
        exp_graph.add_edge(node0_1, node1_2, weight=1)
        exp_graph.add_edge(node1_2, node0_1, weight=1)
        exp_graph.add_edge(node0_2, node1_3, weight=1)
        exp_graph.add_edge(node1_3, node0_2, weight=1)

        self.assert_graph_match(graph, exp_graph)
    
    def test_flatten(self):
        self.current_test = "flatten"

        # Create midi file, tinyNotation does not work when using multiple parts
        part1 = ms.stream.Part()
        part1.append(ms.note.Note("C"))
        part1.append(ms.note.Note("B"))
        part2 = ms.stream.Part()
        part2.append(ms.note.Note("E", quarterLength=0.5))
        part2.append(ms.note.Note("F#"))
        part2.append(ms.note.Note("G", quarterLength=0.5))
        stream = ms.stream.Stream([part1, part2])
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, layer=False, flatten=True, duration=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        chord1 = ms.chord.Chord("C E", quarterLength=0.5)
        chord2 = ms.chord.Chord("C F#", quarterLength=0.5)
        chord3 = ms.chord.Chord("F# B", quarterLength=0.5)
        chord4 = ms.chord.Chord("G B", quarterLength=0.5)
        node1 = net.build_node(net.parse_elt(chord1))
        node2 = net.build_node(net.parse_elt(chord2))
        node3 = net.build_node(net.parse_elt(chord3))
        node4 = net.build_node(net.parse_elt(chord4))
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=1, pitch_class="C E", duration=0.5)
        exp_graph.add_node(node2, weight=1, pitch_class="C F#", duration=0.5)
        exp_graph.add_node(node3, weight=1, pitch_class="F# B", duration=0.5)
        exp_graph.add_node(node4, weight=1, pitch_class="G B", duration=0.5)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node3, weight=1)
        exp_graph.add_edge(node3, node4, weight=1)

        self.assert_graph_match(graph, exp_graph)

    
    def test_diatonic_interval(self):
        """
        Test diatonic_interval parameter:
            - Diatonic intervals disregards accidentals, so the enharmonic version of C major : C D## G should not give the same result
            - Descending intervals should have negative value
        """
        self.current_test = "diatonic_interval"

        # Create midi file
        stream = ms.converter.parse("tinyNotation: C E## D-")
        file_path = self.test_folder + self.current_test + ".musicxml"
        stream.write("musicxml", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, diatonic_interval=True, enharmony=False)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("E##")
        note3 = ms.note.Note("D-")
        parsed_elt1 = net.parse_elt(note1)
        parsed_elt1["diatonic_interval"] = 3
        parsed_elt2 = net.parse_elt(note2)
        parsed_elt2["diatonic_interval"] = -2
        parsed_elt3 = net.parse_elt(note3)
        parsed_elt3["diatonic_interval"] = 0
        node1 = net.build_node(parsed_elt1)
        node2 = net.build_node(parsed_elt2)
        node3 = net.build_node(parsed_elt3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=1, pitch_class="C", diatonic_interval = 3)
        exp_graph.add_node(node2, weight=1, pitch_class="E##", diatonic_interval = -2)
        exp_graph.add_node(node3, weight=1, pitch_class="D-", diatonic_interval = 0)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node3, weight=1)
        self.assert_graph_match(graph, exp_graph)


    def test_chromatic_interval(self):
        self.current_test = "chromatic_interval"
        # Create midi file
        stream = ms.converter.parse("tinyNotation: C E E-")
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, chromatic_interval=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("E")
        note3 = ms.note.Note("E-")
        parsed_elt1 = net.parse_elt(note1)
        parsed_elt1["chromatic_interval"] = 4
        parsed_elt2 = net.parse_elt(note2)
        parsed_elt2["chromatic_interval"] = -1
        parsed_elt3 = net.parse_elt(note3)
        parsed_elt3["chromatic_interval"] = 0
        node1 = net.build_node(parsed_elt1)
        node2 = net.build_node(parsed_elt2)
        node3 = net.build_node(parsed_elt3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=1, pitch_class="C", chromatic_interval = 4)
        exp_graph.add_node(node2, weight=1, pitch_class="E", chromatic_interval = -1)
        exp_graph.add_node(node3, weight=1, pitch_class="E-", chromatic_interval = 0)
        exp_graph.add_edge(node1, node2, weight=1)
        exp_graph.add_edge(node2, node3, weight=1)
        self.assert_graph_match(graph, exp_graph)
    
    def test_split_chords(self):
        """
        Test split chord parameter:
            - Chords should be splited into individual notes
            - Each note from a chord is linked to all the notes of the next chord
        """
        self.current_test = "split_chords"

        # Create midi file
        stream = ms.stream.Stream()
        chord1 = ms.chord.Chord("C D")
        chord2 = ms.chord.Chord("E F")
        stream.append(chord1)
        stream.append(chord2)
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, split_chords=True)
        net.create_net()
        graph = net.get_net()

        # Create expected graph
        note1 = ms.note.Note("C")
        note2 = ms.note.Note("D")
        note3 = ms.note.Note("E")
        note4 = ms.note.Note("F")
        node1 = net.build_node(net.parse_elt(note1)[0])
        node2 = net.build_node(net.parse_elt(note2)[0])
        node3 = net.build_node(net.parse_elt(note3)[0])
        node4 = net.build_node(net.parse_elt(note4)[0])
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node1, weight=1, pitch_class="C")
        exp_graph.add_node(node2, weight=1, pitch_class="D")
        exp_graph.add_node(node3, weight=1, pitch_class="E")
        exp_graph.add_node(node4, weight=1, pitch_class="F")

        exp_graph.add_edge(node1, node3, weight=1)
        exp_graph.add_edge(node1, node4, weight=1)
        exp_graph.add_edge(node2, node3, weight=1)
        exp_graph.add_edge(node2, node4, weight=1)
        self.assert_graph_match(graph, exp_graph)
    
    def test_duration_weighted_intergraph(self):
        self.current_test = "duration_weighted_intergraph"

        # Create midi file, tinyNotation does not work when using multiple parts
        part1 = ms.stream.Part()
        part1.append(ms.note.Note("E"))
        part1.append(ms.note.Note("C", quarterLength=2))
        part2 = ms.stream.Part()
        part2.append(ms.note.Note("C", quarterLength=0.5))
        part2.append(ms.note.Note("D", quarterLength=2))
        part2.append(ms.note.Note("E", quarterLength=0.5))
        stream = ms.stream.Stream([part1, part2])
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, layer=True, duration_weighted_intergraph=True)
        net.create_net()
        graph = net._get_intergraph(net.net)

        # Create expected graph
        parsed_elt0_1 = net.parse_elt(ms.note.Note("E"))
        parsed_elt0_1["layer"] = 0
        parsed_elt0_2 = net.parse_elt(ms.note.Note("C"))
        parsed_elt0_2["layer"] = 0
        parsed_elt1_1 = net.parse_elt(ms.note.Note("C"))
        parsed_elt1_1["layer"] = 1
        parsed_elt1_2 = net.parse_elt(ms.note.Note("D"))
        parsed_elt1_2["layer"] = 1
        parsed_elt1_3 = net.parse_elt(ms.note.Note("E"))
        parsed_elt1_3["layer"] = 1
        node0_1 = net.build_node(parsed_elt0_1)
        node0_2 = net.build_node(parsed_elt0_2)
        node1_1 = net.build_node(parsed_elt1_1)
        node1_2 = net.build_node(parsed_elt1_2)
        node1_3 = net.build_node(parsed_elt1_3)
        exp_graph = nx.DiGraph()
        exp_graph.add_node(node0_1, weight=1, pitch_class="E")
        exp_graph.add_node(node0_2, weight=1, pitch_class="C")
        exp_graph.add_node(node1_1, weight=1, pitch_class="C")
        exp_graph.add_node(node1_2, weight=1, pitch_class="D")
        exp_graph.add_node(node1_3, weight=1, pitch_class="E")
        exp_graph.add_edge(node0_1, node1_1, weight=0.5)
        exp_graph.add_edge(node1_1, node0_1, weight=0.5)
        exp_graph.add_edge(node0_1, node1_2, weight=0.5)
        exp_graph.add_edge(node1_2, node0_1, weight=0.5)
        exp_graph.add_edge(node0_2, node1_2, weight=1.5)
        exp_graph.add_edge(node1_2, node0_2, weight=1.5)
        exp_graph.add_edge(node0_2, node1_3, weight=0.5)
        exp_graph.add_edge(node1_3, node0_2, weight=0.5)

        self.assert_graph_match(graph, exp_graph)
    
    
    def test_analyze_key(self):
        self.current_test = "analyze_key"
        
        # Create midi file
        stream = ms.converter.parse("tinyNotation: C D E") # does not matter
        file_path = self.test_folder + self.current_test + ".mid"
        stream.write("midi", file_path)

        # Without analysis
        self.current_test = "analyze_key(False)"
        
        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, analyze_key=False)
        net.create_net()
        if net.original_key is not None:
            raise Exception("analyze_key=False, but the analysis has been done")
        
        # Without analysis
        self.current_test = "analyze_key(False)"

        # Create net
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, analyze_key=True)
        net.create_net()
        if net.original_key is None:
            raise Exception("analyze_key=True, but the analysis has not been done")

    
if __name__ == "__main__":
    tester = MultilayerNetworkTester()
    tester.run_unit_tests()
    