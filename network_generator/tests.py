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
            raise Exception("The two graphs don't have the same number of nodes")
        if graph.number_of_edges() != exp_graph.number_of_edges():
            raise Exception("The two graphs don't have the same number of edges")
        for node, data in exp_graph.nodes(data=True):
            if not graph.has_node(node):
                raise Exception(f"The graph does not have the node : {node}")
            for key, value in data.items():
                if not key in graph.nodes[node]:
                    raise Exception(f"The node {node} does not have the attribute {key}")
                if graph.nodes[node][key] != value:
                    raise Exception("Expected value for node {node} on attribute {key} : {value}. Value found : {graph.nodes[node][key]}")
        for node1, node2, data in exp_graph.edges(data=True):
            if not graph.has_edge(node1, node2):
                raise Exception(f"The graph does not have the edge : {(node1,node2)}")
            for key, value in data.items():
                if graph[node1][node2][key] != value:
                    raise Exception(f"Expected value for edge {node1,node2}  on attribute {key} : {value}. Value found : {graph[node1][node2][key]}")

    def run_unit_tests(self):
        self.make_test_folder()
        try:
            self.test_enharmony()
            self.test_chord_function()
            self.test_group_by_beat()
            self.test_pitch()
        except Exception as err:
            print(f"Unit tests {self.current_test} failed, error :")
            print(err)
        else:
            print("Unit tests passed successfully")
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
        """
        self.current_test = "enharmony"

        # Create musicxml file
        stream = ms.converter.parse("tinyNotation: E## F# G-")
        file_path = self.test_folder + "enharmony.musicxml"
        stream.write("musicxml",file_path)

        # With enharmony
        self.current_test = "enharmony(True)"
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, enharmony=True)
        net.create_net()
        graph = net.get_net()
        exp_graph = nx.DiGraph()
        note = ms.note.Note("F#")
        node = net.build_node(net.parse_elt(note))
        exp_graph.add_node(node, weight=3)
        exp_graph.add_edge(node, node, weight=2)
        self.assert_graph_match(graph, exp_graph)

        # Without enharmony
        self.current_test = "enharmony(False)"
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files=file_path, enharmony=False)
        net.create_net()
        graph = net.get_net()
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
        file_path = self.test_folder + "chord.mid"
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
        file_path = self.test_folder + "group_by_beat.mid"
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
        file_path = self.test_folder + "pitch.mid"
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

if __name__ == "__main__":
    tester = MultilayerNetworkTester()
    tester.run_unit_tests()
    