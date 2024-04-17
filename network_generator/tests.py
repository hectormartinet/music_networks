from multilayer_net import *
import networkx as nx
import music21 as ms
import os

class MultilayerTester:
    
    def __init__(self):
        """
        Class to test the MultilayerNetwork class
        """
        self.test_folder = "tests_temp/"
        self.make_test_folder()

    def net_match(self, expected_net, **params):
        """
        Check if the builded net is the same as the expected one
        """
        try :
            net = MultiLayerNetwork(use_gui=False,verbosity=0,**params)
            net.create_net()
            nx.graphs_equal(net.get_net(), expected_net)
        except:
            return False
    
    def sub_net_match(self, expected_sub_nets, **params):
        """
        Check if the builded subnets are the same as the expected ones
        """
        try:
            net = MultiLayerNetwork(use_gui=False,verbosity=0,**params)
            net.create_net()
            for sub_net, expected_sub_net in zip(net.get_sub_net(), expected_sub_nets):
                if not nx.graph_equal(sub_net, expected_sub_net):
                    return False
            return True
        except:
            return False

        
    def test_run(self, **params):
        """
        Run a net build to test if it doesn't crash
        """
        try:
            net = MultiLayerNetwork(use_gui=False,verbosity=0,**params)
            net.create_net()
            net.get_net()
            net.get_sub_net()
            net.get_nodes_list()
            net1.convert_attributes_to_str()
            return True
        except:
            return False
    
    def make_test_folder(self):
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

    def remove_folder(self):
        for file in os.listdir(self.test_folder):
            path = self.test_folder + file
            os.remove(path)
        os.removedirs(self.test_folder)

    def test_enharmony(self, **params):
        stream = ms.converter.parse("tinyNotation: 3/4 E## F# G-")
        file_path = self.test_folder + "enharmony.musicxml"
        stream.write("musicxml",file_path)

        # With enharmony
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files = [file_path], enharmony=True)
        net.create_net()
        graph = net.get_net()
        note = ms.note.Note("F#")
        node = net.build_node(net.parse_elt(note, 0))
        assert(graph.number_of_nodes() == 1)
        assert(graph.number_of_edges() == 1)
        assert(graph.has_node(node))
        assert(graph.nodes[node]["weight"] == 3)
        assert(graph.has_edge(node, node))
        assert(graph[node][node]["weight"] == 2)

        # Without enharmony
        net = MultiLayerNetwork(use_gui=False, verbosity=0, midi_files = [file_path], enharmony=False)
        net.create_net()
        graph = net.get_net()
        notes =[ms.note.Note("E##"), ms.note.Note("F#"), ms.note.Note("G-")]
        nodes = [net.build_node(net.parse_elt(note, 0)) for note in notes]
        assert(graph.number_of_nodes() == 3)
        assert(graph.number_of_edges() == 2)
        for node in nodes:
            assert(graph.has_node(node))
            assert(graph.nodes[node]["weight"] == 1)
        for i in range(2):
            assert(graph.has_edge(nodes[i], nodes[i+1]))
            assert(graph[nodes[i]][nodes[i+1]]["weight"] == 1)
        


if __name__ == "__main__":
    tester = MultilayerTester()
    tester.test_enharmony()
    tester.remove_folder()     
    