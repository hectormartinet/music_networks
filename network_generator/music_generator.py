import json
import music21 as ms
import random
import networkx as nx
import os
import numpy as np

import multilayer_net


class MusicGenerator:
    def __init__(self, outfolder) -> None:
        self.part_list = []
        self.nodes_lists = []
        self.instruments = []
        self.outfolder = outfolder

    def extract_node_info(self, node):
        return json.loads(node)
    
    def duration(self, network, node): 
        result = network.nodes[node]["duration"]
        return result if type(result) == float else 1.
    
    def node_to_note(self, node, network=None):
        node_info = self.extract_node_info(node)
        pitches = node_info["pitch"]
        if pitches == "rest":
            note = ms.note.Rest()
        else:
            pitches = node_info["pitch"].split(" ")
            if len(pitches) > 2:
                note = ms.chord.Chord(pitches)  
            else: 
                note = ms.note.Note(pitches[0])
        if "duration" in node_info:
            note.quarterLength = node_info["duration"]
        return note

    def network_edge_weight(self, network, prev_node, next_node):
        return network[prev_node][next_node]["weight"] if network.has_edge(prev_node, next_node) else 0
    
    def network_node_weight(self, network, node):
        return network.nodes[node]["weight"] if network.has_node(node) else 0
    
    def intergraph_weight(self, intergraph, node, indices, timestamp, duration, laplace_smoothing=1):
        intergraph_log_weight = 0
        assert(len(indices) == len(self.part_list))
        for lst_id,idx in enumerate(indices):
            note = self.part_list[lst_id][idx]
            t_min = note.offset
            t_max = note.offset + note.quarterLength
            assert(t_max>timestamp)
            while t_min < timestamp + duration:
                other_node = self.nodes_lists[lst_id][idx]
                weight = (min(t_max, timestamp + duration) - max(t_min, timestamp))/duration
                intergraph_log_weight +=  weight*np.log(self.network_edge_weight(intergraph, node, other_node) + laplace_smoothing)
                idx += 1
                if idx >= len(self.part_list[lst_id]):
                    break
                note = self.part_list[lst_id][idx]
                t_min = note.offset
                t_max = note.offset + note.quarterLength
        return np.exp(intergraph_log_weight/len(indices))

    def random_node(self, network):
        return random.choice(list(network.nodes()))

    def weighted_random_node(self, network):
        weights = [self.network_node_weight(network, node) for node in network]
        return random.choices(list(network.nodes()), weights)[0]
    
    def total_weighted_random_node(self, network, intergraph, indices, timestamp, alpha, beta, laplace_smoothing=1):
        weights = [self.total_node_weight(network, intergraph, node, indices, timestamp, self.duration(network,node), alpha, beta, laplace_smoothing) for node in network]
        return random.choices(list(network.nodes()), weights)[0]

    def random_next_node(self, network, prev_node):
        if len(network[prev_node])==0:
            print("No next node found")
            return self.random_node(network)
        return random.choice(list(network.neighbors(prev_node)))
    
    def weighted_random_next_node(self, network, prev_node):
        if len(network[prev_node])==0:
            print("No next node found")
            return self.weighted_random_node(network)
        weights = [self.network_edge_weight(network, prev_node, node) for node in network[prev_node]]
        return random.choices(list(network.neighbors(prev_node)), weights)[0]
    
    def total_weighted_random_next_node(self, network, intergraph, prev_node, indices, timestamp, alpha, beta):
        weights = [self.total_edge_weight(network,intergraph,prev_node,node,indices,timestamp,self.duration(network,prev_node),alpha,beta) for node in network[prev_node]]
        if sum(weights)==0:
            return self.total_weighted_random_node(network,intergraph,indices,timestamp,alpha,beta)
        return random.choices(list(network.neighbors(prev_node)), weights)[0]
    
    def total_edge_weight(self, network, intergraph, prev_node, next_node, indices, timestamp, duration, alpha, beta, laplace_smoothing=1):
        network_weight = self.network_edge_weight(network, prev_node, next_node)
        intergraph_weight = self.intergraph_weight(intergraph, next_node, indices, timestamp, duration, laplace_smoothing)
        return network_weight**alpha * intergraph_weight**beta
    
    def total_node_weight(self, network, intergraph, node, indices, timestamp, duration, alpha, beta, laplace_smoothing):
        network_weight = self.network_node_weight(network, node)
        intergraph_weight = self.intergraph_weight(intergraph, node, indices, timestamp, duration, laplace_smoothing)
        return network_weight**alpha * intergraph_weight**beta
    
    def weighted_rw(self, network, number_of_notes=100, first_node=None):
        """Do weighted random walk on the network

        Args:
            network (NetworkX): The network to walk on
            number_of_notes (int, optional): Number of note to generate in the stream. Defaults to 100.

        Returns:
            stream: The final stream
        """
        part = ms.stream.Part()
        part.id = len(self.part_list)
        if first_node is None or not network.has_node(first_node):
            def filter(node): return network.nodes[node]["offset"]==0.0
            node_list = [self.weighted_random_node(nx.subgraph_view(network,filter_node=filter))]
        else:
            node_list = [first_node]
        part.append(self.node_to_note(node_list[-1]))
        for _ in range(number_of_notes-1):
            node_list.append(self.weighted_random_next_node(network, node_list[-1]))
            part.append(self.node_to_note(node_list[-1]))
        # stream.show()
        self.part_list.append(part)
        self.nodes_lists.append(node_list)

    def add_voice_with_rw(self, network, intergraph, alpha=1., beta=1., first_node=None):
        """Do weighted random walk on the network biased by other voices

        Args:
            network (NetworkX): The network to walk on
            intergraph (NetworkX): The intergraph biasing the random walk
            alpha (float, optional): Weight for the network
            beta (float, optional): Weight for the intergraph
            first_node (node of network, optional): first node forced.
        """
        part = ms.stream.Part()
        nb_voices = len(self.nodes_lists)
        part.id = nb_voices
        timestamp = 0
        indices = [0]*nb_voices # to keep track of the notes that play at the same time
        loop = True
        if first_node is None or not network.has_node(first_node):
            other_nodes = [lst[0] for lst in self.nodes_lists]
            def filter(node): return network.nodes[node]["offset"]==0.0
            node_list = [self.total_weighted_random_node(nx.subgraph_view(network, filter_node=filter),intergraph,indices,timestamp,alpha,beta)]
        else:
            node_list = [first_node]
        part.append(self.node_to_note(node_list[-1]))
        while True:
            # Update timestamp
            duration = part[-1].quarterLength
            timestamp += duration
            # Update indices
            for i in range(nb_voices):
                note = self.part_list[i][indices[i]]
                while timestamp >= note.offset + note.quarterLength:
                    indices[i] += 1
                    if indices[i] >= len(self.part_list[i]):
                        loop = False
                        break
                    note = self.part_list[i][indices[i]]
            if not loop:
                break
            node_list.append(self.total_weighted_random_next_node(network,intergraph,node_list[-1],indices,timestamp,alpha,beta))
            part.append(self.node_to_note(node_list[-1]))
        # stream.show()
        self.part_list.append(part)
        self.nodes_lists.append(node_list)

    def show_music(self):
        ms.stream.Stream(self.part_list).show()
            

if __name__ == "__main__":

    directory = "midis\\invent_bach\\"
    files_number = [1,2,5,7,8,9,11,13,14] # pieces in binary to not mess up with the offset parameter
    # midi_files = [directory + f for f in os.listdir(directory)]
    midi_files = [directory + "invent"+str(n)+".mid" for n in files_number]
    melody_file = [directory + "invent1.mid"]

    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net = multilayer_net.MultiLayerNetwork(use_gui=False, output_folder="", name="test", pitch=True, duration=True, octave=True, offset=True, rest=True, midi_files=midi_files, enharmony=True, transpose=True)
    net2 = multilayer_net.MultiLayerNetwork(use_gui=False, output_folder="", name="test", pitch=True, duration=True, octave=True, offset=True, rest=True, midi_files=melody_file, enharmony=True, transpose=True)
    
    # Call createNet function
    net.create_net()
    net2.create_net()

    generator = MusicGenerator("")
    # generator.weighted_rw(net.sub_net[0])
    generator.nodes_lists.append(net2._get_nodes_list(0))
    generator.part_list.append(ms.stream.Part([generator.node_to_note(node) for node in generator.nodes_lists[-1]]))
    for i in range(1,len(net.aggregated_sub_nets)):
        generator.add_voice_with_rw(net.aggregated_sub_nets[i], net.aggregated_intergraph, alpha=1, beta=1)
    generator.show_music()