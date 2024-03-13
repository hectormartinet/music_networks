from music21 import *
import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import pyautogui
import math
import random

class MultiLayerNetwork:
    def __init__(self, midifilename, outfolder):
        self.Net=nx.DiGraph(); self.negNet=nx.DiGraph()
        self.subNet = []; self.negSubNet = []
        self.outfolder = outfolder
        self.stream_list=[]; self.negative_stream_list=[]
        self.instruments=[]
        self.intergraph = None
        self.name = midifilename
        self.midi_file = midifilename
        whole_piece=converter.parse(midifilename)
        for part in whole_piece.parts: # loads each channel/instrument into stream list
            partC= self.stream_to_C(part)            
            self.stream_list.append(partC)
            negPart = self.getNegativeHarmony(part)
            negPartC= self.stream_to_C(negPart)  
            self.negative_stream_list.append(negPartC)     
        for el in whole_piece.recurse():
            if 'Instrument' in el.classes:
                self.instruments.append(str(el))
        
    def createNet(self, bool):
        """Create the main network
        """
        print("[+] Converting MIDI file to network")
        if(not bool):
            self.streamToNetwork()
        else:
            self.streamToCompleteNetwork()
    
    def createNegNet(self):
        """Create the main network
        """
        print("[+] Converting MIDI file to network")
        self.negStreamToNetwork()

    def list_to_string(self,my_list):
        return ','.join(str(x) for x in my_list)

    def string_to_list(self,my_string):
        return [float(x) for x in my_string.split(',')]

    def exportNet(self, filename):
        """Export the network to a graphml file

        Args:
            filename (string): Output filename
        """

        print("[+] Writing main graphml file to : " + filename)
        nx.write_graphml(self.Net, filename)
    
    def exportSubNet(self, folder,filename2):
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
        for i in range(0,len(self.subNet)):
            curOut = filename + "_" + str(i) + ".graphml"
            nx.write_graphml(self.subNet[i], curOut)
    
    def getNet(self):
        """Getter for the network

        Returns:
            NetwrokX: The main network
        """
        return self.Net

    def getNegNet(self):
        """Getter for the network

        Returns:
            NetwrokX: The main network
        """
        return self.negNet
    
    def getInstruments(self):
        """Return the list of instruments name (one per layer)

        Returns:
            List string: List of instruments name
        """
        return self.instruments
        
    def getSubNet(self, layer=None):
        """Return the list of subnetworks

        Args:
            layer (Number of the layer, optional): Specify the layer to return. Defaults to None.

        Returns:
            List NetworkX: The list of subnetworks
        """
        n=self.getNet()
        #list of subnets (maybe to change function in multilayer class)
        s_len=len(self.stream_list)
        self.subNet =[]
        for i in range (0,s_len):
            v= self.filter_nodes(n,i)
            self.subNet.append(n.subgraph(v))
        t=self.filter_edges(n)
        H = n.edge_subgraph(t)
        self.intergraph = H.to_undirected()
        return self.subNet, self.intergraph
    
    def getNegSubNet(self, layer=None):
        """Return the list of subnetworks

        Args:
            layer (Number of the layer, optional): Specify the layer to return. Defaults to None.

        Returns:
            List NetworkX: The list of subnetworks
        """
        n=self.getNet()
        #list of subnets (maybe to change function in multilayer class)
        s_len=len(self.negative_stream_list)
        self.negSubNet =[]
        for i in range (0,s_len):
            v= self.filter_nodes(n,i)
            self.negSubNet.append(n.subgraph(v))
        t=self.filter_edges(n)
        H = n.edge_subgraph(t)
        self.intergraph = H.to_undirected()
        return self.negSubNet, self.intergraph

    def convert_numpy_arrays_to_lists(self):
        for node in self.Net.nodes():
            self.Net.nodes[node]['t'] = self.Net.nodes[node]['t'].tolist()
            mylist=self.Net.nodes[node]['t']
            att=self.list_to_string(mylist)
            self.Net.nodes[node]['t'] = att
            for neighbor in self.Net.neighbors(node):
                self.Net[node][neighbor]['t'] = self.Net[node][neighbor]['t'].tolist()
                mylist=self.Net[node][neighbor]['t']
                att=self.list_to_string(mylist)
                self.Net[node][neighbor]['t']=att
        return self.Net

        #returns list of nodes of given layer
    def filter_nodes(self, n, l):
        filtered_nodes=[]
        for s in n.nodes():
            if n.nodes()[s]['l']==l:
                filtered_nodes.append(s)
        return filtered_nodes
            
    #returns the list of interlayer edges       
    def filter_edges(self, n):
        filtered_edges=[]
        for s in n.edges():
            if n.edges()[s]['inter']==True:
                filtered_edges.append(s)
        return filtered_edges

    def getPositionOfNode(self, rad, angle, xinit, yinit):
        circle_x = rad * math.cos(angle) + xinit # COORDINATES
        circle_y = rad * math.sin(angle) + yinit # COORDINATES
        return circle_x, circle_y

    def add_or_update_node(self, node, timestamp, is_first_note, last_note, i, x, y):
        timestamps = self.Net.nodes[node]['t'] if self.Net.has_node(node) else np.array([])
        timestamps = self.update_timestamps(timestamps, timestamp)
        onw = self.Net.nodes[node]['w'] if self.Net.has_node(node) else 0
        ond = 0 if is_first_note else self.Net.degree[last_note]
        self.Net.add_node(node, w=onw + 1, t=timestamps, at_deg=ond, l=i, instr=self.instruments[i], pitch=node[1],
                          dur=node[2], x_coord=x, y_coord=y)

    def add_or_update_edge(self, from_node, to_node, timestamp, is_inter_layer):
        if self.Net.has_edge(from_node, to_node):
            self.Net[from_node][to_node]['w'] += 1
            edge_timestamps = self.update_timestamps(self.Net[from_node][to_node]['t'], timestamp)
            self.Net[from_node][to_node]['t'] = edge_timestamps
        else:
            edge_timestamps = self.update_timestamps(np.array([]), timestamp)
            self.Net.add_edge(from_node, to_node, w=1, t=edge_timestamps, inter=is_inter_layer)

    def process_intra_layer(self, i, first, last_note, s_len, circle_x, circle_y):
        rad, circle_r, bigangle, alphabig = 35, 4, 0, 360 / s_len
        s_flat = self.stream_list[i].flat
        for el in s_flat.notesAndRests:
            n_temp = self.parseElement(el)
            if n_temp != 'other':  # Skip rest notes
                x, y = self.getPositionOfNode(circle_r * math.sqrt(random.random()), 2 * math.pi * random.random(),
                                              circle_x, circle_y)
                n = (i, n_temp[0], n_temp[1])
                timestamp = float(el.offset)
                self.add_or_update_node(n, timestamp, first, last_note, i, x, y)
                if not first:
                    self.add_or_update_edge(last_note, n, timestamp, False)
                last_note = n
                first = False
        return last_note

    def process_inter_layer(self, i, s_len):
        s_flat = self.stream_list[i].flat
        for el in s_flat.notesAndRests:
            n_temp = self.parseElement(el)
            if n_temp != 'other':  # Skip rest notes
                n = (i, n_temp[0], n_temp[1])
                timestamp = float(el.offset)
                for j in range(i + 1, s_len):
                    sOut = self.stream_list[j].getElementsByOffset(timestamp,
                                                                   timestamp + el.quarterLength).stream().flat
                    for el2 in sOut.notesAndRests:
                        n2_temp = self.parseElement(el2)
                        if n2_temp != 'other':  # Skip rest notes
                            n2 = (j, n2_temp[0], n2_temp[1])
                            self.add_or_update_edge(n, n2, timestamp, True)

    def streamToNetwork(self):
        s_len = len(self.stream_list)
        pbar = tqdm(total=s_len)
        print("[+] Creating network - Intra-layer processing")
        rad, circle_r, bigangle, alphabig = 35, 4, 0, 360 / s_len

        for i in range(s_len):  # For each instrument
            circle_x, circle_y = self.getPositionOfNode(rad, bigangle, 0, 0)
            bigangle += alphabig
            last_note = self.process_intra_layer(i, True, ' ',s_len,circle_x,circle_y)
            pbar.update(1)

        try:
            print("[+] Creating network - Inter-layer processing")
            lenfin = sum(len(self.stream_list[i].flat.notesAndRests) for i in range(s_len - 1))
            pbar = tqdm(total=lenfin)

            for i in range(s_len - 1):
                self.process_inter_layer(i, s_len)
                pbar.update(len(self.stream_list[i].flat.notesAndRests))

            pbar.close()
        except:
            print("[!] Warning - This MIDI file has only one layer")
        return self.Net

    def negStreamToNetwork(self):
        """Convert the stream to a network

        Returns:
            NetworkX: The main network
        """
        s_len=len(self.negative_stream_list)
        pbar = tqdm(total=s_len) 
        #INTRA-LAYER PROCESSING
        print("[+] Creating negative network - Intra-layer processing")       

        rad = 35; circle_r = 4; bigangle = 0; alphabig = 360 / s_len # COORDINATES
        for i in range (0,s_len): #for each instrument in the piece
            circle_x, circle_y = self.getPositionOfNode(rad, bigangle, 0, 0)
            bigangle+=alphabig # COORDINATES
            first=True
            s_flat=self.negative_stream_list[i].flat
            for el in s_flat.notesAndRests: #for each note of the instrument
                x, y = self.getPositionOfNode(circle_r * math.sqrt(random.random()), 2 * math.pi * random.random(), circle_x, circle_y)
                n_temp= self.parseElement(el) # parse note, rest or chord
                timestamp=float(el.offset) # detect offset
                if n_temp!='other':
                    n=(i,n_temp[0],n_temp[1]) # define node name
                    deg=0
                    if first==True: #if first note of the voice
                        self.negNet.add_node(n, w=1,t=timestamp,at_deg=deg,l=i,instr=self.instruments[i],pitch=n_temp[0],dur=n_temp[1],x_coord=x,y_coord=y)
                    else:
                        if self.negNet.has_edge(last_note,n): # if edge already exists update weights
                            self.negNet[last_note][n]['w']+=1 #updated edge weight
                            onw, ont, ond=self.negNet.nodes[n]['w'], self.negNet.nodes[n]['t'], self.negNet.nodes[n]['at_deg']
                        else: #if edge does not exist, add edge
                            if self.negNet.has_node(n):
                                onw, ont, ond=self.negNet.nodes[n]['w'], self.negNet.nodes[n]['t'], self.negNet.nodes[n]['at_deg']
                            else:
                                onw, ont, ond=1, timestamp, self.negNet.degree[last_note]
                            self.negNet.add_edge(last_note,n, w=1,t=timestamp,inter=False)        
                        self.negNet.add_node(n, w=onw+1,t=ont,at_deg=ond,l=i,instr=self.instruments[i],pitch=n_temp[0],dur=n_temp[1],x_coord=x,y_coord=y) #add node
                last_note=n
                first=False
            last_note=' ' 
            pbar.update(1)
        pbar.close()
        # INTER-LAYER PROCESSING
        print("[+] Creating negative network - Inter-layer processing")
        lenfin = 0
        for i in range (0,s_len-1): #for each voice in the piece
            lenfin += len(self.negative_stream_list[i].flat.notesAndRests)
        pbar = tqdm(total=lenfin)
        for i in range (0,s_len-1):# for all voices
            s_flat= self.negative_stream_list[i].flat
            for el in s_flat.notesAndRests: #for each element of the voice
                n_temp= self.parseElement(el)
                n=(i,n_temp[0],n_temp[1])
                timestamp=float(el.offset)
                for j in range (i+1,s_len):
                    sOut = self.negative_stream_list[j].getElementsByOffset(timestamp,timestamp+el.quarterLength).stream()
                    sOut=sOut.flat
                    for el2 in sOut.notesAndRests:
                        n2_temp= self.parseElement(el2)
                        n2=(j,n2_temp[0],n2_temp[1])
                        comb=str(i)+" "+str(j) 
                        if self.negNet.has_edge(n,n2):
                            self.negNet[n][n2]['w']+=1
                        else:
                            self.negNet.add_edge(n,n2, w=1,t=timestamp,inter=True,layers=comb)
                pbar.update(1)
        pbar.close()
        return self.negNet

    def update_timestamps(self,timestamps, new_time):
        # add a new number to the array and update the node attribute
        new_timestamps = np.append(timestamps, new_time)
        return new_timestamps

    def parseElement(self, el):
        p='other'
        if el.isNote:
            p=(str(el.pitch),str(el.quarterLength))
        if el.isRest:
            p=(el.name,str(el.quarterLength))
            p = 'other'
        if el.isChord:
            c=''
            for i in el.pitches:
                c+=str(i)+" "
            p=(c,str(el.quarterLength))
        return p
    
    def most_frequent(self, List):
        """Give the most common element in a list

        Args:
            List (Any): A list

        Returns:
            Any: The most common element in the list
        """
        if(len(List)!=0):
            return max(set(List), key = List.count)
        else:
            return None
    
    def convertFracToFloat(self, frac):
        """Convert a fraction to a float

        Args:
            frac (String): A fraction in the form of a string

        Returns:
            float: The float value of the fraction
        """
        return float(frac.split("/")[0])/float(frac.split("/")[1])
    
    def getDynamicAnalysis(self, type="degree"):
        lst_X = []; lst_Y = []
        lst_t = sorted([self.Net.nodes[n]['t'] for n in self.Net.nodes])
        lst_edge_t = sorted([self.Net.edges[n]['t'] for n in self.Net.edges])
        lst_t = sorted([*set(lst_t)])
        lst_edge_t = sorted([*set(lst_edge_t)])
        for subnet in self.subNet: # For each instrument
            tmp_lst_x = []; tmp_lst_y = []
            nodes=sorted(subnet.nodes(data=True), key=lambda t: t[1]['t'])
            subnet_t = sorted([subnet.nodes[n]['t'] for n in subnet.nodes])
            subnet_edge_t = sorted([subnet.edges[n]['t'] for n in subnet.edges])
            nb_node = 0; nb_edge = 0
            for i in range(0,len(lst_t)):        
                if(type=="edge"):       
                    if(lst_edge_t[i] in subnet_edge_t):
                        tmp_lst_x.append(lst_edge_t[i])   
                        lst_indic = []
                        for j in range(0,len(subnet_edge_t)):
                            if(subnet_edge_t[j]==lst_edge_t[i]):
                                lst_indic.append(j)              
                        nb_edge += len(lst_indic)
                        tmp_lst_y.append(nb_edge)
                    else:
                        tmp_lst_y.append(nb_edge)
                else:
                    if(lst_t[i] in subnet_t):
                        index = subnet_t.index(lst_t[i])
                        tmp_lst_x.append(lst_t[i])

                        if(type=="degree"):
                            tmp_lst_y.append(subnet.degree[nodes[index][0]])
                        elif(type=="node"):
                            indices = [i for i, x in enumerate(subnet_t) if x == lst_t[i]]
                            nb_node += len(indices)
                            tmp_lst_y.append(nb_node)
                        elif(type=="duration"):
                            try:
                                tmp_lst_y.append(float(subnet.nodes[nodes[index][0]]['dur']))
                            except:
                                tmp_lst_y.append(self.convertFracToFloat(subnet.nodes[nodes[index][0]]['dur']))
                        elif(type=="weight"):
                            tmp_lst_y.append(float(subnet.nodes[nodes[index][0]]['w']))
                    else:
                        if(type=="node"):
                            tmp_lst_y.append(nb_node)
                        else:
                            tmp_lst_y.append(0)
            lst_X.append(lst_t)
            lst_Y.append(tmp_lst_y)
        if(type=="degree"):
            self.showPlot(lst_X, lst_Y, "Dynamic degree")
        elif(type=="node"):
            self.showPlot(lst_X, lst_Y, "Dynamic cumulative new nodes")
        elif(type=="edge"):
            self.showPlot(lst_X, lst_Y, "Dynamic cumulative new edges")
        elif(type=="duration"):
            self.showPlot(lst_X, lst_Y, "Dynamic duration")
        elif(type=="weight"):
            self.showPlot(lst_X, lst_Y, "Dynamic weight")
    
    def getMultiLayerAnalysis(self):
        """Do the multi-layer analysis

        Returns:
            Panda datafram: The datafram containing all the different analysis
        """
        nb_nodes = []; duration = []; pitch = []; pitch_layer = []; dur_layer = []
        
        # Get file song name 
        if(len(self.midi_file.split("/")) > 1):
            name = self.midi_file.split("/")[-1].split(".")[0].replace(" ","") 
        else:
            name = self.midi_file.split("\\")[-1].split(".")[0].replace(" ","")
        
        for net in self.subNet:
            tmp_lay_pitch = []; tmp_lay_dur = []
            nb_nodes.append(len(net.nodes()))
            for val in nx.get_node_attributes(net, "dur").values():
                if("/" in val):
                    val = self.convertFracToFloat(val)
                duration.append(float(val))
                tmp_lay_dur.append(float(val))
            for val in nx.get_node_attributes(net, "pitch").values():
                if(val != "rest"):
                    pitch.append(val)
                    tmp_lay_pitch.append(val)
            pitch_layer.append(tmp_lay_pitch)
            dur_layer.append(tmp_lay_dur)
        
        # Give most common pitch by layer
        pitches = {}
        for i in range(0,len(pitch_layer)):
            res = self.most_frequent(pitch_layer[i])
            pitches[self.instruments[i]] = [res]
        dfpitch = pd.DataFrame(pitches)
        
        # Give most common duration by layer
        durations = {}
        for i in range(0,len(dur_layer)):
            res = round(np.mean(dur_layer[i]), 3)
            durations[self.instruments[i]] = [res]
        dfdur = pd.DataFrame(durations)
            
        nb_instru = [len(self.stream_list)]
        max_nodes = [max(nb_nodes)]
        avg_nodes = [np.mean(nb_nodes)]
        most_common_duration = [self.most_frequent(duration)]
        avg_duration = [np.mean(duration)]
        most_common_pitch = [self.most_frequent(pitch)]
        
        bach_data = {'Num instruments': nb_instru, "Longest seq of notes": max_nodes, 
                     "Avg num of notes by instru": avg_nodes, "Most common pitch": most_common_pitch,
                     "Most common duration": most_common_duration, "Avg duration": avg_duration}
        df = pd.DataFrame(bach_data)
        return df, dfdur, dfpitch, self.instruments, name
    
    def getAnalysis(self):
        """Do simple analysis

        Returns:
            Pandas datafram: The pandas dataframe containing the analysis
        """
        list_lengths,list_nodes,list_edges=[],[],[]
        list_avg_degree,list_max_degree,list_diameter=[],[],[]
        list_clustcoef,list_avg_path,list_density=[],[],[]

        list_nodes.append((len(self.Net.nodes())))
        list_edges.append(len(self.Net.edges()))
        degrees = [deg for node, deg in nx.degree(self.Net)]
        kavg = np.mean(degrees)
        kmax = np.max(degrees)
        list_avg_degree.append(kavg)
        list_max_degree.append(kmax)
        und=self.Net.to_undirected()
        diam=nx.diameter(und)
        list_diameter.append(diam)
        cc = nx.clustering(self.Net)
        avg_cc = (sum(cc.values()) / len(cc))
        list_clustcoef.append(float(avg_cc))
        avg_path=nx.average_shortest_path_length(self.Net, weight=None)
        list_avg_path.append(avg_path)
        list_density.append(nx.density(self.Net))
        #list_lengths.append(self.Net.graph['track_length'])
        bach_data = {'Nodes': list_nodes,'Edges': list_edges, 'Avg Degree': list_avg_degree, 
                     'Max Degree': list_max_degree, 'Diameter': list_diameter, 'Avg Path': list_avg_path,
                     'Clustering': list_clustcoef, 'Density': list_density}
        df = pd.DataFrame(bach_data, columns = ['Nodes', 'Edges', 'Avg Degree',
                                                'Max Degree','Diameter','Avg Path',
                                                'Clustering','Density'])
        return df


    def plotDegreeDistrib(self, folder):
        """Save the degree distribution plot for the main network and all the subnetworks

        Args:
            folder (string): The folder where the plot will be saved
        """
        filename = folder + "layer"
        print("[+] Plotting degree distribution")
        
        # For the big net
        if(len(self.midi_file.split("/")) > 1):
            curOut = folder + "0_"+ self.midi_file.split("/")[-1].split(".")[0].replace(" ","") + ".png"
        else:
            curOut = folder + "0_"+ self.midi_file.split("\\")[-1].split(".")[0].replace(" ","") + ".png"
        degree_sequence = sorted([d for n, d in self.Net.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        fig, ax = plt.subplots()
        plt.bar(deg, cnt, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.savefig(curOut)
        
        # For each layer (subnet)
        pbar = tqdm(total=len(self.subNet))
        for i in range(0,len(self.subNet)):
            curOut = filename + "_" +str(i+1) + ".png"
            degree_sequence = sorted([d for n, d in self.subNet[i].degree()], reverse=True)  # degree sequence
            degreeCount = collections.Counter(degree_sequence)
            deg, cnt = zip(*degreeCount.items())
            fig, ax = plt.subplots()
            plt.bar(deg, cnt, color='b')
            plt.title("Degree Histogram for " + self.instruments[i])
            plt.ylabel("Count")
            plt.xlabel("Degree")
            plt.savefig(curOut)
            pbar.update(1)
        pbar.close()

    def jointoNet(self, small_net,big_net):
        big_net.graph['track_length']+=small_net.graph['track_length']
        for node in small_net.nodes():
            if big_net.has_node(node):
                w=big_net.nodes[node]['weight']+small_net.nodes[node]['weight']
            else:
                w=small_net.nodes[node]['weight']
            big_net.add_node(node,weight=w)
        for edge in small_net.edges():
            if big_net.has_edge(edge[0],edge[1]):
                big_net[edge[0]][edge[1]]['weight']+=small_net[edge[0]][edge[1]]['weight']
            else:
                ew=small_net[edge[0]][edge[1]]['weight']
                big_net.add_edge(edge[0],edge[1],weight=ew)
        return big_net
    
    def stream_to_C(self, part):
        k = part.flat.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        part_transposed = part.transpose(i)
        return part_transposed
    
    def streams_to_MinMajStreams(self, streams_list):
        """
        Transposes all streams to 2 lists of streams in C major and C minor
        """
        M_streams_list=[]
        m_streams_list=[]
        for piece in streams_list:
            k = piece.analyze('key')
            if k.mode == 'major':
                i = interval.Interval(k.tonic, pitch.Pitch('C'))
                piece_transposed =piece.transpose(i)
                M_streams_list.append(piece_transposed)
            else:
                i = interval.Interval(k.tonic, pitch.Pitch('C'))
                piece_transposed =piece.transpose(i)
                m_streams_list.append(piece_transposed)
                
        return M_streams_list, m_streams_list
    
    def getNegativeHarmony(self, part):
        k = part.flat.analyze('key')
        i = self.negative_dict()[k.tonic.name]
        part_transposed = part.transpose(i)
        return part_transposed
    
    def negative_dict(self):
        # 1,i2,3,4,i5,6,7,i8,9,i10,11,12
        # C,C#,D,D#,E,F,F#,G,G#,A,A#,B
        # A,Ab,G,F#,F,E,Eb,D,Db,C,B,Bb
        #define notes in scale
        d={}
        a1 = note.Note('C3')
        a2 = note.Note('C#3')
        a3 = note.Note('D3')
        a4 = note.Note('D#3')
        a5 = note.Note('E3')
        a6 = note.Note('F3')
        a7 = note.Note('F#3')
        a8 = note.Note('G3')
        a9 = note.Note('G#3')
        a10 = note.Note('A3')
        a11=note.Note('A#3')
        a12=note.Note('B3')
        #define intervals following Negative Harmony in G
        i1 = interval.Interval(a1, a10)
        i2 = interval.Interval(a2, a9)
        i3 = interval.Interval(a3, a8)
        i4 = interval.Interval(a4, a7)
        i5 = interval.Interval(a5, a6)
        i6 = interval.Interval(a6, a5)
        i7 = interval.Interval(a7, a4)
        i8 = interval.Interval(a8, a3)
        i9 = interval.Interval(a9, a2)
        i10 = interval.Interval(a10, a1)
        i11= interval.Interval(a11, a12)
        i12= interval.Interval(a12, a11)
        d = {'C':i1,'C#':i2,'D-':i3,'D':i3,'D#':i4,'E-':i4,'E':i5,'F':i6,'F#':i7,'G-':i7,'G':i8,'G#':i9,'A-':i9,'A':i10,'A#':i11,'B-':i11,'B':i12}
        return d

    def showPlot(self, lst_X, lst_Y, title):
        """Plot a bar chart

        Args:
            lst_x (lst): List of the x axis
            lst_y (lst): List of the y axis
            title (str): Title of the chart
        """
        fig = plt.figure(figsize = (18, 8))
        width = 0.8
        for i in range(0,len(lst_X)):
            if("new" in title):
                plt.plot(lst_X[i], lst_Y[i], label=self.instruments[i], alpha=0.5)
            elif("duration" in title):
                plt.plot(lst_X[i], lst_Y[i], label=self.instruments[i], marker='.', linestyle = 'None', alpha=0.5)
            else:
                plt.bar(lst_X[i], lst_Y[i], label=self.instruments[i], width=width, alpha=0.75)
        plt.ylabel(title.split(" ")[-1])
        plt.xlabel("Timeline")
        plt.xticks(rotation = 45, fontsize=6) 
        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.subplots_adjust(right=0.7)
        name = self.outfolder + title.replace(" ", "_") + ".png"
        try:
            plt.savefig(name)
        except:
            plt.show()
            print("[-] Error - Unable to save the chart")