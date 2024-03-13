from re import sub
from select import select
import networkx as nx
import matplotlib.pyplot as plt
from music21 import *
import numpy as np
from random import choice,choices, randint
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


class MusicGenerator:
    def __init__(self, outfolder) -> None:
        self.Stream_list=[]
        self.instruments = []
        self.outfolder = outfolder
    
    def setInstruments(self, instruments):
        self.instruments = instruments
    
    def getInstruments(self):
        return self.instruments
    
    def reset(self):
        self.Stream_list=[]
        self.instruments = []
    
    def simpleRW(self, network, iterations=1, number_of_notes=50, max_lenght= 50):
        """Do simple random walk on the network

        Args:
            network (NetworkX): The network to walk on
            iterations (int, optional): The number of iteration for the algorithm. Defaults to 5.
            number_of_notes (int, optional): Number of note to generate in the stream. Defaults to 50.

        Returns:
            stream: The final stream
        """
        tmp_stream=stream.Stream()
        pbar = tqdm(total=iterations)
        for _ in range(0,iterations):
            offset = 0; current_count = 0
            last_node = choice(list(network.nodes))
            if(number_of_notes!=None):
                LIMIT = number_of_notes + 1
            else:
                LIMIT = max_lenght
            while(current_count < LIMIT):
                a1=[]
                for el in network.neighbors(last_node): 
                    a1.append(el)
                if(len(a1)!=0):
                    selected = a1[randint(0,len(a1)-1)]
                    tmp_stream,offset=self.add_note(network,last_node,tmp_stream,offset)
                    last_node = selected
                if(number_of_notes!=None):
                    current_count+=1
                else:
                    dur = last_node[-1]
                    if("/"in dur):
                        dur = self.convertFracToFloat(dur)
                    current_count+=float(dur)
            pbar.update(1)
        pbar.close()
        self.Stream_list.append(tmp_stream)
        return self.Stream_list
    
    def biaisedWeightRW(self, network, iterations=5, number_of_notes=50, max_lenght = 50):
        """Do biaised random walk on the network based on the weight of the edges

        Args:
            network (NetworkX): The network to walk on
            iterations (int, optional): The number of iteration for the algorithm. Defaults to 5.
            number_of_notes (int, optional): Number of note to generate in the stream. Defaults to 50.

        Returns:
            stream: The final stream
        """
        stream1=stream.Stream()
        pbar = tqdm(total=iterations)
        for _ in range(0,iterations): 
            last_node_1 = choice(list(network.nodes)) # todo: to be selected randomnly amongst network nodes
            current_count = 0; offset1=0
            LIMIT = max_lenght
            if(number_of_notes!=None):
                LIMIT = number_of_notes + 1
            while(current_count < LIMIT):
                a1,w1=[],[]
                for el in network.neighbors(last_node_1): # for each neighbour of last played note
                    a1.append(el) # appends neighbors 
                    w1.append(network.edges[last_node_1, el]["w"])
                try:
                    selected_1=choices(a1,w1,k=1)[0] # weighted random choice
                except:
                    selected_1 = choice(list(network.nodes)) 
                stream1,offset1=self.add_note(network,last_node_1,stream1,offset1) # note added to stream
                last_node_1=selected_1 # update last note
                if(number_of_notes!=None):
                    current_count+=1
                else:
                    dur = last_node_1[-1]
                    if("/"in dur):
                        dur = self.convertFracToFloat(dur)
                    current_count+=float(dur)
            pbar.update(1)
        pbar.close()
        self.Stream_list.append(stream1)
        return self.Stream_list

    def biaisedDegreeRW(self, network, iterations=5, number_of_notes=50, max_lenght=50):
        """Baised random walk on the network based on the degree of the nodes

        Args:
            network (NetworkX): The network to walk on
            iterations (int, optional): The number of iteration for the algorithm. Defaults to 5.
            number_of_notes (int, optional): Number of note to generate in the stream. Defaults to 50.

        Returns:
            stream: The final stream
        """
        tmp_stream=stream.Stream()
        pbar = tqdm(total=iterations)
        for _ in range(0,iterations):
            offset = 0; current_count = 0
            last_node = choice(list(network.nodes))
            LIMIT = max_lenght
            if(number_of_notes!=None):
                LIMIT = number_of_notes + 1
            while(current_count < LIMIT):
                a1,w1=[],[]
                for el in network.neighbors(last_node):
                    a1.append(el)
                    w1.append(network.degree[el])
                try:
                    selected=choices(a1,w1,k=1)[0] # weighted random choice
                except:
                    selected = choice(list(network.nodes)) 
                tmp_stream,offset=self.add_note(network,last_node,tmp_stream,offset)
                last_node = selected
                if(number_of_notes!=None):
                    current_count+=1
                else:
                    dur = last_node[-1]
                    if("/"in dur):
                        dur = self.convertFracToFloat(dur)
                    current_count+=float(dur)
            pbar.update(1)
        pbar.close()
        self.Stream_list.append(tmp_stream)
        return self.Stream_list


    def ACO_RW(self, subgraphs, intergraph, alphas, betas, reputation, tau, rho, n_iter, n_notes, max_lenght, reputbool, a_L,func_list):
        paths = []; used_nodes =[]; offset_list=[]; iterations_lst = []; first_note = []
        paths.append([])
        used_nodes.append([])
        s_len=len(subgraphs)
        # Compute Edge probabilities -> Heuristic
        len_used_nodes_by_it = [[] for i in range(0, s_len)]
        pheromoned_edges = [[] for i in range(0, s_len)]
        probable_edges = [[] for i in range(0, s_len)]
        chosen_edges = [[] for i in range(0, s_len)]
        
        # Get the list of note and edge from input midi (not network needed) and update pheromone
        
        for l in range (0,s_len):
            first_note.append(self.get_first_node(subgraphs[l]))
            # Initialize Tao probabilities -> Initial Pheromone for each edge
            subgraphs[l]=self.initialize_pheromones(tau,subgraphs[l])
            # Normalize graph weights
            subgraphs[l]=self.initialize_probabilities(subgraphs[l])
            # Compute probabilities Pij
            subgraphs[l]=self.compute_probability(subgraphs[l],alphas[l],betas[l])
            self.Stream_list.append(stream.Stream())
            offset_list.append(0)   

        pbar = tqdm(total=n_iter)
        for i in range (0,n_iter):
            iterations_lst.append(i)
            paths = []; used_nodes =[]
            paths.append([])
            used_nodes.append([])
            # For each layer
            for l in range (0,s_len):
                # TODO : Except for 1 iteration and 1 layer (because we use the input live layer stream)
                paths.append([])
                used_nodes.append([])
                offset_list[l] = 0; current_count = 0
                self.Stream_list[l]=stream.Stream()
                last_node = first_note[l] # add initial note to the stream
                self.Stream_list[l],offset_list[l]=self.add_note(subgraphs[l],last_node, self.Stream_list[l],offset_list[l])
                LIMIT = max_lenght
                if(n_notes!=None):
                    LIMIT = n_notes + 1                    
                while(current_count < LIMIT):
                    a1=[]; w1=[]
                    alert=False
                    if all(False for _ in subgraphs[l].neighbors(last_node))==True: # except: if node has no neighbor
                        a1.append(choice(list(subgraphs[l].nodes)))
                        w1.append(1)
                        alert=True # to avoid adding pheromones in unexisting edges
                    else:
                        for el in subgraphs[l].neighbors(last_node):
                            a1.append(el)
                            w1.append(subgraphs[l][last_node][el]["prob"])
                        #compute probabilities
                    # weighted random choice
                    selected_1=choices(a1,w1,k=1)[0]
                    # add edge to path
                    # TODO : Get list node and edge from the input live stream
                    if alert==False: 
                        paths[l].append((last_node,selected_1))
                    used_nodes[l].append((last_node))
                    last_node=selected_1
                    self.Stream_list[l],offset_list[l]=self.add_note(subgraphs[l],last_node, self.Stream_list[l],offset_list[l])
                    # add nodes to list of used nodes
                    if(n_notes!=None):
                        current_count+=1
                    else:
                        dur = last_node[-1]
                        if("/"in dur):
                            dur = self.convertFracToFloat(dur)
                        current_count+=float(dur)
                # TODO: Introduce multiple ants iterations
            for l in range (0,s_len):
                #evaporate pheromone
                subgraphs[l]=self.evaporate_pheromone(subgraphs[l],rho)
                #deposit intralayer pheromones
                #if(len(reputbool)>0):
                #    subgraphs[l]=self.deposit_pheromone(paths[l],reputation[l],subgraphs[l], reputbool[l],self.obj_path,paths[l])
                #else:
                subgraphs[l]=self.deposit_pheromone(paths[l],reputation[l],subgraphs[l], False,func_list[l],paths[l])
                #Interlayer pheromone
                subgraphs=self.external_pheromone(used_nodes, paths[l], intergraph,subgraphs, reputation[l])
                subgraphs[l]=self.compute_probability(subgraphs[l],alphas[l],betas[l])
                nb_it = len(set(used_nodes[l])) # Plot this
                len_used_nodes_by_it[l].append(nb_it)
                pheromoned_edges[l].append(self.compute_pheromoned_edges(subgraphs[l]))
                probable_edges[l].append(self.compute_probable_edges(subgraphs[l]))
                chosen_edges[l].append(self.compute_chosen_edges(subgraphs[l],used_nodes[l]))
            pbar.update(1)
        pbar.close()
        #print(len_used_nodes_by_it)
        self.showPlot(iterations_lst, len_used_nodes_by_it, "ACO evolution")
        self.showPlot(iterations_lst,pheromoned_edges, "Edges with Pheromone")
        self.showPlot(iterations_lst, probable_edges, "Edges with Probabilities")
        self.showPlot(iterations_lst, chosen_edges, "Edges with Probability to be chosen")
        return self.Stream_list

    def multACO_RW(self, subgraphs, intergraph, alphas1,alphas2, betas, reputation, in_tau,ex_tau, rho1,rho2, n_iter, n_notes, max_lenght, reputbool,
               a_L, func_list):
        paths = [];
        used_nodes = [];
        offset_list = [];
        iterations_lst = [];
        first_note = []
        paths.append([])
        used_nodes.append([])
        s_len = len(subgraphs)
        # Compute Edge probabilities -> Heuristic
        len_used_nodes_by_it = [[] for i in range(0, s_len)]
        pheromoned_edges = [[] for i in range(0, s_len)]
        deltas = [[] for i in range(0, s_len)]
        probable_edges = [[] for i in range(0, s_len)]
        chosen_edges = [[] for i in range(0, s_len)]
        used_ext_ph= [[] for i in range(0, s_len)]
        used_in_ph = [[] for i in range(0, s_len)]
        used_w = [[] for i in range(0, s_len)]
        # Get the list of note and edge from input midi (not network needed) and update pheromone

        for l in range(0, s_len):
            #TODO: Change first note not to be a rest
            first_note.append(self.get_first_node(subgraphs[l]))
            print(first_note)
            # Initialize Tao probabilities -> Initial Pheromone for each edge
            subgraphs[l] = self.initialize_multipheromones(in_tau,ex_tau,subgraphs[l])
            # Normalize graph weights
            subgraphs[l] = self.initialize_probabilities(subgraphs[l])
            # Compute probabilities Pij
            subgraphs[l] = self.compute_probability_multi(subgraphs[l], alphas1[l], alphas2[l], betas[l])
            self.Stream_list.append(stream.Stream())
            offset_list.append(0)

        pbar = tqdm(total=n_iter)
        for i in range(0, n_iter):
            start_time = time.time()
            iterations_lst.append(i)
            paths = []
            delta = []
            delta.append([])
            used_nodes = []
            paths.append([])
            used_nodes.append([])

            # For each layer
            for l in range(0, s_len):
                prepaths = []
                #ITERATIONS PER EACH ANT

                for a in range(0, a_L):
                    prepaths.append([])
                    last_node = first_note[l]
                    LIMIT = max_lenght
                    current_count=0
                    if (n_notes != None):
                        LIMIT = n_notes + 1
                    while (current_count < LIMIT):
                        a1 = [];
                        w1 = []
                        alert = False
                        if all(False for _ in subgraphs[l].neighbors(last_node)) == True:  # except: if node has no neighbor
                            a1.append(choice(list(subgraphs[l].nodes)))
                            w1.append(1)
                            alert = True  # to avoid adding pheromones in unexisting edges
                        else:
                            for el in subgraphs[l].neighbors(last_node):
                                a1.append(el)
                                w1.append(subgraphs[l][last_node][el]["prob"])
                            # compute probabilities
                        # weighted random choice
                        selected_1 = choices(a1, w1, k=1)[0]
                        # add edge to path
                        if alert == False:
                            prepaths[a].append((last_node, selected_1))
                        last_node = selected_1
                        # add nodes to list of used nodes
                        if (n_notes != None):
                            current_count += 1
                        else:
                            dur = last_node[-1]
                            if ("/" in dur):
                                dur = self.convertFracToFloat(dur)
                            current_count += float(dur)

                #Evaluate all prepaths
                delta.append([])
                #print("--- %s ants seconds ---" % (time.time() - start_time))
                delta[l], selected_path = func_list[l](prepaths) #return best path
                #print("--- %s selecting best ant seconds ---" % (time.time() - start_time))
                #Add selected prepath to path
                paths.append([])
                used_nodes.append([])
                paths[l]=selected_path
                offset_list[l] = 0;
                self.Stream_list[l] = stream.Stream()
                for i in range(0, len(paths[l])):
                    last_node = paths[l][i][0]
                    used_nodes[l].append((last_node))
                    self.Stream_list[l], offset_list[l] = self.add_note(subgraphs[l], last_node, self.Stream_list[l],
                                                                    offset_list[l])
            for l in range(0, s_len):
                used_ext_ph[l].append(self.measure_used_att(paths[l],subgraphs[l],l,"ex_tau"))
                used_in_ph[l].append(self.measure_used_att(paths[l],subgraphs[l],l,"in_tau"))
                used_w[l].append(self.measure_used_att(paths[l], subgraphs[l],l,"w"))
                # evaporate pheromone
            for l in range(0, s_len):
                subgraphs[l] = self.evaporate_multipheromone(subgraphs[l], rho1,rho2)

                # deposit intralayer pheromones
                # if(len(reputbool)>0):
                #    subgraphs[l]=self.deposit_pheromone(paths[l],reputation[l],subgraphs[l], reputbool[l],self.obj_path,paths[l])
                # else:
            #print("--- %s evaporate pheromone ---" % (time.time() - start_time))
            for l in range (0,s_len):

                delta_f, subgraphs[l] = self.deposit_multipheromone(paths[l], reputation[l], subgraphs[l], delta[l])
                #print("--- %s deposit intra pheromone ---" % (time.time() - start_time))
                # Interlayer pheromone
                if delta_f>0:
                    subgraphs = self.external_multipheromone(used_nodes[l], paths[l], intergraph, subgraphs, reputation[l],delta_f)
                #print("--- %s deposit external pheromone ---" % (time.time() - start_time))
                deltas[l].append(delta_f)

            for l in range(0, s_len):
                subgraphs[l] = self.compute_probability_multi(subgraphs[l], alphas1[l], alphas2[l], betas[l])
                nb_it = len(set(used_nodes[l]))  # Plot this
                len_used_nodes_by_it[l].append(nb_it)
                #pheromoned_edges[l].append(self.compute_pheromoned_edges(subgraphs[l]))
                #probable_edges[l].append(self.compute_probable_edges(subgraphs[l]))
                #chosen_edges[l].append(self.compute_chosen_edges(subgraphs[l], used_nodes[l]))
            #print("--- %s compute prob ---" % (time.time() - start_time))



            #interaction_idx = self.compute_interactions(paths, intergraph)
            #print('Interaction:'+str(interaction_idx))
            pbar.update(1)
        pbar.close()
        # print(len_used_nodes_by_it)

        self.showPlot(iterations_lst, len_used_nodes_by_it, "ACO evolution", "Number of different nodes in the path")
        #self.showPlot(iterations_lst, pheromoned_edges, "Edges with Pheromone")
        #self.showPlot(iterations_lst, probable_edges, "Edges with Probabilities")
        #self.showPlot(iterations_lst, chosen_edges, "Edges with Probability to be chosen")
        self.showPlot(iterations_lst, deltas, "Objective Functions","Goal")
        self.showPlot(iterations_lst, used_ext_ph, "Use of External Pheromone","External Pheromone in Solution")
        self.showPlot(iterations_lst, used_in_ph, "Use of Internal Pheromone", "Internal Pheromone in Solution")
        self.showPlot(iterations_lst, used_w, "Use of Heuristic Information ", "Heuristic Information in Solution")

        return self.Stream_list
    
    # Use the live input to initialize the random walk
    def ACO_RW_LIVE(self, subgraphs, intergraph, lst_livenote, alphas=[], betas=[], reputation=[], tau=1, rho=0.2, n_iter=500, n_notes=200, max_lenght=50, reputbool=False, a_L=1):
        paths = []; used_nodes =[]; offset_list=[]; iterations_lst = []; first_note = []
        paths.append([])
        used_nodes.append([])
        s_len=len(subgraphs)
        # Compute Edge probabilities -> Heuristic
        len_used_nodes_by_it = [[] for i in range(0, s_len)]
        pheromoned_edges = [[] for i in range(0, s_len)]
        probable_edges = [[] for i in range(0, s_len)]
        chosen_edges = [[] for i in range(0, s_len)]
        
        # Get the list of note and edge from input midi (not network needed) and update pheromone
        
        for l in range (0,s_len):
            first_note.append(self.get_first_node(subgraphs[l]))
            # Initialize Tao probabilities -> Initial Pheromone for each edge
            subgraphs[l]=self.initialize_pheromones(tau,subgraphs[l])
            # Normalize graph weights
            subgraphs[l]=self.initialize_probabilities(subgraphs[l])
            # Compute probabilities Pij
            subgraphs[l]=self.compute_probability(subgraphs[l],alphas[l],betas[l])
            self.Stream_list.append(stream.Stream())
            offset_list.append(0)   

        print("Live note:", lst_livenote)
    
        pbar = tqdm(total=n_iter)
        for i in range (0,n_iter):
            iterations_lst.append(i)
            paths = []; used_nodes =[]
            paths.append([])
            used_nodes.append([])

            for l in range (0,s_len): # For each layer
                paths.append([])
                used_nodes.append([]) 
                offset_list[l] = 0; current_count = 0

                last_node = lst_livenote[0]
                last_node = (l, last_node[1], last_node[2])

                self.Stream_list[l]=stream.Stream()
                self.Stream_list[l],offset_list[l]=self.add_note(subgraphs[l],last_node, self.Stream_list[l],offset_list[l])
                LIMIT = max_lenght
                if(n_notes!=None):
                    LIMIT = n_notes + 1                    
                while(current_count < LIMIT):
                    a1=[]; w1=[]
                    alert=False
                    if all(False for _ in subgraphs[l].neighbors(last_node))==True: # except: if node has no neighbor
                        if(l==0): #for the first layer we use the live input
                            a1.append(choice(list(lst_livenote)))
                        else:
                            a1.append(choice(list(subgraphs[l].nodes)))
                        w1.append(1)
                        alert=True # to avoid adding pheromones in unexisting edges
                    else:
                        for el in subgraphs[l].neighbors(last_node):
                            a1.append(el)
                            w1.append(subgraphs[l][last_node][el]["prob"])
                    selected_1=choices(a1,w1,k=1)[0]                    
                    if alert==False: 
                        paths[l].append((last_node,selected_1))
                    used_nodes[l].append((last_node))
                    last_node=selected_1
                    self.Stream_list[l],offset_list[l]=self.add_note(subgraphs[l],last_node, self.Stream_list[l],offset_list[l])
                    # add nodes to list of used nodes
                    if(n_notes!=None):
                        current_count+=1
                    else:
                        dur = last_node[-1]
                        if("/"in dur):
                            dur = self.convertFracToFloat(dur)
                        current_count+=float(dur)
                        
            for l in range (0,s_len):
                #evaporate pheromone
                subgraphs[l]=self.evaporate_pheromone(subgraphs[l],rho)
                #deposit intralayer pheromones
                if(len(reputbool)>0):
                    subgraphs[l]=self.deposit_pheromone(paths[l],reputation[l],subgraphs[l], reputbool[l])
                else:
                    subgraphs[l]=self.deposit_pheromone(paths[l],reputation[l],subgraphs[l], False)
                #Interlayer pheromone
                subgraphs=self.external_pheromone(used_nodes, paths[l], intergraph,subgraphs, reputation[l])
                subgraphs[l]=self.compute_probability(subgraphs[l],alphas[l],betas[l])
                nb_it = len(set(used_nodes[l])) # Plot this
                len_used_nodes_by_it[l].append(nb_it)
                #pheromoned_edges[l].append(self.compute_pheromoned_edges(subgraphs[l]))
                #probable_edges[l].append(self.compute_probable_edges(subgraphs[l]))
                #chosen_edges[l].append(self.compute_chosen_edges(subgraphs[l],used_nodes[l]))
            pbar.update(1)
        pbar.close()
        #print(len_used_nodes_by_it)
        #self.showPlot(iterations_lst, len_used_nodes_by_it, "ACO evolution")
        #self.showPlot(iterations_lst,pheromoned_edges, "Edges with Pheromone")
        #self.showPlot(iterations_lst, probable_edges, "Edges with Probabilities")
        #self.showPlot(iterations_lst, chosen_edges, "Edges with Probability to be chosen")
        return self.Stream_list

    def get_first_node(self, n):
        s={}
        for x in n.nodes():
            if 'rest' not in x:
                s[x]=n.nodes()[x]["t"]
        return min(s, key=s.get)    
    
    def convertFracToFloat(self, frac):
        """Convert a fraction to a float

        Args:
            frac (String): A fraction in the form of a string

        Returns:
            float: The float value of the fraction
        """
        return float(frac.split("/")[0])/float(frac.split("/")[1])
    
    def add_note(self, G, x, s, offset): 
        """ Adds note to stream and returns stream and updated offset

        Args:
            G (NetworkX): The network
            x (_type_): _description_
            s (_type_): _description_
            offset (_type_): _description_

        Returns:
            Multiple: Returns stream and updated offset
        """
        pitch= G.nodes[x]['pitch'] # gets pitch from node attribute
        dur= G.nodes[x]['dur'] # gets note duration from node attribute
        split=str.split(pitch,' ')
        if len(split)>2: #CHORD
            ch=' '.join(split[0:len(split)-1])
            f = chord.Chord(ch) # defines chord object
            f.quarterLength=self.convert(dur) # defines note duration
        else:
            if pitch=='rest':
                f= note.Rest()
            else:
                f = note.Note(pitch)
            f.quarterLength=self.convert(dur)
        f.offset = offset
        if(f.quarterLength == None):
            f.quarterLength = 0
        offset=f.offset+f.quarterLength
        s.append(f)
        return s, offset # returns stream and updated offset

    def convert(self, s): 
        """ Computes irrational durations

        Args:
            s (String): The irrational duration (fraction)

        Returns:
            Multiple: float or numerator & demoninator of the fraction
        """
        try:
            return float(s)
        except ValueError:
            num, denom = s.split('/')
    
    def play(self):
        """Play the generated midi file
        """
        self.Stream_list[0].show('midi')
    
    def export(self, outpath, name):
        """Export the stream to a midi file

        Args:
            outpath (string): The path to the output file
        """
        print("[+] Writing "+ str(len(self.Stream_list)) +" midi files to : " + outpath)
        for i in range(0,len(self.Stream_list)):
            filename = outpath + name.replace(" ","_") + "_" + str(i+1) + "_monophonic.mid"
            self.Stream_list[i].write('midi', filename)    
    
    def exportPolyphonic(self, outpath, name):
        s_len = len(self.Stream_list)
        s = stream.Score(id='mainScore')
        filename = outpath + name.replace(" ","_") + "_" + "polyphonic.mid"
        for l in range (0,s_len):
            m = stream.Measure(number=l)
            m=self.Stream_list[l]
            p0 = stream.Part(id='part0')
            p0.append(m)
            s.insert(0, p0)
        for p in s.parts :
            p.insert(0,instrument.Banjo())
        s.write('midi', filename)
        
        
    # ACO
    def initialize_pheromones(self, tau,n):
        nx.set_edge_attributes(n, tau, "tau")
        return n

    def initialize_multipheromones(self,in_tau,ex_tau,n):
        nx.set_edge_attributes(n, in_tau, "in_tau")
        nx.set_edge_attributes(n, ex_tau, "ex_tau")
        return n

    # ACO
    def initialize_probabilities(self, n):
        for x in n.nodes():
            tot=0
            for i in nx.neighbors(n,x):
                tot+=(n.edges[x, i]["w"])
            for s in nx.neighbors(n,x):
                n.edges[x, s]["w"]=n.edges[x, s]["w"]/tot
        return n

    # ACO
    def compute_probability(self, n,alpha,beta):
        d1=nx.get_edge_attributes(n, "tau")
        d2=nx.get_edge_attributes(n, "w")
        d3={}
        for k in d1:
            d3[k]=np.multiply(d1[k]**alpha,d2[k]**beta)
        nx.set_edge_attributes(n,d3, "probaux")
        #DONE
        for x in n.nodes():
            tot=0
            for i in nx.neighbors(n,x):
                tot+=(n.edges[x, i]["probaux"])
            for s in nx.neighbors(n,x):
                n.edges[x, s]["prob"]=(n.edges[x, s]["probaux"]/tot)
        return n
    def compute_probability_multi(self, n,alpha1,alpha2,beta):
        d1=nx.get_edge_attributes(n, "in_tau")
        d11=nx.get_edge_attributes(n, "ex_tau")
        d2=nx.get_edge_attributes(n, "w")
        d3={}
        for k in d1:
            d3[k]=(d1[k]**alpha1)*(d11[k]**alpha2)*(d2[k]**beta)
        nx.set_edge_attributes(n,d3, "probaux")
        #DONE
        for x in n.nodes():
            tot=0
            for i in nx.neighbors(n,x):
                tot+=(n.edges[x, i]["probaux"])
            for s in nx.neighbors(n,x):
                n.edges[x, s]["prob"]=(n.edges[x, s]["probaux"]/tot)
        return n


    # ACO
    def evaporate_pheromone(self, N,rho):
        d=nx.get_edge_attributes(N, "tau")
        for ij in d:
            d[ij]*=(1-rho)
        nx.set_edge_attributes(N, d, "tau")
            #  (1-rho)·tau_ij
            # self.pheromones = (1 - self.rho) * self.pheromones
        return N

    def evaporate_multipheromone(self,N,rho1,rho2):
        d1 = nx.get_edge_attributes(N, "in_tau")
        d11 = nx.get_edge_attributes(N, "ex_tau")
        for ij in d1:
            d1[ij]*=(1-rho1)
            d11[ij]*=(1-rho2)
        nx.set_edge_attributes(N, d1, "in_tau")
        nx.set_edge_attributes(N, d11, "ex_tau")
        return N

    # ACO Objective Functions

    def obj_cst(self,prepaths):
        delta=1/2
        sel=0
        return delta, prepaths[sel]

    def obj_loop(self,prepaths):
        delta=0
        sel=0
        for i in range(0,len(prepaths)):
            if len(set(prepaths[i]))==8:
                temp_delta=1
            else:
                temp_delta=max(0,1-(abs(len(set(prepaths[i]))-8)/8))
            if temp_delta>delta:
                delta=temp_delta
                sel=i
        print('delta loop:' + str(delta))
        return delta, prepaths[sel]

    def obj_path_32(self, prepaths):
        delta=0
        sel=0
        length=32
        for i in range (0,len(prepaths)):
            if len(set(prepaths[i]))==length:
                temp_delta=1
            else:
                x = abs(len(set(prepaths[i]))-length)
                temp_delta=1-(x*1/len(prepaths[i]))
            if temp_delta>delta:
                delta=temp_delta
                sel=i
                if delta==1:
                    break
        print('delta path:'+str(delta))
        return delta, prepaths[sel]

    def obj_path_4(self, prepaths):
        delta=0
        sel=0
        length=4
        for i in range (0,len(prepaths)):
            if len(set(prepaths[i]))==length:
                temp_delta=1
            else:
                x = abs(len(set(prepaths[i]))-length)
                temp_delta=1-(x*1/len(prepaths[i]))
            if temp_delta>delta:
                delta=temp_delta
                sel=i
                if delta==1:
                    break
        print('delta path:'+str(delta))
        return delta, prepaths[sel]

    def obj_path_16(self, prepaths):
        delta=0
        sel=0
        length=16
        for i in range (0,len(prepaths)):
            if len(set(prepaths[i]))==length:
                temp_delta=1
            else:
                x = abs(len(set(prepaths[i]))-length)
                temp_delta=1-(x*1/len(prepaths[i]))
            if temp_delta>delta:
                delta=temp_delta
                sel=i
                if delta==1:
                    break
        print('delta path:'+str(delta))
        return delta, prepaths[sel]

    def obj_path_8(self, prepaths):
        delta=0
        sel=0
        length=8
        for i in range (0,len(prepaths)):
            if len(set(prepaths[i]))==length:
                temp_delta=1
            else:
                x = abs(len(set(prepaths[i]))-length)
                temp_delta=1-(x*1/len(prepaths[i]))
            if temp_delta>delta:
                delta=temp_delta
                sel=i
                if delta==1:
                    break
        print('delta path:'+str(delta))
        return delta, prepaths[sel]

    def obj_rhythm(self,prepaths):
        delta = 0
        temp_delta = 0
        sel = 0
        for i in range(0, len(prepaths)):
            rhythm_list, rhythm_goal = [], []
            for x in range(0, len(prepaths[i])):
                rhythm_list.append(prepaths[i][x][0][-1])
                rhythm_goal.append('1.0')
            for s in range(0, len(prepaths[i])):
                if rhythm_list[s] == rhythm_goal[s]:
                    temp_delta += 0.1
            if temp_delta > delta:
                delta = temp_delta
                sel = i
        print('delta rhythm:' + str(delta))

        return delta, prepaths[sel]

    def obj_rhythm_2(self,prepaths):
        delta = 0
        sel = 0
        max_val=0
        for i in range(0, len(prepaths)):
            d={}
            for el in set(prepaths[i]):
                dur_not=el[0][-1]
                if dur_not not in d:
                    d[dur_not] = 0
                d[dur_not]+=1
            d2=d.values()
            x=max(d2)
            if x>max_val:
                max_val=x
                delta= max_val/len(prepaths[i])
                sel=i
        print(delta)
        return delta, prepaths[sel]


    def obj_loop_2(self,prepaths):
        delta, sel,temp_delta=0,0,0
        temp_dur=0.0
        duration_target=8.0
        for i in range (0,len(prepaths)):
            target = prepaths[i][0][0]
            j=0
            while 'rest' in target and j+1<len(prepaths[i]):
                j+=1
                target = prepaths[i][j][0]
            if 'rest' in target:
                break
            for x in range(j+1, len(prepaths[i])):
                dur = prepaths[i][x][0][-1]
                if ("/" in dur):
                    dur = self.convertFracToFloat(dur)
                temp_dur+=float(dur)
                if prepaths[i][x][0]==target:
                    temp_delta= 1- (abs(duration_target-temp_dur) % 8)
                if temp_delta > delta:
                    delta = temp_delta
                    sel = i
                if delta==1:
                    break
            break
        print('delta loop2:' + str(delta))
        return delta, prepaths[sel]




    def deposit_pheromone(self, path, rep, N, cnst,obj_fun,*args):
        if obj_fun==self.obj_cst:
            delta = obj_fun()
        else:
            delta = obj_fun(*args)
        print(delta)
        #if cnst==True:
        #    delta = 1 /rep
        #else:
        #    delta = len(set(path)) / len(path)
        #delta = len(set(path)) / len(path)
        #delta = 1 /rep #TODO change Lk
        d=nx.get_edge_attributes(N, "tau")
        for ij in set(path):
            d[ij]+=delta

        nx.set_edge_attributes(N, d, "tau")
        return delta, N

    def deposit_multipheromone(self, path, rep, N, delta):
        #if cnst==True:
        #    delta = 1 /rep
        #else:
        #    delta = len(set(path)) / len(path)
        #delta = len(set(path)) / len(path)
        #delta = 1 /rep #TODO change Lk
        d=nx.get_edge_attributes(N, "in_tau")
        for ij in set(path):
            d[ij]+=delta
            d[ij] = min(d[ij], 1)

        nx.set_edge_attributes(N, d, "in_tau")
        return delta, N

    # ACO
    def external_pheromone (self, used_nodes, path, intergraph,subgraphs, reputation):
        delta = 1 /reputation #TODO change Lk
        #delta = len(set(path)) / len(path)
        for node in used_nodes:
            if node in intergraph:
                for el in intergraph.neighbors(node):
                    layer= intergraph.nodes[el]['l']
                    d=nx.get_edge_attributes(subgraphs[layer], "tau")
                    for x in subgraphs[layer].in_edges(el):
                        d[x]+=delta
                    nx.set_edge_attributes(subgraphs[layer], d, "tau")
        return subgraphs

    def external_multipheromone (self, used_nodes, path, intergraph,subgraphs, reputation,delta_f):
        #delta = 1 /reputation
        factor=len(subgraphs)
        #delta = len(set(path)) / len(path)
        for node in set(used_nodes):
            if node in intergraph:
                for el in intergraph.neighbors(node):
                    layer= intergraph.nodes[el]['l']
                    d=nx.get_edge_attributes(subgraphs[layer], "ex_tau")
                    for x in subgraphs[layer].in_edges(el):
                        d[x]+=delta_f/10*factor
                        d[x]= min(d[x],1)
                    nx.set_edge_attributes(subgraphs[layer], d, "ex_tau")
        return subgraphs


    def measure_used_att(self,paths,subgraph,l,att):
        #print('layer: '+str(l))
        used_pheromone=0
        d = nx.get_edge_attributes(subgraph, att)
        for i in paths:
            #print(d[i])
            x=d[i]
            used_pheromone+=x
        return used_pheromone

    def compute_interactions(self,paths,intergraph):
        tot=0
        coup=0
        for j in range(0,len(paths)): # for each path
            for i in range(0,len(paths[j])): # for each note in path
                nod=paths[j][i][0] # store node
                tot+=1
                for s in range (0,len(paths)): # for each other path
                    for t in range (0,len(paths[s])): #for each note
                        nod2=paths[s][t][0] # save node
                        if nod2 in intergraph and nod in intergraph.neighbors(nod2):
                            coup+=1
                            break
        print("Interconnected:"+str(coup))
        print("Total:"+str(tot))
        return coup/tot



    def compute_pheromoned_edges(self, n):
        d = nx.get_edge_attributes(n, "tau")
        n_edges=0
        for i in d:
            if d[i]>0.5:
                n_edges+=1
        return n_edges

    def compute_probable_edges(self, n):
        d = nx.get_edge_attributes(n, "prob")
        n_edges = 0
        for i in d:
            if d[i] > 0.5:
                n_edges += 1
        return n_edges

    def compute_chosen_edges(self, n,used_nodes):
        d = nx.get_edge_attributes(n, "prob")
        n_edges = 0
        for node in used_nodes:
            n_edges_node = 0
            for x in n.out_edges(node):
                if d[x] > 0.1:
                    n_edges_node += 1
            if n_edges_node > 1:
                    n_edges+=1
        return n_edges


    def showPlot(self, lst_X, lst_Y, title,ylabel):
        """Plot a bar chart

        Args:
            lst_x (lst): List of the x axis
            lst_y (lst): List of the y axis
            title (str): Title of the chart
        """
        fig = plt.figure(figsize = (15, 8))
        for i in range(0,len(lst_Y)):
            plt.plot(lst_X, lst_Y[i], alpha=0.5, label="Layer"+str(i))
        plt.ylabel(ylabel)
        plt.xlabel("Iterations")
        plt.xticks(rotation = 45, fontsize=6) 
        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.subplots_adjust(right=0.7)
        name =  self.outfolder + "ACO_RW_" + title.replace(" ", "_") + ".eps"
        try:
            plt.savefig(name,format='eps')
        except:
            plt.show()
            print("[-] Error - Unable to save the chart")
        plt.close("all")

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))