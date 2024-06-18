from multilayer_net import MultiLayerNetwork
from multi_order import PrefixTree
from ortools.linear_solver import pywraplp
from multi_order import duration_to_note
import json
import numpy as np

if __name__ == "__main__":

    # Parameters
    C = 5
    entry_cost_function = lambda x:x+1
    L = 10
    t = 5

    # Build network
    # midi_file = "midis\\invent_bach\\invent14.mid"
    midi_file = "datasets\\mozart_sonatas\\K545-3.xml"
    # midi_file = "datasets\\mozart_sonatas\\K331-1.xml"
    net = MultiLayerNetwork(use_gui=False, enharmony=False, pitch=False, structure="monolayer", diatonic_interval=True, duration=True, rest=True, midi_files=[midi_file], offset=False, offset_period=1.)
    net.load_new_midi(midi_file)
    nodes_lst = net._get_nodes_list(0)
    nodes_lst = [json.loads(elt) for elt in nodes_lst]
    nodes_lst = [duration_to_note(elt["duration"],elt["rest"]) for elt in nodes_lst]
    edges_lst = net._get_edges_list(0)
    edges_lst = [elt[22:-1] for elt in edges_lst]
    sequence = []
    for i in range(len(edges_lst)):
        sequence.append(nodes_lst[i])
        # sequence.append(edges_lst[i])
    sequence.append(nodes_lst[-1])
    print("sequence length:",len(sequence))

    # Build PrefixTree
    tree = PrefixTree()
    tree.add_nodes(sequence, L)
    for child in tree.children.values():
        child.weight_pruning(t)

    # Instatiate Solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.set_time_limit(60000)
    # if not solver:
    #     return
    # Create variables
    
    
    # y[node] = is this node used ?
    y = {}
    y_length = {}
    subtrees = tree.get_tree_list()
    nodes = [(tuple(subtree.rebuild_sequence()),subtree.depth) for subtree in subtrees if subtree.depth!=0]
    for node,length in nodes:
        y[node] = solver.BoolVar(str(node))
        y_length[node] = length
    
    print("number of distinct elements:",len(y))
    # x[i,l] = is word ending at i with length l used ?
    x = {}
    for i in range(len(sequence)):
        for l in range(1,min(i+2,L+1)):
            node = tuple(sequence[i-l+1:i+1])
            if node in y:
                x[i,l] = solver.BoolVar(f"x{i},{l}")
                solver.Add(y[node] >= x[i,l])

    solver.SetHint([elt for elt in x.values()], [key[1]==1 for key in x.keys()])
    solver.SetHint([elt for elt in y.values()], [len(key)==1 for key in y.keys()])
    print("number of elements:",len(x))
    # Add constraints
    # Exaclty one node associated per note
    for i in range(len(sequence)):
        solver.Add(
            solver.Sum(
                [ x[j,l]
                    for j in range(i, min(len(sequence), i+L))
                    for l in range(j-i+1, min(j+2, L+1))
                if (j,l) in x] 
            ) == 1
        )

    # Define objective components

    length = solver.Sum([elt for elt in x.values()])

    alphabet_cost = solver.Sum([entry_cost_function(y_length[node])*y[node] for node in y.keys()])

    objective = length + alphabet_cost

    solver.Minimize(objective)
    status = solver.Solve()
    print(f"Status = {status}")
    print(f"length = {length.solution_value()}")
    print(f"alphabet_cost = {alphabet_cost.solution_value()}")
    print(f"objective = {objective.solution_value()}")
    print(f"Best bound = {solver.Objective().BestBound()}")
    used_nodes = {}
    for node,value in y.items(): 
        if value.solution_value() == 1.0:
            used_nodes[node] = 0
    new_sequence = []
    for key,value in x.items():
        if value.solution_value() == 1.0:
            i,l = key
            node = tuple(sequence[i-l+1:i+1])
            new_sequence.append(node)
            used_nodes[node] += 1
            # print(node)
    for node,value in sorted(used_nodes.items(), key=lambda x: -x[1]): 
        print(f"weight: {value}, pattern:", ",".join([str(elt) for elt in node]))