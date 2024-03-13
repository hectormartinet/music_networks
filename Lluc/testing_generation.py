import multilayernet2 as multi
import musicgenerator as mus
import utility as ut
from tqdm import tqdm
import math
import random
from music21 import *
import networkx as nx
import pandas as pd
import numpy as np
import os

# Input MIDI file path and output folder
input_file_path = 'midis/bwv772.mid'  # Replace with your MIDI file path
output_folder = '2023_embeddings_2/20th/single_output'  # Replace with your desired output folder

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the MusicGenerator object (assuming it's necessary for your setup)
mus1 = mus.MusicGenerator(output_folder)

# Check if the file has a .mid or .midi extension
if input_file_path.endswith('.mid') or input_file_path.endswith('.mxl'):
    print(f"Processing {input_file_path}")

    # Create the MultiLayerNetwork object with the MIDI file and output folder
    net1 = multi.MultiLayerNetwork(input_file_path, output_folder)

    # Call createNet function
    net1.createNet(False)

    # Get the network
    n = net1.getNet()

    # Convert numpy arrays to lists for export
    net1.convert_numpy_arrays_to_lists()

    # Get the subnet and intergraph
    net1.lstSubNet, net1.interGraph = net1.getSubNet()

    # Check if the number of subnets is 2
    # Take into account that generated midi file in that sense should have 2 channels.
    # You can try with midis with only one channel as well.
    if len(net1.lstSubNet) == 2:
        # Derive the output filename from the input MIDI filename
        output_filename = os.path.splitext(os.path.basename(input_file_path))[0] + '.graphml'

        # Extract the name without extension
        name_without_extension = os.path.splitext(output_filename)[0]

        # Construct the path for the new subfolder
        subfolder_path = os.path.join(output_folder, name_without_extension)

        # Check if the subfolder exists; if not, create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Export the multilayer network
        net1.exportNet(os.path.join(subfolder_path, output_filename))

        # Export the subnet
        net1.exportSubNet(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file_path))[0]) + os.path.sep, os.path.splitext(os.path.basename(input_file_path))[0])
    else:
        print(f"Skipped exporting for {os.path.basename(input_file_path)} because the number of subnets is not 2.")
else:
    print(f"File {input_file_path} is not a MIDI file.")

print("\nProcessing complete!")