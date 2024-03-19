from multilayer_net import MultiLayerNetwork
import os
import sys
import random as rd

def rand_bool():
    return bool(rd.randint(0,1))

def randomize_args():
    args = {}
    args["rest"] = rand_bool()
    args["stop_at_unwanted"] = rand_bool()
    args["octave"] = rand_bool()
    args["pitch"] = rand_bool()
    args["duration"] = rand_bool()
    args["offset"] = rand_bool()
    args["offset_period"] = rd.randint(1,16)/4
    args["transpose"] = rand_bool()
    args["strict_link"] = rand_bool()
    args["layer"] = rand_bool()
    args["interval"] = rand_bool()
    args["diatonic"] = rand_bool()
    return args

if __name__ == "__main__" :
    input_file_path = 'midis/invent1.mid'  # Replace with your MIDI file path
    output_folder = 'tests_temp'  # Replace with your desired output folder

    n=10
    if len(sys.argv) > 1:
        n=int(sys.argv[1])
    print(n)
    for i in range(n):
        # Create the MultiLayerNetwork object with the MIDI file and output folder
        net1 = MultiLayerNetwork(input_file_path, output_folder, **randomize_args())

        # Call createNet function
        net1.create_net()

        # Get the subnet and intergraph
        net1.get_sub_net()

        # Derive the output filename from the input MIDI filename
        output_filename = os.path.splitext(os.path.basename(input_file_path))[0] + '.graphml'

        # Extract the name without extension
        name_without_extension = os.path.splitext(output_filename)[0]

        # Construct the path for the new subfolder
        subfolder_path = os.path.join(output_folder, name_without_extension)

        # Check if the subfolder exists; if not, create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        net1.convert_attributes_to_str()

        # Export the multilayer network
        net1.export_net(os.path.join(subfolder_path, output_filename))

        # Export subnets
        net1.export_sub_net(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file_path))[0]) + os.path.sep, os.path.splitext(os.path.basename(input_file_path))[0])
    
    print("Random tests passed")

