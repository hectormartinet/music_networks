from multilayer_net import MultiLayerNetwork
import os
import sys
import random as rd

def rand_bool():
    return bool(rd.randint(0,1))

# TODO add all new parameters
def randomize_args():
    args = {}
    args["rest"] = rand_bool()
    args["octave"] = rand_bool()
    args["pitch"] = rand_bool()
    args["duration"] = rand_bool()
    args["offset"] = rand_bool()
    args["enharmony"] = rand_bool()
    args["offset_period"] = rd.randint(1,16)/4
    args["transpose"] = rand_bool()
    args["strict_link"] = rand_bool()
    args["layer"] = rand_bool()
    args["flatten"] = rand_bool()
    args["chromatic_interval"] = rand_bool()
    args["diatonic_interval"] = rand_bool()
    args["max_link_time_diff"] = rd.randint(1,16)/4
    return args

if __name__ == "__main__" :
    input_file_path = 'midis/invent_bach/invent1.mid'  # Replace with your MIDI file path
    output_folder = 'random_tests'  # Replace with your desired output folder

    n=10
    if len(sys.argv) > 1:
        n=int(sys.argv[1])
    for i in range(n):

        args = randomize_args()
        try :
            # Create the MultiLayerNetwork object with the MIDI file and output folder
            net = MultiLayerNetwork(use_gui=False, midi_folder_or_file=input_file_path, output_folder=output_folder, **args, verbosity=0)
            net.create_net()
            net.export_nets(types='all')
        
        except Exception as e:
            print(e)
            print(args)

