from multilayer_net import MultiLayerNetwork

class DemoMultilayer:
    def __init__(self, midi_file, outfolder) -> None:
        self.midi_file = midi_file
        self.outfolder = outfolder

    def run(self, **kwargs):
        net = MultiLayerNetwork(use_gui=False, music_files = [self.midi_file], outfolder=self.outfolder, **kwargs)
        net.create_net(separate_graphs=True, output_txt=True, parallel=False)
        net.export_nets()



if __name__ == "__main__":
    demo = DemoMultilayer(
        midi_file="midis\\mozart_sonata.xml",
        outfolder="results\\demos\\"    
    )        
    # demo.run(name="pitch_class", pitch=True)
    # demo.run(name="pitch", pitch=True, octave=True)
    # demo.run(name="duration", pitch=False, duration=True)
    # demo.run(name="offset_1", pitch=False, offset=True, beat_duration=1.)
    # demo.run(name="offset_2", pitch=False, offset=True, beat_duration=2.)
    # demo.run(name="chromatic_interval", pitch=False, order=2, chromatic_interval=True)
    # demo.run(name="diatonic_interval", pitch=False, order=2, diatonic_interval=True, enharmony=False)
    # demo.run(name="chords", pitch=True, structure="chordify")
    demo.run(name="chordify_by_beat", pitch=True, chordify_per_beat=True)

    # demo.run(name="pitch_duration", pitch=True, octave=True, duration=True, rest=True)

    # demo.run(name="duration_n2", pitch=False, duration=True, order=2)
    # demo.run(name="duration_n3", pitch=False, duration=True, order=3)
    # demo.run(name="duration_n4", pitch=False, duration=True, order=4)

    # demo.run(name="duration_grouped_by_beat", pitch=False, duration=True, group_by_beat=True, beat_duration=1, rest=True)
    # demo.run(name="duration_grouped_by_measure", pitch=False, duration=True, group_by_beat=True, beat_duration=2, rest=True)

    # demo.run(name="ultimate", pitch=False, duration=True, group_by_beat=True, beat_duration=1, rest=True, diatonic_interval=True, enharmony=False)

    # demo.run(name="offset_duration", pitch=False, duration=True, offset=True, rest=True, beat_duration=2)
    # demo.run(name="offset_duration_interval", pitch=False, duration=True, offset=True, rest=True, enharmony=False, diatonic_interval=True, order=2)