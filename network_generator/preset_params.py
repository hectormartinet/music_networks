# All defaults parameters from different papers

# Liu et al.
liu_params = {
    "pitch":True,
    "octave":True,
    "duration":True,
    "enharmony":True
}

# Ferreti
ferreti_params = {
    "rest":True,
    "octave":True,
    "pitch":True,
    "duration":True,
    "enharmony":True # Not explicit
}

# Gomez et al.
# Post-processing to undirected graph
gomez_params = {
    "pitch":True,
    "flatten":True,
    "enharmony": False
}

# Serra et al.
# Missing timbre and loudness
serra_params = {
    "pitch":True,
    "group_by_beat":True,
    "enharmony":True
}

# Nardelli et al.
# Post-processing to remove PCsets with weight under a given threshold
nardelli_params = {
    "pitch":True,
    "flatten":True,
    "enharmony": False
}

# Kulkarni et al.
# Post processing needed to split chords and relink nodes (+ remove self-loops)
kulkarni_params = {
    "pitch":True,
    "octave":True,
    "enharmony":True, # I guess ?
}

# Rolla et al.
# Different parts need to be merged together and each node contains multiple notes with their duration
rolla_params = {
    "pitch":True,
    "octave":True,
    "enharmony":True
}

# Perkins et al.
perkins_params = {
    "pitch":True,
    "rest":True,
    "enharmony":True
}

# Frottier et al.
# Use human annoted partitions
frottier_params = {
    "chord_function":True,
    "flatten":True
}

# Mrad et al.
# Two networks :

# Melodic 
# Missing duration weighted network
mrad_melodic_params = {
    "pitch":True,
    "enharmony":True
}

# Harmonic
# Restrict chords to roman numerals + augmented/diminished symbols
mrad_harmonic_params = {
    "flatten":True,
    "chord_function":True
}

preset_params = {
    "liu" : liu_params,
    "ferreti" : ferreti_params,
    "gomez" : gomez_params,
    "serra" : serra_params,
    "nardelli" : nardelli_params,
    "kulkarni" : kulkarni_params,
    "rolla" : rolla_params,
    "perkins" : perkins_params,
    "frottier" : frottier_params,
    "mrad_melodic" : mrad_melodic_params,
    "mrad_harmonic" : mrad_harmonic_params
}

def get_preset_params(name):
    return preset_params[name]