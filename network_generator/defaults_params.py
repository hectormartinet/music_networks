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
# Missing grouping notes by beat
serra_params = {
    "pitch":True,
    "enharmony":True
}

# Nardelli et al.
# Post-processing to remove PCsets with weight under a given threshold
nardelli_params = {
    "pitch":True,
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
# Different parts need to be merged together and each node contains multiple notes with its duration
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

# Mrad et al.
# Two networks :

# Melodic 
# Missing duration weighted network
mrad_melodic_params = {
    "pitch":True,
    "enharmony":True
}

# Harmonic
# Missing functional analysis