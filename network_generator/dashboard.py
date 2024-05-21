import streamlit as st
from multi_order import SuffixTree
import os
import numpy as np

def load_tree():
    if st.session_state.tree_name != st.session_state.filename:
        st.session_state.tree_name = st.session_state.filename
        st.session_state.tree = SuffixTree("trees\\" + st.session_state.filename)

st.set_page_config(layout="wide")

st.title("Word prediction")

sbtabs = ["Parameters"]
sbtab1 = st.sidebar.tabs(sbtabs)[0]
col1, col2, col3 = st.columns(3)

if 'words' not in st.session_state:
    st.session_state.words = []
if 'tree' not in st.session_state:
    st.session_state.tree = SuffixTree()
if 'tree_name' not in st.session_state:
    st.session_state.tree_name = ""
if 'reset_depth' not in st.session_state:
    st.session_state.reset_depth = True

max_depth = st.session_state.tree.pattern_depth(st.session_state.words)
if 'depth' not in st.session_state:
    st.session_state.depth = max_depth

filenames = os.listdir('trees\\')
if 'filename' not in st.session_state:
    if not st.session_state.tree_name:
        filenames += ["<select>"]
    st.session_state.filename = "<select>"

with sbtab1:
    def reset_depth():
        st.session_state.reset_depth = True
    if max_depth > 0:
        if st.session_state.reset_depth:
            st.session_state.depth = max_depth
            st.session_state.reset_depth = False
        st.slider(
            'Depth',
            0,max_depth, step=1,
            key="depth"
        )
    else:
        st.write("depth = 0")


    st.selectbox('Select a file', filenames, key='filename', on_change=load_tree)

    # sort_names = ["weight","entropy","relative entropy","info gain"]
    # if 'sort' not in st.session_state:
    #     st.session_state.sort = sort_names[0]
    # st.selectbox('Sorting by', sort_names, key='sort')

if st.session_state.tree.weight == 0:
    st.title("Choose a predictor")
else:
    with col1:
        st.title("Nodes")
        subtree = st.session_state.tree.get_tree(st.session_state.words[len(st.session_state.words)-st.session_state.depth:], copy_node_lst=False)
        weight = subtree.weight
        st.write("  \n".join(st.session_state.words[:len(st.session_state.words)-st.session_state.depth]))
        st.write(f":blue[{"  \n".join(st.session_state.words[len(st.session_state.words)-st.session_state.depth:])}]")
        st.write(f"motif weight:{weight}")
        st.write(f"motif entropy:{round(subtree.entropy(),3)}")
        if st.session_state.words:
            def remove_word():
                st.session_state.words.pop()
                reset_depth()
            st.button("Erase", on_click=remove_word, type='primary')

    with col2:
        st.title("Next node probability")
        for key, value in sorted(subtree.transition_weight.items(), key=lambda x:-x[1]):
            def add_word(word=key): 
                st.session_state.words.append(word)
                reset_depth()
            proba = round(value/weight,3)
            proba_txt = str(proba) if proba!=0 else "<0.001"
            st.button(f"{key}: :blue[{proba_txt}]", on_click=add_word, type='secondary')
    # with col3:
        # trees = [st.session_state.tree.get_tree(st.session_state.words[])]
