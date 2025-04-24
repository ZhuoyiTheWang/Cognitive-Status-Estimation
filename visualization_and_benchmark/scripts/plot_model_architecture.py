from graphviz import Digraph

dot = Digraph(comment='Neural Network Architecture', format='pdf')

# Adjust the overall layout direction and spacing
dot.attr(rankdir='TB')
dot.attr(
    'graph',
    margin="0.2",        # minimal outer margin around the entire diagram
    nodesep="0.1",       # horizontal space between nodes on the same rank
    ranksep="0.3",       # vertical space between ranks
    concentrate="true",  # merge multi-edge paths where possible
    ratio="fill"         # attempt to fill the drawing page
)

# Adjust default node appearance
# - larger font size
# - bigger margin inside each node box
# - optionally fixed size for uniform boxes
dot.attr(
    'node',
    shape="box",
    style="rounded,filled",
    fillcolor="lightgray",
    fontsize="18",
    margin="0.15,0.1",
    width="1.5",         # minimum width of each node
    height="0.8",        # minimum height of each node
    fixedsize="false"    # set to "true" if you want uniform-sized boxes
)

# Adjust default edge appearance
# - smaller arrow size
dot.attr('edge', arrowsize="0.7")

# Define nodes for each component of the network
dot.node('A', 'Sequence Input\n(shape=(seq_length, feature_dim))')
dot.node('B', 'LSTM(128)')
dot.node('D', 'LSTM(64)')
dot.node('F', 'LSTM(64)')
dot.node('H', 'Next Features Input\n(shape=(feature_dim,))')
dot.node('I', 'Concatenate\n(merge LSTM output\nwith next features)')
dot.node('J', 'Dense(64,\nactivation="relu")')
dot.node('L', 'Dense(32,\nactivation="relu")')
dot.node('N', 'Output\n(Dense(num_classes,\nactivation="softmax"))')

# Connect the nodes
dot.edge('A', 'B')
dot.edge('B', 'D')
dot.edge('D', 'F')
dot.edge('F', 'I')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('J', 'L')
dot.edge('L', 'N')

# Render and save the diagram as a PDF file
dot.render('model_architecture_diagram', cleanup=True)
