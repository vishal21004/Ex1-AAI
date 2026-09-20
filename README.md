# Implementation of Bayesian Networks
### Aim :
To create a bayesian Network for the given dataset in Python.
    
### Algorithm:

Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```python
pip install pandas networkx matplotlib pybbn

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Step 2: Set pandas options
pd.set_option('display.max_columns', None)

df = pd.read_csv('/content/weatherAUS.csv')

df = df.dropna(subset=['RainTomorrow'])

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df['RainTomorrow'] = df['RainTomorrow'].astype(str)


# Humidity: > 60 is High, <= 60 is Low
df['Humidity9amCat'] = df['Humidity9am'].apply(lambda x: 'High' if x > 60 else 'Low')
df['Humidity3pmCat'] = df['Humidity3pm'].apply(lambda x: 'High' if x > 60 else 'Low')

# WindGustSpeed: > 40 is High, <= 40 is Low
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(lambda x: 'High' if x > 40 else 'Low')

def probs(df, child, parents=[]):
    if not parents:
        # Calculate marginal probabilities (no parents)
        prob = df[child].value_counts(normalize=True).sort_index().tolist()
    else:
        # Calculate conditional probabilities (with parents)
        # Group by parents, then calculate value counts of the child
        prob = df.groupby(parents)[child].value_counts(normalize=True).unstack().fillna(0)
        # Flatten the matrix into a single list as required by pybbn
        prob = prob.sort_index().values.flatten().tolist()
    return prob

H9am = BbnNode(Variable(0, 'Humidity9amCat', ['High', 'Low']), probs(df, 'Humidity9amCat'))

H3pm = BbnNode(Variable(1, 'Humidity3pmCat', ['High', 'Low']), 
               probs(df, child='Humidity3pmCat', parents=['Humidity9amCat']))

WGS = BbnNode(Variable(2, 'WindGustSpeedCat', ['High', 'Low']), probs(df, 'WindGustSpeedCat'))

RT = BbnNode(Variable(3, 'RainTomorrow', ['No', 'Yes']), 
             probs(df, child='RainTomorrow', parents=['Humidity3pmCat', 'WindGustSpeedCat']))

# Step 9: Create Bbn object and add nodes and edges
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(WGS) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(WGS, RT, EdgeType.DIRECTED))


pos = {
    0: (-1, 1),   # Humidity9amCat
    1: (-1, 0),   # Humidity3pmCat
    2: (1, 0),    # WindGustSpeedCat
    3: (0, -1)    # RainTomorrow
}

join_tree = InferenceController.apply(bbn)

# Mapping node IDs to their actual names for the labels
labels = {
    0: 'Humidity9am',
    1: 'Humidity3pm',
    2: 'WindGustSpeed',
    3: 'RainTomorrow'
}

g = nx.DiGraph()
# Add nodes to the networkx graph using their IDs
for node_id in labels.keys():
    g.add_node(node_id)

# Add edges to the networkx graph, handling potential string representations
for edge_item in bbn.edges:
    if isinstance(edge_item, Edge):
        # If it's an actual Edge object
        from_node_id = edge_item.from_node.variable.id
        to_node_id = edge_item.to_node.variable.id
    elif isinstance(edge_item, str):
        # If it's a string (e.g., '0->1'), parse it
        parts = edge_item.split('->')
        if len(parts) == 2:
            try:
                from_node_id = int(parts[0])
                to_node_id = int(parts[1])
            except ValueError:
                print(f"Warning: Could not convert edge string parts to int: {edge_item}. Skipping.")
                continue
        else:
            print(f"Warning: Unexpected edge string format: {edge_item}. Skipping.")
            continue
    else:
        print(f"Warning: Unexpected type for edge: {type(edge_item)}. Skipping.")
        continue
    g.add_edge(from_node_id, to_node_id)

# Step 12 (Options) & Step 14: Update margins and display the graph
plt.figure(figsize=(8, 6))
nx.draw(g, pos=pos, labels=labels, with_labels=True,
        node_color='skyblue', node_size=4000,
        font_size=10, font_weight='bold',
        arrowsize=20, edge_color='gray')

plt.margins(0.2) # Update margins to prevent labels from being cut off
plt.title('Weather Prediction Bayesian Network')
plt.show()
```
## Output:
<img width="819" height="642" alt="image" src="https://github.com/user-attachments/assets/39981d30-8081-448d-b42f-45acc2de6e75" />

## Result:
   Thus a Bayesian Network is generated using Python
