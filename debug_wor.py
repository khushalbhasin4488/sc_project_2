"""Quick diagnostic: check tree_.value for leaf nodes."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

np.random.seed(42)
df = pd.read_csv('/Users/khushalbhasin/Documents/code/sc_project/diabetes.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for c in cols:
    m = df[c].replace(0, np.nan).median()
    df[c] = df[c].replace(0, m)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

dt = DecisionTreeClassifier(criterion='entropy', random_state=42, min_samples_leaf=10)
dt.fit(X, y)

tree_ = dt.tree_
print(f"n_leaves: {dt.get_n_leaves()}")
print(f"\nFirst 10 leaf nodes:")
count = 0
for node in range(tree_.node_count):
    if tree_.feature[node] == _tree.TREE_UNDEFINED:
        vals = tree_.value[node][0]
        pc = int(np.argmax(vals))
        CC = vals[pc]
        IC = np.sum(vals) - CC
        if count < 10:
            print(f"  Node {node}: class={pc}, values={vals}, CC={CC}, IC={IC}, total={np.sum(vals)}")
        count += 1
print(f"\nTotal leaf nodes: {count}")

# Show max CC for class-0 leaves
max_cc = 0
for node in range(tree_.node_count):
    if tree_.feature[node] == _tree.TREE_UNDEFINED:
        vals = tree_.value[node][0]
        pc = int(np.argmax(vals))
        if pc == 0:
            CC = vals[pc]
            if CC > max_cc:
                max_cc = CC
print(f"\nMax CC for class-0 leaf: {max_cc}")
