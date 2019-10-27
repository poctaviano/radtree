# Radial Plot for sklearn Decision Trees

<table width="100%">
  <tr>
    <td><img src="./assets/iris.png?raw=true" width="100%"></td>
    <td><img src="./assets/bcancer.png?raw=true" width="100%"></td>
    <td><img src="./assets/titanic3.png?raw=true" width="100%"></td>
  </tr>
</table>

## What is it?

`radtree` is a tool to make radial plots of Decision Trees Classifiers.   
It uses networkx to create the tree structure and matplotlib to plot.

I would like to thanks <a href='https://stackoverflow.com/users/1429402/fnord'>Fnord</a> for tem <a href='https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy'>b-spline algorithm</a>.

## Installation

- Install radtree from Pypi :

```
pip install git+https://github.com/poctaviano/radtree
```

## Examples

Please check the [/notebooks folder](./notebooks/) to view some examples.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

import radtree

X, y = datasets.load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

radtree.plot_radial(dtree, X=X_test,Y=y_test,
                    smooth_d=8, l_alpha=.2, l_width=1,
                    random_seed=42,
                   )

```

<table width="100%">
  <tr>
    <td><img src="./assets/iris2.png?raw=true" width="100%"></td>
  </tr>
</table>
<!-- <div bgcolor="#000000"><img src="./assets/iris2.png?raw=true" width="90%"></div> -->
