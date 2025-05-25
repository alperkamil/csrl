# Control Synthesis from Linear Temporal Logic Specifications using Model-Free Reinforcement Learning

This repository includes the implementation of the learning-based synthesis algorithm described in this [article](https://arxiv.org/abs/1909.07299).
## Dependencies
 - [Python](https://www.python.org/): (>=3.5)
 - [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` (```ltl2ldra``` is optional)
 - [NumPy](https://numpy.org/): (>=1.15)
 
The examples in this repository also require the following optional libraries for visualization:
 - [Matplotlib](https://matplotlib.org/): (>=3.03)
 - [JupyterLab](https://jupyter.org/): (>=1.0)
 - [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/): (>=7.5)

## Installation
To install the current release:
```
git clone https://github.com/alperkamil/csrl.git
cd csrl
pip3 install .
```
## Basic Usage
The package consists of three main classes ```GridMDP```, ```OmegaAutomaton``` and ```ControlSynthesis```. The class ```GridMDP``` constructs a grid-world MDP using the parameters ```shape```, ```structure``` and ```label```. The class ```OmegaAutomaton``` takes an LTL formula ```ltl``` and translates it into an LDBA. The class ```ControlSynthesis``` can then be used to compose a product MDP of the given ```GridMDP``` and ```OmegaAutomaton``` objects and its method ```q_learning``` can be used to learn a control policy for the given objective. For example,
```shell
$ python
```
```python
>>> from csrl.mdp import GridMDP
>>> from csrl.oa import OmegaAutomaton
>>> from csrl import ControlSynthesis
>>> import numpy as np
>>> 
>>> ltl = '(F G a | F G b) & G !c'  # LTL formula
>>> oa = OmegaAutomaton(ltl)  # LDBA
>>> print('LDBA Size (including the trap state):',oa.shape[1])
LDBA Size (including the trap state): 4
>>> 
>>> shape = (5,4)  # Shape of the grid
>>> structure = np.array([  # E:Empty, T:Trap, B:Obstacle
... ['E',  'E',  'E',  'E'],
... ['E',  'E',  'E',  'T'],
... ['B',  'E',  'E',  'E'],
... ['T',  'E',  'T',  'E'],
... ['E',  'E',  'E',  'E']
... ])
>>> label = np.array([  # Labels
... [(),       (),     ('c',),()],
... [(),       (),     ('a',),('b',)],
... [(),       (),     ('c',),()],
... [('b',),   (),     ('a',),()],
... [(),       ('c',), (),    ('c',)]
... ],dtype=np.object)
>>> grid_mdp = GridMDP(shape=shape,structure=structure,label=label)
>>> grid_mdp.plot()
>>> 
>>> csrl = ControlSynthesis(grid_mdp,oa) # Product MDP
>>> 
>>> Q=csrl.q_learning(T=100,K=100000)  # Learn a control policy
>>> value=np.max(Q,axis=4)
>>> policy=np.argmax(Q,axis=4)
>>> policy[0,0]
array([[1, 3, 0, 2],
       [2, 3, 3, 6],
       [0, 3, 0, 2],
       [6, 0, 5, 0],
       [3, 0, 0, 0]])
``` 

## Examples
The repository contains a couple of example IPython notebooks:
 - [LTL to Omega-Automata Translation](Examples%20of%20LTL%20to%20Omega-Automata%20Translation.ipynb)
 - [MDPs](Examples%20of%20MDPs.ipynb)
 - [Safe Absorbing States](Safe%20Absorbing%20States.ipynb)
 - [Robust Controller](Robust%20Controller.ipynb)
 - [Adversary](Adversary.ipynb)

Animations of the case studies: [safe_absorbing.mp4](safe_absorbing.mp4) and [nursery.mp4](nursery.mp4).
