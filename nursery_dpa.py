#!/usr/bin/env python
# coding: utf-8

from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np

import sys

method = sys.argv[1]
T = 2**int(sys.argv[2])
K = 2**int(sys.argv[3])
suffix = sys.argv[2]+'-'+sys.argv[3]
print('robust_dpa-'+suffix)

# LTL Specification
ltl = 'FGe & GFc & GFb & Fa & G(a->X(G!a)) & G!d'

oa = OmegaAutomaton(ltl,oa_type='dpa')

# MDP Description
shape = (5,5)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
['E',  'E',  'E',  'E',  'E'],
['E',  'E',  'B',  'E',  'E'],
['E',  'E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E',  'E'],
['E',  'E',  'E',  'E',  'E']
])

# Labels of the states
label = np.array([
[(),        (),        ('d',),    (),        ()],
[('e',),    ('e',),    (),        ('e',),    ('e',)],
[('c','e'), ('e',),    ('a','e'), ('e',),    ('b','e',)],
[('e',),    ('e',),    ('e',),    ('e',),    ('e',)],
[('e',),    ('e',),    ('e',),    ('e',),    ('e',)]
],dtype=np.object)

# Colors of the labels
lcmap={
    'a':'gold',
    'b':'palegreen',
    'c':'turquoise',
    'd':'lightcoral',
    'e':'lightsteelblue',
}
reward = np.zeros(shape)

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=5,robust=True,lcmap=lcmap)
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)

if method == 'shapley':
    csrl.shapley(T=T,name='nursery')
elif method =='minimax_q':
    csrl.minimax_q(T=T,K=K,start=(0,0),name='nursery')
