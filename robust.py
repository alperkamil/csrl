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
print('robust-'+suffix)

# Specification
ltl = 'G F b & G F c & (F G d | F G e)'
oa = OmegaAutomaton(ltl,oa_type='dra')
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])
print('Number of accepting pairs:',oa.shape[0])

# MDP Description
shape = (5,5)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
    ['E',  'E',  'B',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E']
])

label = np.array([
    [('b','d'), ('c','d'), (),     ('b','d'), ('c','d')],
    [('e',),    ('e',),    ('e',), ('e',),    ('e',)],
    [('e',),    ('e',),    ('e',), ('e',),    ('e',)],
    [('e',),    ('e',),    (),     ('e',),    ('e',)],
    [('e',),    ('b','e'), ('e',), ('c','e'), ('e',)]
],dtype=np.object)

reward = np.zeros(shape)

lcmap={
    'b':'peachpuff',
    'c':'plum',
    'd':'greenyellow',
    'e':'palegreen'
}

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=5,robust=True,lcmap=lcmap)  # Use figsize=4 for smaller figures
grid_mdp.plot()
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)


if method == 'shapley':
    csrl.shapley(T=T)
elif method =='minimax_q':
    csrl.minimax_q(T=T,K=K)