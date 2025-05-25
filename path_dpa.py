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

# Specification
ltl = '(GFa & FGb) | (GFc & FGd) | (GFe & GFf & FGg)'
oa = OmegaAutomaton(ltl,oa_type='dpa')

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
    [('a','b'), ('b',),    (),     ('d',),    ('c','d')],
    [('g',),    ('g',),    ('g',), ('g',),    ('g',)],
    [('g',),    ('g',),    ('g',), ('g',),    ('g',)],
    [('g',),    ('g',),    (),     ('g',),    ('g',)],
    [('g',),    ('e','g'), ('g',), ('f','g'), ('g',)]
],dtype=np.object)

reward = np.zeros(shape)

lcmap={
    'a':'plum',
    'b':'pink',
    'c':'palegreen',
    'd':'turquoise',
    'e':'lightcoral',
    'f':'sandybrown',
    'g':'gold',
}

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=5,robust=True,lcmap=lcmap)
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa)

if method == 'shapley':
    csrl.shapley(T=T,name='path')
elif method =='minimax_q':
    csrl.minimax_q(T=T,K=K,start=(0,0),name='path')
