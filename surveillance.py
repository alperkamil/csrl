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
print('surveillance'+suffix)

# Specification
phi_det = 'F(u & Xu & (XXm | XXXm))'
phi_obj = '(GFb & GFc & FGf)'
ltl = phi_det +' | ' + phi_obj
oa = OmegaAutomaton(ltl,oa_type='dra')
print('Number of Omega-automaton states (including the trap state):',oa.shape[1])
print('Number of accepting pairs:',oa.shape[0])

# MDP Description
shape = (7,9)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
    ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E']
])

label = np.array([
    [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('b','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('f',),    ('f',),    (),        ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('c','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
    [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)]
],dtype=np.object)

reward = np.zeros(shape)

lcmap={
    'b':'peachpuff',
    'c':'plum',
    'f':'palegreen'
}

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=9,secure=True,lcmap=lcmap)  # Use figsize=4 for smaller figures
grid_mdp.plot()
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa,discount=0.999,discountB=0.99,discountC=0.9)


if method == 'shapley':
    csrl.shapley(T=T,name='surveillance',tt=2**15)
elif method =='minimax_q':
    csrl.minimax_q(T=T,K=K,name='surveillance',tt=2**15)