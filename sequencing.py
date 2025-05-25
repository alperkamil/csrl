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
print('sequencing'+suffix)

# Specification
phi_det = 'F(u & Xu & ((XXm & XXXFm) | (XXXm & XXXXFm)))'
phi_obj = '(F(b & F(c & F(d & Fe))) & G!a)'
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
    [(),        ('b',),    (),        (),        (),        (),        (),        ('c',),    ()],
    [(),        (),        (),        (),        (),        (),        (),        (),        ()],
    [(),        (),        (),        (),        (),        (),        (),        (),        ()],
    [(),        (),        (),        (),        ('a',),    (),        (),        (),        ()],
    [(),        (),        (),        (),        (),        (),        (),        (),        ()],
    [(),        (),        (),        (),        (),        (),        (),        (),        ()],
    [(),        ('e',),    (),        (),        (),        (),        (),        ('d',),    ()]
],dtype=np.object)

reward = np.zeros(shape)

lcmap={
    'b':'peachpuff',
    'c':'plum',
    'd':'lightskyblue',
    'e':'paleturquoise',
    'a':'red'
}

grid_mdp = GridMDP(shape=shape,structure=structure,reward=reward,label=label,figsize=9,secure=True,lcmap=lcmap)  # Use figsize=4 for smaller figures
grid_mdp.plot()
# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa,discount=0.999,discountB=0.99,discountC=0.9)


if method == 'shapley':
    csrl.shapley(T=T,name='sequencing',tt=2**15)
elif method =='minimax_q':
    csrl.minimax_q(T=T,K=K,name='sequencing',tt=2**15)