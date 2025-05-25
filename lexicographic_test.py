from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np 

# LTL Specification
ltl = 'GFb & FGc & (d | !d)'

# Translate the LTL formula to an LDBA
oa = OmegaAutomaton(ltl)

# MDP Description
shape = (5,6)
# E: Empty, T: Trap, B: Obstacle
structure = np.array([
['E',  'E',  'E',  'E',  'E',  'E'],
['E',  'E',  'E',  'B',  'E',  'E'],
['E',  'E',  'T',  'T',  'E',  'E'],
['E',  'E',  'E',  'E',  'E',  'E'],
['E',  'E',  'T',  'T',  'E',  'E']
])

# Labels of the states
label = np.array([
[(),       (),       (),       ('d',),   ('c',),    ('c',)],
[(),       (),       (),       (),       ('c',),    ('c',)],
[(),       (),       (),       (),       ('c',),    ('b','c')],
[(),       (),       (),       (),       ('c',),    ('c',)],
[(),       (),       (),       (),       ('c',),    ('c',)]
],dtype=np.object)
# Colors of the labels
lcmap={
    'b':'turquoise',
    'c':'gold',
    'd':'lightcoral'
}
reward = np.zeros(shape)
reward[0,5]=1
grid_mdp = GridMDP(shape=shape,structure=structure,label=label,reward=reward,lcmap=lcmap,figsize=6,lexicographic=True)  # Use figsize=4 for smaller figures
# grid_mdp.plot()


# Construct the product MDP
csrl = ControlSynthesis(grid_mdp,oa,discountB=0.9)
Q_psi,Q_phi,Q=csrl.q_learning(start=(0,0),T=2**10,K=2**17)

from itertools import product
tau=0.05

value_psi=np.max(Q_psi,axis=4)
Q_phi_ = np.copy(Q_phi)
for state in product(range(Q.shape[0]),range(Q.shape[1]),range(Q.shape[2]),range(Q.shape[3])):
    for action in range(Q.shape[4]):
        if Q_psi[state][action] < value_psi[state]-tau:
            Q_phi_[state][action] = -1
    
value_phi=np.max(Q_phi_,axis=4) 
Q_ = np.copy(Q)
for state in product(range(Q.shape[0]),range(Q.shape[1]),range(Q.shape[2]),range(Q.shape[3])):
    for action in range(Q.shape[4]):
        if Q_phi_[state][action] < value_phi[state]-tau:
            Q_[state][action] = -1
            
value=np.max(Q_,axis=4)
policy=np.argmax(Q_,axis=4)
# csrl.plot(value,policy,save='lexicographic_policy.pdf')
# csrl.plot(value,policy)


policy_ = policy[0,1]

K=2**20
T=2**10
print('K=20')

R = 0
state=(0,5)
for k in range(K):
    for t in range(T):
        if state==(0,5):
            R += 0.99**t
        states, probs = grid_mdp.transition_probs[state][policy_[state]]
        state = states[np.random.choice(len(states),p=probs)]
R = R/K
print(R)

R = 0
state=(0,5)
for k in range(K):
    for t in range(T):
        if state==(0,5):
            R += 0.99**t
        if state not in [(0,4),(2,4),(3,4),(4,4)] and np.random.random() < 0.05:
            a = np.random.randint(4)
        else:
            a = policy_[state]
        states, probs = grid_mdp.transition_probs[state][a]
        state = states[np.random.choice(len(states),p=probs)]
R = R/K
print(R)

state=(0,5)
T=T*K
count=0
for t in range(T):
    if state==(2,5):
        count += 1
    if state not in [(0,4),(2,4),(3,4),(4,4)] and np.random.random() < 0.05:
        a = np.random.randint(4)
    else:
        a = policy_[state]
    states, probs = grid_mdp.transition_probs[state][a]
    state = states[np.random.choice(len(states),p=probs)]
f = count/T
print(1/f)