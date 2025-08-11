
from ..envs import GridWorldEnv
import numpy as np 

def test_gridworld_construction():
    """
    Test the construction of GridWorld instances with a specific shapes and structures, and labels.
    """


    ## GridWorld with one-sided boundaries

    shape = (8,8)
    structure = np.array([
        ['E','E','E','E','B','E','E','E'],
        ['E','E','E','E','R','E','E','E'],
        ['E','E','E','E','B','E','E','E'],
        ['E','E','E','E','B','B','B','B'],
        ['T','B','T','E','E','E','E','E'],
        ['E','E','E','E','E','D','E','E'],
        ['E','E','E','E','R','E','L','E'],
        ['E','E','E','E','E','U','E','E']
    ])

    labels = np.empty(shape, dtype=object)
    labels.fill(())
    labels[0,0] = ('A',)
    labels[0,7] = ('A','B')
    labels[5,1] = ('C',)
    labels[6,5] = ('A','B','C')
        
    gw = GridWorld(shape=shape, structure=structure, labels=labels, figsize=10)
    gw.plot()


    ## GridWorld for the nursery scenario

    shape = (5,4)
    # E: Empty, T: Trap, B: Obstacle
    structure = np.array([
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E']
    ])

    # Labels of the states
    labels = np.array([
        [(),    (),    ('b',),('d',)],
        [(),    (),    (),    ()],
        [(),    (),    (),    ()],
        [('a',),(),    (),    ()],
        [(),    ('c',),(),    ()]
    ], dtype=object)
    # Colors of the labels
    lcmap={
        'a': 'yellow',
        'b': 'greenyellow',
        'c': 'turquoise',
        'd': 'pink'
    }
    # Use figsize=4 for smaller figures
    gw = GridWorld(shape=shape, structure=structure, labels=labels, lcmap=lcmap, figsize=5)
    gw.plot()


    ## GridWorld for the safe absorbing states scenario

    shape = (5,4)
    # E: Empty, T: Trap, B: Obstacle
    structure = np.array([
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'T'],
        ['B',  'E',  'E',  'E'],
        ['T',  'E',  'T',  'E'],
        ['E',  'E',  'E',  'E']
    ])

    # Labels of the states
    labels = np.array([
        [(),       (),     ('c',),()],
        [(),       (),     ('a',),('b',)],
        [(),       (),     ('c',),()],
        [('b',),   (),     ('a',),()],
        [(),       ('c',), (),    ('c',)]
    ], dtype=object)
    # Colors of the labels
    lcmap={
        'a': 'lightgreen',
        'b': 'lightgreen',
        'c': 'pink'
    }

    # Use figsize=4 for smaller figures
    gw = GridWorld(shape=shape, structure=structure, labels=labels, lcmap=lcmap, figsize=5)
    gw.plot()


    ## GridWorld for the robust controller scenario

    shape = (5,5)
    # E: Empty, T: Trap, B: Obstacle
    structure = np.array([
        ['E',  'E',  'B',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E']
    ])

    labels = np.array([
        [('b','d'), ('c','d'), (),     ('b','d'), ('c','d')],
        [('e',),    ('e',),    ('e',), ('e',),    ('e',)],
        [('e',),    ('e',),    ('e',), ('e',),    ('e',)],
        [('e',),    ('e',),    (),     ('e',),    ('e',)],
        [('e',),    ('b','e'), ('e',), ('c','e'), ('e',)]
    ], dtype=object)

    rewards = np.zeros(shape)

    lcmap={
        'b': 'peachpuff',
        'c': 'plum',
        'd': 'greenyellow',
        'e': 'palegreen'
    }

    # Use figsize=4 for smaller figures
    gw = GridWorld(shape=shape, structure=structure, rewards=rewards, labels=labels, figsize=5, lcmap=lcmap)
    gw.plot()


    ## GridWorld for the adversary scenario

    shape = (5,5)
    # E: Empty, T: Trap, B: Obstacle
    structure = np.array([
        ['E',  'E',  'B',  'E',  'E'],
        ['T',  'E',  'T',  'E',  'T'],
        ['B',  'E',  'B',  'E',  'B'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E']
    ])

    labels = np.array([
        [('b','d'),('c','d'),(),('c','d'),('b','d')],
        [(),       (),       (),       (),       ()],
        [(),       (),       (),       (),       ()],
        [('c','e'),(),       (),       (),('c','e')],
        [('b','e'),(),       (),       (),('b','e')]
    ], dtype=object)

    rewards = np.zeros(shape)
    lcmap={
        'b': 'peachpuff',
        'c': 'plum',
        'd': 'greenyellow',
        'e': 'palegreen'
    }
    # Use figsize=4 for smaller figures
    gw = GridWorld(shape=shape, structure=structure, rewards=rewards, labels=labels, figsize=5, lcmap=lcmap, prob_intended=0.6)
    gw.plot()


    ## GridWorld for the surveillance scenario
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

    labels = np.array([
        [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('b','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('f',),    ('f',),    (),        ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('c','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
        [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)]
    ], dtype=object)

    rewards = np.zeros(shape)

    lcmap={
        'b': 'peachpuff',
        'c': 'plum',
        'f': 'palegreen'
    }

    gw = GridWorld(shape=shape, structure=structure, rewards=rewards, labels=labels, figsize=9, lcmap=lcmap)
    gw.plot()


   ## GridWorld for the sequencing scenario

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

    labels = np.array([
        [(),        ('b',),    (),        (),        (),        (),        (),        ('c',),    ()],
        [(),        (),        (),        (),        (),        (),        (),        (),        ()],
        [(),        (),        (),        (),        (),        (),        (),        (),        ()],
        [(),        (),        (),        (),        ('a',),    (),        (),        (),        ()],
        [(),        (),        (),        (),        (),        (),        (),        (),        ()],
        [(),        (),        (),        (),        (),        (),        (),        (),        ()],
        [(),        ('e',),    (),        (),        (),        (),        (),        ('d',),    ()]
    ], dtype=object)

    rewards = np.zeros(shape)

    lcmap={
        'b': 'peachpuff',
        'c': 'plum',
        'd': 'lightskyblue',
        'e': 'paleturquoise',
        'a': 'red'
    }

    gw = GridWorld(shape=shape, structure=structure, rewards=rewards,labels=labels,figsize=9, lcmap=lcmap)
    gw.plot()


    ## GridWorld for the safe and persistent repetition scenario

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
    labels = np.array([
    [(),       (),       (),       ('d',),   ('c',),    ('c',)],
    [(),       (),       (),       (),       ('c',),    ('c',)],
    [(),       (),       (),       (),       ('c',),    ('b','c')],
    [(),       (),       (),       (),       ('c',),    ('c',)],
    [(),       (),       (),       (),       ('c',),    ('c',)]
    ], dtype=object)

    # Colors of the labels
    lcmap={
        'b': 'turquoise',
        'c': 'gold',
        'd': 'lightcoral'
    }

    # Native rewards
    rewards = np.zeros(shape)
    rewards[0,5]=1

    gw = GridWorld(shape=shape, structure=structure, labels=labels, rewards=rewards, lcmap=lcmap, figsize=6)
    gw.plot()


def test_gridworld_transitions():

    shape = (4,4)
    # E: Empty, T: Trap, B: Obstacle
    structure = np.array([
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E']
    ])

    # Labels of the states
    labels = np.array([
        [(),    (),    (),    ()],
        [(),    (),    (),    ()],
        [(),    (),    (),    ()],
        [(),    (),    (),    ()]
    ], dtype=object)

    # Use figsize=4 for smaller figures
    gw = GridWorld(shape=shape, structure=structure, labels=labels, figsize=5)
    
    dsts, probs = gw.get_transition_probs((0, 0), 'U')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(0,1)  # 'R'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.8)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'R'
    np.testing.assert_almost_equal(probs[2], 0.1)  # 'L'

    dsts, probs = gw.get_transition_probs((0, 0), 'D')
    assert dsts[0]==(1,0)  # 'D'
    assert dsts[1]==(0,1)  # 'R'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.8)  # 'D'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'R'
    np.testing.assert_almost_equal(probs[2], 0.1)  # 'L'


    dsts, probs = gw.get_transition_probs((0, 0), 'R')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(1,0)  # 'D'
    assert dsts[2]==(0,1)  # 'R'
    np.testing.assert_almost_equal(probs[0], 0.1)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'D'
    np.testing.assert_almost_equal(probs[2], 0.8)  # 'R'


    dsts, probs = gw.get_transition_probs((0, 0), 'L')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(1,0)  # 'D'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.1)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'D'
    np.testing.assert_almost_equal(probs[2], 0.8)  # 'L'


    structure = np.array([
        ['T',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E']
    ])
    gw = GridWorld(shape=shape, structure=structure, labels=labels)
    dsts, probs = gw.get_transition_probs((0, 0), 'U')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(0,0)  # 'R'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.8)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'R'
    np.testing.assert_almost_equal(probs[2], 0.1)  # 'L'

    dsts, probs = gw.get_transition_probs((0, 0), 'D')
    assert dsts[0]==(0,0)  # 'D'
    assert dsts[1]==(0,0)  # 'R'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.8)  # 'D'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'R'
    np.testing.assert_almost_equal(probs[2], 0.1)  # 'L'

    dsts, probs = gw.get_transition_probs((0, 0), 'R')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(0,0)  # 'D'
    assert dsts[2]==(0,0)  # 'R'
    np.testing.assert_almost_equal(probs[0], 0.1)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'D'
    np.testing.assert_almost_equal(probs[2], 0.8)  # 'R'

    dsts, probs = gw.get_transition_probs((0, 0), 'L')
    assert dsts[0]==(0,0)  # 'U'
    assert dsts[1]==(0,0)  # 'D'
    assert dsts[2]==(0,0)  # 'L'
    np.testing.assert_almost_equal(probs[0], 0.1)  # 'U'
    np.testing.assert_almost_equal(probs[1], 0.1)  # 'D'
    np.testing.assert_almost_equal(probs[2], 0.8)  # 'L'
