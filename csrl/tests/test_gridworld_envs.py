
from .. import GridWorldEnv
import numpy as np 


gridworld_dict = {
    'Safe Aborbing States': {
        'shape': (5, 4),
        'structure': np.array([
            ['E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'T'],
            ['B',  'E',  'E',  'E'],
            ['T',  'E',  'T',  'E'],
            ['E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [(),     (),     ('c',), ()    ],
            [(),     (),     ('a',), ('b',)],
            [(),     (),     ('c',), ()    ],
            [('b',), (),     ('a',), ()    ],
            [(),     ('c',), (),     ('c',)]
        ], dtype=object),
        'lcmap': {
            'a': 'lightgreen',
            'b': 'lightgreen',
            'c': 'pink'
        }
    },

    'Nursery': {
        'shape': (5, 4),
        'structure': np.array([
            ['E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [(),     (),     ('b',), ('d',)],
            [(),     (),     (),     ()    ],
            [(),     (),     (),     ()    ],
            [('a',), (),     (),     ()    ],
            [(),     ('c',), (),     ()    ]
        ], dtype=object),
        'lcmap': {
            'a': 'yellow',
            'b': 'greenyellow',
            'c': 'turquoise',
            'd': 'pink'
        }
    },

    'Robust Control': {
        'shape': (5, 5),
        'structure': np.array([
            ['E',  'E',  'B',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [('b','d'), ('c','d'), (),     ('b','d'), ('c','d')],
            [('e',),    ('e',),    ('e',), ('e',),    ('e',)   ],
            [('e',),    ('e',),    ('e',), ('e',),    ('e',)   ],
            [('e',),    ('e',),    (),     ('e',),    ('e',)   ],
            [('e',),    ('b','e'), ('e',), ('c','e'), ('e',)   ]
        ], dtype=object),
        'lcmap': {
            'b': 'peachpuff',
            'c': 'plum',
            'd': 'greenyellow',
            'e': 'palegreen'
        }
    },

    'Monitoring while Avoiding Adversary': {
        'shape': (5, 5),
        'structure': np.array([
            ['E',  'E',  'B',  'E',  'E'],
            ['T',  'E',  'T',  'E',  'T'],
            ['B',  'E',  'B',  'E',  'B'],
            ['E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [('b','d'), ('c','d'), (), ('c','d'), ('b','d')],
            [(),        (),        (),        (), ()       ],
            [(),        (),        (),        (), ()       ],
            [('c','e'), (),        (),        (), ('c','e')],
            [('b','e'), (),        (),        (), ('b','e')]
        ], dtype=object),
        'lcmap': {
            'b': 'peachpuff',
            'c': 'plum',
            'd': 'greenyellow',
            'e': 'palegreen'
        }
    },

    'Surveillance': {
        'shape': (7, 9),
        'structure': np.array([
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('b','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('f',),    ('f',),    (),        ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('c','f'), ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)],
            [(),        ('f',),        ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',),    ('f',)]
        ], dtype=object),
        'lcmap': {
            'b': 'peachpuff',
            'c': 'plum',
            'f': 'palegreen'
        }
    },

    'Sequencing': {
        'shape': (7, 9),
        'structure': np.array([
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E',  'E',  'E',  'E']
        ]),
        'labels': np.array([
            [(),        ('b',),    (),        (),        (),        (),        (),        ('c',),    ()],
            [(),        (),        (),        (),        (),        (),        (),        (),        ()],
            [(),        (),        (),        (),        (),        (),        (),        (),        ()],
            [(),        (),        (),        (),        ('a',),    (),        (),        (),        ()],
            [(),        (),        (),        (),        (),        (),        (),        (),        ()],
            [(),        (),        (),        (),        (),        (),        (),        (),        ()],
            [(),        ('e',),    (),        (),        (),        (),        (),        ('d',),    ()]
        ], dtype=object),
        'lcmap': {
            'b': 'peachpuff',
            'c': 'plum',
            'd': 'lightskyblue',
            'e': 'paleturquoise',
            'a': 'red'
        }
    },

    'Safe and Persistent Repetition': {
        'shape': (5, 6),
        'structure': np.array([
            ['E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'E',  'B',  'E',  'E'],
            ['E',  'E',  'T',  'T',  'E',  'E'],
            ['E',  'E',  'E',  'E',  'E',  'E'],
            ['E',  'E',  'T',  'T',  'E',  'E']
        ]),
        'labels': np.array([
            [(),       (),       (),       ('d',),   ('c',),    ('c',)],
            [(),       (),       (),       (),       ('c',),    ('c',)],
            [(),       (),       (),       (),       ('c',),    ('b','c')],
            [(),       (),       (),       (),       ('c',),    ('c',)],
            [(),       (),       (),       (),       ('c',),    ('c',)]
        ], dtype=object),
        'lcmap': {
            'b':'turquoise',
            'c':'gold',
            'd':'lightcoral'
        }
    }

}

def test_gridworld_construction():
    """
    Test the construction of GridWorld instances with a specific shapes and structures, and labels.
    """

    ## GridWorld with various cell types

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
        
    gw = GridWorldEnv(shape=shape, structure=structure, labels=labels, figsize=10)
    gw.plot()
    transition_states, transition_probs, rewards = gw.get_transition_reward_arrays()


    for name, kwargs in gridworld_dict.items():
        gw = GridWorldEnv(**kwargs)
        gw.plot()
        transition_states, transition_probs, rewards = gw.get_transition_reward_arrays()

    


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
    gw = GridWorldEnv(shape=shape, structure=structure, labels=labels, figsize=5)
    
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
    gw = GridWorldEnv(shape=shape, structure=structure, labels=labels)
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

