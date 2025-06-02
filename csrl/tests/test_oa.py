
from ..oa import OmegaAutomaton, spot
from collections import Counter
from random import choice

R = 2000
ltl_dict = {
    'Safe Aborbing States': {
        'ltl': r'(FGa | FGb) & G!c',
        'paths': [
            (True, [('b',), ('a',), ('a', 'b'), (), *[('a',) for _ in range(R)]]),
            (True, [('b',), ('a',), ('a', 'b'), (), *[('b',) for _ in range(R)]]),
            (False, [('b',), ('a',), ('a', 'b'), ('c',), *[('a',) for _ in range(R)]]),
            (False, [('b',), ('a',), ('a', 'b'), ('c',), *[('b',) for _ in range(R)]]),
        ],
    },
    'Nursery': {
        'ltl': r'G(!d & ((b & X!b) -> X(!b U (a | c))) & ((!b & Xb & XX!b) -> (!a U b)) & (c -> (!a U b)) & ((b & X b)->Fa))',
    },
    'Monitoring while Avoiding Adversary': {
        'ltl': r'GFb & GFc & (FGd | FGe) & G!a',
    },
    'Intrusion Detection': {
        'ltl': r'F("anomaly" & X("anomaly" & X("attack" & XF"attack")))',
    },
    'Surveillance': {
        'ltl': r'GFb & GFc & FGd',
    },
    'Sequencing': {
        'ltl': r'F(b & F(c & F(d & Fe))) & G!a',
    },
    'Safe and Persistent Repetition': {
        'ltl': r'GFb & FGc & G!(d & Xd)',
    },
    'Charging in Workspace': {
        'ltl': r'((FGw & GFc & GFr) | FGc) & G!d',
    },
    'Garbage Collection': {
        'ltl': r'(G(Fg & (g -> (Xt | XXt))) | (FGw & GFc & GFr)) & G!d',
    },
    'Readching Goals I': { 
        'ltl': r'((GF"green" & GF"blue" & FG"boundary") | GF("green" & "blue")) & G!"red"',
    },
    'Reaching Goals II': {
        'ltl': r'FG("red" & !"boundary") | F("red" & "green") | F("red" & "blue") | GF("green" & "blue" & !"red")',
    },
    'Cart Pole': {
        'ltl': r'G("position_x>-10" & "position_x<10") & G("velocity_x>-10.0" & "velocity_x<10.0") & F("cos_theta<-0.5" & F"cos_theta>0.5")'
    },
    'Cheetah': {
        'ltl': r'G"tip_height>-7.5" & GF"tip_height>-7.0" & F("tip_velocity_x>3.0" & F"tip_velocity_x<0")'
    },
}


def test_oa_construction():
    for name, info in ltl_dict.items():
        for oa_type in ['dpa', 'ldba']:
            ltl = info['ltl']
            oa = OmegaAutomaton(ltl, oa_type)
            
            # Check if the automaton has been created
            assert hasattr(oa, 'ltl')
            assert hasattr(oa, 'hoa') 
            assert hasattr(oa, 'spot_oa')
            assert hasattr(oa, 'aps')
            assert hasattr(oa, 'labels')
            assert hasattr(oa, 'q0') 
            assert hasattr(oa, 'delta') 
            assert hasattr(oa, 'acc')
            assert hasattr(oa, 'shape')

            assert isinstance(oa.ltl, str) and oa.ltl == ltl
            assert isinstance(oa.hoa, bytes)
            assert isinstance(oa.spot_oa, spot.twa_graph)
            assert isinstance(oa.aps, list)
            assert isinstance(oa.labels, list)
            assert isinstance(oa.q0, int)
            assert isinstance(oa.delta, list)
            assert isinstance(oa.acc, list)
            assert isinstance(oa.shape, tuple)

            oa.q0 is not None
            assert len(oa.delta) > 0
            assert len(oa.acc) > 0
            
            # Check if the acceptance condition is well-formed
            assert isinstance(oa.acc, list)
            for acc in oa.acc:
                assert isinstance(acc, dict)
            
            # Check if the shape is correctly defined
            assert isinstance(oa.shape, tuple)
            assert len(oa.shape) == 2

            oa = OmegaAutomaton(ltl, save_hoa=False, save_svg=True)
            oa = OmegaAutomaton(ltl, save_hoa=True, save_svg=False)
            oa = OmegaAutomaton(ltl, save_hoa=True, save_svg=True)



def test_oa_acceptance():
    for name, info in ltl_dict.items():
        ltl = info['ltl']
        for accepting, path in info.get('paths', []):
            dpa = OmegaAutomaton(ltl, 'dpa')
            visited_colors = []
            q = dpa.q0
            for label in path:
                visited_colors.append(dpa.acc[q][label])
                q = dpa.delta[q][label]
            
            visited_colors_suffix = visited_colors[len(path)//2:]
            counts = Counter(visited_colors_suffix)

            repeated_colors = []
            for color in counts:
                if counts[color] >= len(path) / (dpa.shape[1]**2):
                    repeated_colors.append(color)

            accepts = max(repeated_colors)%2==1
            assert accepting==accepts


            ldba = OmegaAutomaton(ltl, 'ldba')
            accepts = False
            for i in range(100):
                visited_colors = []
                q = ldba.q0
                for label in path:
                    color, q = choice(list(zip(ldba.acc[q][label], ldba.delta[q][label])))
                    visited_colors.append(color)

                visited_colors_suffix = visited_colors[len(path)//2:]
                counts = Counter(visited_colors_suffix)

                repeated_colors = []
                for color in counts:
                    if counts[color] >= len(path) / (dpa.shape[1]**2):
                        repeated_colors.append(color)

                accepts = accepts or (max(repeated_colors)%2==1)
            assert accepting==accepts





