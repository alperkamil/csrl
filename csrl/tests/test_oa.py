
from ..oa import OmegaAutomaton, spot

ltl_dict = {
    'Safe Aborbing States': {
        'ltl': r'(FGa | FGb) & G!c',
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
}

def test_oa_construction():
    for name, info in ltl_dict.items():
        for oa_type in ['dpa', 'ldba']:
            ltl = info['ltl']
            oa = OmegaAutomaton(ltl)
            
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

