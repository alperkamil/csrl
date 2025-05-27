
from ..oa import OmegaAutomaton

ltl_dict = {
    'Safe Aborbing': {
        'ltl': r'(FGa | FGb) & G!c',
    },
    'Nursery': {
        'ltl': r'G(!d & ((b & X!b) -> X(!b U (a | c))) & ((!b & Xb & XX!b) -> (!a U b)) & (c -> (!a U b)) & ((b & X b)->Fa))',
    },
}

def test_oa_construction():
    for ltl_name, ltl_data in ltl_dict.items():
        for oa_type in ['dpa', 'ldba']:
            ltl = ltl_data['ltl']
            oa = OmegaAutomaton(ltl)
            
            # Check if the automaton has been created
            assert oa.q0 is not None
            assert len(oa.delta) > 0
            assert len(oa.acc) > 0
            
            # Check if the acceptance condition is well-formed
            assert isinstance(oa.acc, list)
            for acc in oa.acc:
                assert isinstance(acc, dict)
            
            # Check if the shape is correctly defined
            assert isinstance(oa.shape, tuple)
            assert len(oa.shape) == 2
            
