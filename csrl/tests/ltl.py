

formulas = {
    'Safe Aborbing': {
        'ltl': r'(FGa | FGb) & G!c',
    },
    'Nursery': {
        'ltl': r'G(!d & ((b & X!b) -> X(!b U (a | c))) & ((!b & Xb & XX!b) -> (!a U b)) & (c -> (!a U b)) & ((b & X b)->Fa))',
    },
}