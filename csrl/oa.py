"""
Omega-Automaton Construction

"""

import os
import socket
import time
from datetime import datetime
from subprocess import check_output
import random
from itertools import chain, combinations

import spot

class OmegaAutomaton:
    """
    Transforms the LTL formula to an omega-automaton (OA) and stores the specifications.

    Attributes
    ----------
    q0 : int
        The initial state of the OA.

    delta : list of dicts
        A list representation of the transition function of the OA.
        For example, `delta[q][label]` will give you the OA state that the OA makes a transition to when the symbol `label` is consumed in the OA state `q`.

    epsmoves : list of lists
        TODO: The attribute has not been implemented.
        A list representation of epsilon-moves of the OA. 
        For example, `epsmoves[q]` will give you the set of states the OA can nondeterministically transition to from the OA state `q` without consuming any symbol.

    acc : list of dicts
        A list representation of the acceptance condition of the OA. For DPAs, `acc[q][label]` is the color of transition triggered by consuming the symbol `label` in the OA state `q`. 
        TODO: The following have not been implemented.
        For LDBAs, `acc[q][label]` is `True` if the transition `q--label-->` is accepting and `None` otherwise.
        For DRAs, `acc[q][label]` is an array such that  `acc[q][label][k]` is `False` if `q--label-->` belongs to the first set in the `k`th Rabin pair; 
        `True` if `q--label-->` belongs to the second set in the `k`th Rabin pair; and `None` if `q--label-->` doesn't belong to either of them.
    
    shape : tuple
        The pair of the number of colors/sets in the acceptance condition and the number of OA states; i.e., : `(n_accs, n_qs)`

    svg : str
        The string of the SVG representation of the OA within div tags.


    Parameters
    ----------
    ltl: str
        The linear temporal logic (LTL) formula to be translated to a OA.

    oa_type: str, optional
        The type of the OA to be constructed. The default value is `'dpa'`.
    
    """

    def __init__(self, ltl, oa_type='dpa'):

        self.oa_type = oa_type

        self.ltl = ltl
        self.hoa = self.ltl2hoa(ltl, oa_type)
        self.spot_oa = self.hoa2spot(self.hoa)
        aps, labels, q0, delta, epsmoves, acc, shape = self.spot2specs(self.spot_oa)

        self.aps = aps
        self.labels = labels
        self.q0 = q0
        self.delta = delta
        self.epsmoves = epsmoves
        self.acc = acc
        self.shape = shape

    def ltl2hoa(self, ltl, oa_type):
        """
        Returns the HOA representation of the OA obtained by executing the command-line tool `ltl2[oa_type]` with `ltl` as the input LTL formula.

        Parameters
        ----------
        ltl: str
            The linear temporal logic (LTL) formula to be transformed to a OA.
        
        oa_type: str
            The type of the OA to be constructed.
        
        Returns
        -------
        hoa: str
            The HOA representation of the OA constructed.
        
        """
        
        tool = 'ltl2'+oa_type
        hoa = check_output(['owl',tool,'-f',ltl,'-t','SMALLEST_AUTOMATON', '--complete'])
        return hoa

    def hoa2spot(self, hoa):
        """
        Constructs a spot omega automaton from the given OA description in the HOA format.
        
        Parameters
        ----------
        hoa: str
            The HOA representation of the OA.
        
        Returns
        -------
        spot_oa: spot.twa_graph
            The `spot.twa_graph` object constructed from the HOA representation of the OA.
        
        """

        filename = self.random_filename('hoa')
        with open(filename,'wb') as f:
            f.write(hoa)
        spot.setup()
        spot_oa = spot.automaton(filename)
        time.sleep(0.1)
        os.remove(filename)

        # Make the oa complete and change its acceptance condition to 'max odd' if it is a DPA
        spot_oa = spot.complete(spot_oa)
        if spot_oa.acc().is_parity()[0]:
            spot_oa = spot.colorize_parity(spot.change_parity(spot_oa,spot.parity_kind_max,spot.parity_style_odd),True)
            spot.highlight_nondet_states(spot_oa, 5)
            spot.highlight_nondet_edges(spot_oa, 4)
        
        return spot_oa

    def spot2specs(self, spot_oa):
        """
        Returns a tuple of OA specifications obtained from the spot OA object `spot_oa`.
        The tuple consists of the initial state, list representations of the transition function, the epsilon-moves, the acceptance condition, and the shape of the OA.

        Parameters
        ----------
        spot_oa: spot.twa_graph
            The spot OA object.
        
        Returns
        -------
        output: tuple
            A tuple containing the initial OA state `q0`; a list representations of the OA transition function `delta`;
            a list representation of the epsilon-moves `epsmoves`; a list representation of the OA acceptance condition `acc`;
            and the shape of the OA `shape=(n_accs, n_qs)`, the number of colors/pairs in the acceptance condition; i.e., `(q0, delta, epsmoves, acc, shape)`.

        """

        q0 = spot_oa.get_init_state_number()  # Get the initial state
        shape = n_accs, n_qs = spot_oa.num_sets(), spot_oa.num_states()  # The number of colors/sets, Number of OA states
        
        # Get the atomic propositions
        spot_ap_map = {str(a):a for a in spot_oa.ap()}  # The mapping from each atomic proposition to the corresponding `spot.formula` atomic formula
        aps = sorted(spot_ap_map.keys())  # The list of atomic propositions
        
        # Get the list of all possible labels
        # A label is a set of atomic propositions
        labels = self.powerset(aps)
        spot_clause_map = {}  # The mapping from each label to the corresponding `spot.formula` clause
        for label in labels:
            # Create the list of spot.formula literals for `label`
            literals = []
            for a in aps:
                if a in label:
                    literals.append(spot_ap_map[a])
                else:
                    literals.append(spot.formula_Not(spot_ap_map[a]))
            # Map `label` to the clause of `literals`
            clause = spot.formula_And(literals)
            spot_clause_map[label] = clause
        
        
        delta = [{label:[] for label in labels} for i in range(n_qs)]  # The transition function
        acc = [{label:[] for label in labels} for i in range(n_qs)]  # The transition colors
        epsmoves = [[] for q in range(shape[1])]  # The epsilon-moves for LDBAs TODO: This has not been implemented.
        
        for e in spot_oa.edges():
            color = e.acc.max_set()-1
            cond = spot.bdd_to_formula(e.cond)  # The transition condition expressed as a formula instead of a list of labels triggering the transition
            for label in labels:
                spot_clause = spot_clause_map[label]
                if spot.formula_Implies(spot_clause,cond).equivalent_to(spot.formula.tt()):  # If the transition condition satisfied with the clause corresponding to `label`
                    if spot_oa.is_deterministic():
                        delta[e.src][label] = e.dst
                    else:
                        delta[e.src][label].append(e.dst)
                    if spot_oa.acc().is_parity()[0]:
                        acc[e.src][label] = color
                    else:  # TODO: This has not been implemented.
                        acc[e.src][label].append(color)
        
        shape = max(sum(list(map(lambda x: list(x.values()), acc)),[]))+1, n_qs
        
        output = (aps, labels, q0, delta, epsmoves, acc, shape)
        return output

    def _repr_html_(self):
        """
        Returns the string of the SVG representation of the OA within div tags for visualization in a Jupyter notebook.

        Returns
        -------
        self.svg: str
            The string of the SVG representation of the OA within div tags.
        
        """

        return self.spot_oa._repr_html_()

    def powerset(self, a):
        """
        Returns the power set of the given list.

        Parameters
        ----------
        a: list
            The input list.

        Returns
        -------
        powerset: list
            The power set of the list.
        
        """

        powerset = list(chain.from_iterable(combinations(a, k) for k in range(len(a)+1)))
        return powerset
    
    def random_filename(self, extension):
        """
        Returns a random file name with the given extension.

        Parameters
        ----------
        extension: str  
            The extension of the file name to be generated.

        Returns
        -------
        filename: str
            A random file name.
        
        """
        time = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S_%f_')
        # Generate a nonexistent file name
        filename = 'temp_hoa_' + socket.gethostname() + time + ('%032x.' % random.getrandbits(128)) + extension
        while os.path.isfile(filename):
            time = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S_%f_')
            filename = 'temp_hoa_' + socket.gethostname() + time + ('%032x.' % random.getrandbits(128)) + extension
        return filename
