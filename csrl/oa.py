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
import re


try:
    import spot
except ImportError:
    raise ImportError('The "spot" library is required for this module. Please install it using the instructions on https://spot.lre.epita.fr/install.html.')

class OmegaAutomaton:
    """
    Transforms the LTL formula to an omega-automaton (OA) and stores the specifications.

    Attributes
    ----------
    ltl : str
        The linear temporal logic (LTL) formula used to construct the OA.

    oa_type : str
        The type of the OA to be constructed. It can be either `'dpa'` (deterministic parity automaton) or `'ldba'` (limit-deterministic Büchi automaton).
        The default value is `'dpa'`.

    self.hoa : bytes
        The HOA representation of the OA constructed from the LTL formula or read from the HOA file.

    hoa_path : str
        The path to the HOA file containing the OA description. If the OA is constructed from an LTL formula, this is the path to the generated HOA file.

    spot_oa : spot.twa_graph
        The `spot.twa_graph` object constructed from the HOA representation of the OA.

    aps : list of str
        The atomic propositions of the OA. These are the basic building blocks of the LTL formula.

    labels : list of str
        The labels of the OA. A label is a set of atomic propositions, and the labels are the power set of the atomic propositions.
        
    q0 : int
        The initial state of the OA.

    delta : list of dicts
        A list representation of the transition function of the OA.
        For DPAs, `delta[q][label]` is the OA state that the OA makes a transition to when the symbol `label` is consumed in the OA state `q`.
        For LDBAs, `delta[q][label]` is the list of OA states that the OA can make a nondeterministic transition to when the symbol `label` is consumed in the OA state `q`.

    acc : list of dicts
        A list representation of the acceptance condition of the OA. 
        For DPAs, `acc[q][label]` is the color of transition triggered by consuming the symbol `label` in the OA state `q`. 
        For LDBAs, `acc[q][label]` is the list of Boolean values indicating if the transition is accepting.
    
    shape : tuple
        The pair of the number of colors/sets in the acceptance condition and the number of OA states; i.e., : `(n_accs, n_qs)`


    Parameters
    ----------
    ltl: str
        The linear temporal logic (LTL) formula to be translated to a OA.

    oa_type: str, optional
        The type of the OA to be constructed. The default value is `'dpa'`.

    hoa_path: str, optional
        The path to the HOA file containing the OA description. If provided, the `ltl` parameter is ignored.

    keep_hoa: bool, optional
        If `True`, the HOA file will be kept after constructing the OA. The default value is `False`. Ignored if `hoa_path` is provided.

    save_svg: bool, optional
        If `True`, the SVG representation of the OA will be saved to a file. The default value is `False`.
        The SVG file will be saved in the same directory as the HOA file with the same name but with a `.svg` extension.
    
    """

    def __init__(self, ltl=None, oa_type='dpa', hoa_path=None, keep_hoa=False, save_svg=False):

        if ltl is None and hoa_path is None:
            raise ValueError('Either "ltl" or "hoa_path" must be provided to construct an OmegaAutomaton.')
        
        if hoa_path is None and oa_type not in ['dpa', 'ldba']:
            raise ValueError('Invalid OA type. Supported types are "dpa" and "ldba".')
        

        if hoa_path:
            with open(hoa_path, 'rb') as f:
                self.hoa = f.read()
            hoa_args = self.hoa.decode('utf-8').splitlines()[3].split('"')
            self.ltl = hoa_args[5].replace("'", '"')
            self.oa_type = hoa_args[1][4:]

        else:
            self.hoa = self.ltl2hoa(ltl, oa_type)
            hoa_path = self.random_hoa_path(ltl, oa_type)
            with open(hoa_path, 'wb') as f:
                f.write(self.hoa)
            self.ltl = ltl
            self.oa_type = oa_type


        self.hoa_path = hoa_path

        self.spot_oa = self.hoa2spot(hoa_path, save_svg)
        if hoa_path is None and not keep_hoa:
            os.remove(hoa_path)
        
        aps, labels, q0, delta, acc, shape = self.spot2specs(self.spot_oa)

        self.aps = aps
        self.labels = labels
        self.q0 = q0
        self.delta = delta
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

    def hoa2spot(self, hoa_path, save_svg=False):
        """
        Constructs a spot omega automaton from the given OA description in the HOA format.
        
        Parameters
        ----------
        hoa_path: str
            The path to the HOA file containing the OA description.

        save_svg: bool, optional
            If `True`, the SVG representation of the OA will be saved to a file. The default value is `False`.
            The SVG file will be saved in the same directory as the HOA file with the same name but with a `.svg` extension.
        
        Returns
        -------
        spot_oa: spot.twa_graph
            The `spot.twa_graph` object constructed from the HOA representation of the OA.
        
        """
        spot.setup()
        spot_oa = spot.automaton(hoa_path)

        # Make the oa complete and change its acceptance condition to 'max odd' if it is a DPA
        spot_oa = spot.complete(spot_oa)
        if spot_oa.acc().is_parity()[0]:
            spot_oa = spot.colorize_parity(spot.change_parity(spot_oa,spot.parity_kind_max,spot.parity_style_odd),True)
            spot.highlight_nondet_states(spot_oa, 5)
            spot.highlight_nondet_edges(spot_oa, 4)

        if save_svg:
            svg_path= hoa_path + '.svg'
            with open(svg_path, 'w') as f:
                f.write(spot_oa._repr_svg_())
            print(f'SVG representation of the OA saved to {svg_path}')
        
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
        aps: list of str
            The list of atomic propositions of the OA.
        labels: list of str
            The list of labels of the OA. A label is a set of atomic propositions, and the labels are the power set of the atomic propositions.
        q0: int
            The initial state of the OA.
        delta: list of dicts
            A list representation of the transition function of the OA.
            For DPAs, `delta[q][label]` is the OA state that the OA makes a transition to when the symbol `label` is consumed in the OA state `q`.
            For LDBAs, `delta[q][label]` is the list of OA states that the OA can make a nondeterministic transition to when the symbol `label` is consumed in the OA state `q`.
        acc: list of dicts
            A list representation of the acceptance condition of the OA.
            For DPAs, `acc[q][label]` is the color of transition triggered by consuming the symbol `label` in the OA state `q`.
            For LDBAs, `acc[q][label]` is the list of Boolean values indicating if the transition is accepting.
        shape: tuple
            The pair of the number of colors/sets in the acceptance condition and the number of OA states; i.e., : `(n_accs, n_qs)`

        """

        q0 = spot_oa.get_init_state_number()  # Get the initial state
        shape = n_accs, n_qs = spot_oa.num_sets(), spot_oa.num_states()  # The number of colors/sets, Number of OA states
        
        # Get the atomic propositions
        spot_ap_map = {str(a):a for a in spot_oa.ap()}  # The mapping from each atomic proposition to the corresponding `spot.formula` atomic formula
        aps = sorted(spot_ap_map.keys())  # The list of atomic propositions
        
        # Get the list of all possible labels
        # A label is a set of atomic propositions
        labels = list(chain.from_iterable(combinations(aps, k) for k in range(len(aps)+1)))  # The power set of `aps`
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
       
        for e in spot_oa.edges():
            color = e.acc.max_set()-1
            cond = spot.bdd_to_formula(e.cond)  # The transition condition expressed as a formula instead of a list of labels triggering the transition
            for label in labels:
                spot_clause = spot_clause_map[label]
                if spot.formula_Implies(spot_clause,cond).equivalent_to(spot.formula.tt()):  # If the transition condition satisfied with the clause corresponding to `label`
                    if spot_oa.is_deterministic():
                        delta[e.src][label] = e.dst
                        acc[e.src][label] = color
                    else:
                        delta[e.src][label].append(e.dst)
                        acc[e.src][label].append((color%2)==1)
                        
        n_accs = 2 if not spot_oa.is_deterministic() else max(sum(list(map(lambda x: list(x.values()), acc)),[]))+1
        shape = n_accs, n_qs
        
        output = (aps, labels, q0, delta, acc, shape)
        return output

    def _repr_html_(self):
        """
        Returns the string of the SVG representation of the OA within div tags for visualization in a Jupyter notebook.

        Returns
        -------
        svg: str
            The string of the SVG representation of the OA within div tags.
        
        """

        return self.spot_oa._repr_html_()

    
    def random_hoa_path(self, ltl, oa_type):
        """
        Returns a random file name with the given extension.

        Parameters
        ----------
        ltl: str
            The linear temporal logic (LTL) formula used to construct the OA.
        
        oa_type: str
            The type of the OA to be constructed.

        Returns
        -------
        hao_path: str
            A random HAO file path.
        
        """
        os.makedirs('.hoa', exist_ok=True)
        name = oa_type + '_' + re.sub('[^0-9a-zA-Z]+', '_', ltl) + '_'
        # Generate a nonexistent file name
        time = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S_%f_')
        filename = '.hoa' + os.path.sep + name + socket.gethostname() + time + ('%032x' % random.getrandbits(128)) + '.hoa'
        while os.path.isfile(filename):
            time = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S_%f_')
            filename = '.hoa' + os.path.sep + name + socket.gethostname() + time + ('%032x' % random.getrandbits(128)) + '.hoa'
        return filename
