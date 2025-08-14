"""
Grid World Implementation

"""
from itertools import product

import numpy as np
import gymnasium as gym


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from ipywidgets.widgets import IntSlider
from ipywidgets import interact


class GridWorldEnv(gym.Env):
    """
    This class implements a 2D grid world where an agent can move up, down, right or left.
    The `structure[i, j]` entry stores the type of the cell at `(i, j)`.
    If `structure[i, j]` is `'E'`, the cell is empty and the agent can move in any direction.
    If it is `'B'`, the cell is blocked and the agent cannot enter it.
    If it is one of `'D'`, `'U'`, `'R'`, or `'L'`, the agent can enter the cell from any direction but cannot leave
    in the direction opposite to the label. For example, if the label is `'D'`, the agent cannot move upward.
    If it is `'T'`, the cell is a trap: once the agent enters, it cannot leave.

    Attributes & Parameters
    ----------
    shape : (n_rows, n_cols)
        The shape of the grid world.

    structure : array, shape=(n_rows, n_cols), optional
        The structure of the grid function, `structure[i][j]` stores the cell type of the cell `(i,j)`.
        If `structure[i,j]` is `'E'` it means the cell is empty and the agent is free to move in any direction. If it is `'B'` then the cell is blocked, the agent cannot go there.
        If it is one of `'U'`,`'D'`,`'L'` and `'R'`, the agent is free to enter the cell in any direction, but it cannot leave the cell in the opposite direction of the type.
        For example, if the type is `'U'`, then the agent cannot go down as if there is an obstacle there.
        If it is `'T'`, then the cell is a trap cell, which means if the agent cannot leave the cell once it reaches it.
        The default value is `None`.

    rewards : array, shape = (n_rows, n_cols), optional
        The reward function, `reward[i,j]` is the reward for the cell `(i,j)`. If `reward[i,j]` is `None`, then the cell is occupied by an obstacle.
        The default value is `None`.

    labels : array, shape = (n_rows, n_cols), optional
        The labeling function, `label[i,j]` is the set of atomic propositions the cell `(i,j)` is labeled with.
        The default value is `None`.

    prob_intended : float, optional
        The probability that the agent moves in the intended direction. It moves in a perpendicular direction with a probability of `1-prob_intended` (`(1-prob_intended)/2` for each).
        The default value is `0.8`.

    figsize: int, optional
        The size of the matplotlib figure to be drawn when the method plot is called. The default value is `5`.

    lcmap: dict, optional
        The dictionary mapping labels to colors. The default value is `{}`.

    cmap: matplotlib.colors.Colormap, optional
        The colormap to be used when drawing the plot of the grid world. The default value is `matplotlib.cm.RdBu`.

    Other Attributes
    ----------
    action_dirs : list
        The list of available actions: `['U', 'D', 'R', 'L']`.

    s0 : tuple
        The initial state of the agent, which is `(0, 0)` by default.

    max_transitions : int
        The maximum number of different transitions for the agent to take in a single step, which is set to `3` by default.

    observation_space : gym.spaces.MultiDiscrete
        The observation space of the environment, which is a multi-discrete space representing the grid coordinates.

    action_space : gym.spaces.Discrete
        The action space of the environment, which is a discrete space representing the action directions.
     
    s : tuple
        The current state of the agent, initialized to `s0`.

    """

    def __init__(self, shape, structure=None, rewards=None, labels=None, prob_intended=0.8, figsize=5, lcmap={}, cmap=plt.cm.RdBu):

        self.action_dirs = ['U', 'D', 'R', 'L']  # The action list for "Up, Down, Right, Left"
        self.shape = shape

        # Create the default structure, reward and label if they are not defined.
        self.structure = structure if structure is not None else np.full(shape, 'E')
        self.rewards = rewards if rewards is not None else np.zeros(shape)
        if labels is not None:
            # Sort the atomic propositions in the tuples to be able to map the labels to the the automaton transitions 
            self.labels = np.copy(labels)
            for cell in self.cells():
                self.labels[cell] = tuple(sorted(labels[cell]))
        else:
            self.labels = np.empty(shape, dtype=object)
            self.labels.fill(())

        self.prob_intended = prob_intended

        self.figsize = figsize
        self.lcmap = lcmap
        self.cmap = cmap 

        self.s0 = (0, 0)  # The initial state of the agent
        self.max_transitions = 3  # The maximum number of different transitions for the agent to take in a single step

        self.observation_space = gym.spaces.MultiDiscrete(shape)  # The observation space is the grid coordinates
        self.action_space = gym.spaces.Discrete(len(self.action_dirs))  # The action space is the action directions

        self.s = self.s0  # The current state of the agent

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state.
        
        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator. The default value is `None`.
        options : dict, optional
            Additional options for resetting the environment. The default value is `None`.
        
        Returns
        -------
        s : tuple
            The initial state of the agent after reset.
        info : dict
            Additional information about the reset state. The default value is an empty dictionary.
        """
        self.s = self.s0  # Reset the agent's state to the initial state
        return self.s, {}
    
    def step(self, action):
        """Takes a step in the environment based on the action taken by the agent.
        Parameters
        ----------
        action : int
            The action taken by the agent, represented as an integer corresponding to the action direction.
        Returns
        -------
        s : tuple
            The new state of the agent after taking the action.
        r : float
            The reward received by the agent for taking the action in the current state.
        done : bool
            A boolean indicating whether the episode has ended. Always `False` in this implementation.
        truncated : bool
            A boolean indicating whether the episode has been truncated. Always `False` in this implementation.
        info : dict
            Additional information about the step, including the labels of the current state.
        """
        dst_cells, probs = self.get_transition_probs(self.s, action)       
        self.s = np.random.choice(dst_cells, p=probs)
        r = self.rewards[self.s]  # Get the reward for the current state
        info = {'labels': self.labels[self.s]}  # Get the labels for the current state
        return self.s, r, False, False, info
        
        

    def cells(self):
        """The cell generator.

        Yields
        ------
        cell: tuple
            The state coordinates.

        """

        n_rows, n_cols = self.shape
        for cell in product(range(n_rows), range(n_cols)):
            yield cell

    def random_cell(self):
        """Generates a random cell coordinate.

        Returns
        -------
        cell: tuple
            A random cell coordinate.

        """

        n_rows, n_cols = self.shape
        cell = np.random.randint(n_rows),np.random.randint(n_cols)
        return cell

    def get_transition_probs(self, cell, action_dir, attack_type=None, attack_dir=None):
        """Returns the list of possible destionation cells with their probabilities `(dst_cells, probs)` when the action with the name `action_dir` is taken in the cell at the coordinates specified by `cell`.
        The agent moves in the intented direction with a probability of `self.prob_intended`, and moves sideways with a probability of `(1-self.prob_intended)/2`.
        If exists, the adversarial action with the name `attack_dir` can further manipulate the movement:
        if the the adversarial action is in the opposite direction to the agent's action, the agent moves as described above (default); 
        if the the adversarial action is in the same direction, the agent moves in the intended direction with a probability of 1;
        if the the adversarial action is perpendicular to the agent's action, the agent moves in the intended direction with a probability of `self.prob_intended`, 
        and moves in the direction of the adversarial action with a probability of `1-self.prob_intended`.
        Lastly, if the cell in the direction of the movement is occupied by an obstacle or the agent is in a trap cell, the agent cannot move and stays in the same cell.

        # TODO: Implement support for action-override attack types.

        Parameters
        ----------
        cell : tuple
            The coordinate of the cell `(i,j)`,

        action_dir: str
            The name of the action.

        attack_dir: str, optional
            The name of the attack. The default value is `None`.

        Returns
        -------
        output: tuple
            The list of possible destionation cells and their probabilities.

        """
        if action_dir not in self.action_dirs:
            raise ValueError(f"Invalid action direction: {action_dir}. Expected one of {self.action_dirs}.")

        if attack_type not in [None, 'noise', 'override']:
            raise ValueError(f"Invalid attack type: {attack_type}. Expected one of [None, 'noise', 'override'].")
        
        n_rows, n_cols = self.shape
        if cell[0] < 0 or cell[0] >= n_rows:  # The row boundaries
            raise ValueError(f"Cell row {cell} is out of bounds for the grid of shape {self.shape}.")
        elif cell[1] < 0 or cell[1] >= n_cols:  # The column boundaries
            raise ValueError(f"Cell column {cell} is out of bounds for the grid of shape {self.shape}.")
       

        if attack_type is None:  # No attack
            return self.get_transition_probs_without_attack(cell, action_dir)

        else: # If there is an actuation attack
            if attack_dir not in self.action_dirs:
                raise ValueError(f"Invalid attack direction: {attack_dir}. Expected one of {self.action_dirs}.")

            # If the attack direction is the same as the agent's action direction (reinforcement attack)
            if action_dir == attack_dir:
                # The agent moves in the intended direction with a probability of 1
                dst = self.move(cell, action_dir)
                return [dst] * 3, [1, 0, 0] 

            # If the attack direction is in the opposite direction to the agent's action (noise attack)
            elif self.opposite(action_dir, attack_dir):
                # Noisy transitions as if there is no attack
                return self.get_transition_probs_without_attack(cell, action_dir)
            
            # If the attack direction is perpendicular to the agent's action (side attack)
            # then the agent moves in the intended direction with a probability of `self.prob_intended`,
            # and moves in the attack direction with a probability of `1-self.prob_intended`.
            else:  # elif self.perpendicular(action_dir,attack_dir):
                probs = []  # Reset the default transition probabilities
                dst_action = self.move(cell, action_dir)  # The destination cell in the action direction
                dst_attack = self.move(cell, attack_dir)  # The destination cell in the attack direction
                dst_cells = [dst_action, dst_attack, cell]  # The cell is a placeholder
                probs = [self.prob_intended, 1-self.prob_intended, 0]
                return dst_cells, probs

        # Should not reach here

    
    def get_transition_probs_without_attack(self, cell, action_dir):
        """Returns the list of possible destionation cells with their probabilities `(dst_cells, probs)` when the action with the name `action_dir` is taken in the cell at the coordinates specified by `cell`.
        The agent moves in the intented direction with a probability of `self.prob_intended`, and moves sideways with a probability of `(1-self.prob_intended)/2`.
        If the cell in the direction of the movement is occupied by an obstacle or the agent is in a trap cell, the agent cannot move and stays in the same cell.
        Parameters
        ----------
        cell : tuple
            The coordinate of the cell `(i,j)`.
        action_dir: str
            The name of the action.
        
        Returns
        -------
        output: tuple
            The list of possible destionation cells and their probabilities.

        """
        
        # Get the destination cells and their corresponding transition probabilities for each direction
        dst_cells, probs = [], []  # The destination cells and transition probabilities
        for direction in self.action_dirs:  # `['U', 'D', 'L', 'R']`
            if direction == action_dir:  # The intended direction
                dst_cells.append(self.move(cell, direction))
                probs.append(self.prob_intended)

            elif self.perpendicular(direction, action_dir):  # The directions perpendicular to the intended direction
                dst_cells.append(self.move(cell, direction))
                probs.append((1-self.prob_intended)/2)
            
            # else / elif self.opposite(action_dir, attack_dir):
                # Opposite direction; cannot move; ignore

        return dst_cells, probs
    

    def opposite(self, first_direction, second_direction):
        """Returns `True` if `first_direction` is in the opposite direction to `second_direction`, and `False` otherwise.

        Parameters
        ----------
        first_direction : tuple
            The first direction.

        second_direction : str, optional
            The second direction.
        
        Returns
        -------
        output: bool
            A Boolean value that is `True` if `first_direction` is in the opposite direction to `second_direction`, and `False` otherwise.

        """

        if first_direction=='U' and second_direction=='D':
            output = True
        elif first_direction=='D' and second_direction=='U':
            output = True
        elif first_direction=='L' and second_direction=='R':
            output = True
        elif first_direction=='R' and second_direction=='L':
            output = True
        else:
            output = False
        
        return output

    def perpendicular(self, first_direction, second_direction):
        """Returns `True` if `first_direction` is in the perpendicular direction to `second_direction`, and `False` otherwise.

        Parameters
        ----------
        first_direction : tuple
            The first direction.

        second_direction : str, optional
            The second direction.
        
        Returns
        -------
        output: bool
            A Boolean value that is `True` if `first_direction` is in the perpendicular direction to `second_direction`, and `False` otherwise.

        """

        if first_direction=='U' and second_direction in ['L', 'R']:
            output = True
        elif first_direction=='D' and second_direction in ['L', 'R']:
            output = True
        elif first_direction=='L' and second_direction in ['U', 'D']:
            output = True
        elif first_direction=='R' and second_direction in ['U', 'D']:
            output = True
        else:
            output = False

        return output

    def move(self, src, direction):
        """Returns the destination cell if the agent can move from the cell `src` towards `direction` and `None` otherwise.
        The agent cannot move from `src` if `src` is a trap cell or occupied by an obstacle.
        Similarly, the agent cannot move if the destination cell is occupied by an obstacle or beyond the walls.
        Lastly, the agent cannot move if the direction is blocked.

        Parameters
        ----------
        src : tuple
            The cell to be moved from.

        direction : str
            The direction to be moved in.
        
        Returns
        -------
        dst: tuple or None
            The destination cell
        
        """

        n_rows, n_cols = self.shape

        # Check if `src` is within the boundaries
        if src[0] < 0 or src[0] >= n_rows:  # The row boundaries
            raise ValueError(f"Source cell row {src} is out of bounds for the grid of shape {self.shape}.")
        elif src[1] < 0 or src[1] >= n_cols:  # The column boundaries
            raise ValueError(f"Source cell column {src} is out of bounds for the grid of shape {self.shape}.")
        
        # Check if `src` is a trap cell or occupied by an obstacle
        src_type = self.structure[src]
        if src_type in ['B', 'T']:
            return src
        
        # Get the neighbor cell in the direction
        if direction=='U':
            dst = (src[0]-1, src[1])
        elif direction=='D':
            dst = (src[0]+1, src[1])
        elif direction=='R':
            dst = (src[0], src[1]+1)
        elif direction=='L':
            dst = (src[0], src[1]-1)
        else:
            raise ValueError(f"Invalid direction: {direction}. Expected one of {self.action_dirs}.")

        # Check if `dst` is within the boundaries
        if dst[0] < 0 or dst[0] >= n_rows:  # The row boundaries
            return src
        elif dst[1] < 0 or dst[1] >= n_cols:  # The column boundaries
            return src

        # Check if `dst` is occupied by an obstacle
        dst_type = self.structure[dst]
        if dst_type == 'B':
            return src
        
        # Check if the direction is blocked
        if self.opposite(direction, src_type):
            return src
            
        return dst

    def plot(self, values=None, policy=None, adversarial_policy=None, agent=None, adversarial_agent=None, adversarial_agent_label=None, adversarial_override=False, save=None, hidden=[], path={}, title=None):
        """Plots the values of the cells as a color matrix.

        Parameters
        ----------
        values : array, shape=(n_mdps,n_qs,n_rows,n_cols), optional
            The value function. If it is None, the reward function will be plotted. The default value is `None`

        policy : array, shape=(n_mdps,n_qs,n_rows,n_cols), optional
            The policy to be visualized. The default value is `None`.

        agent : tuple, optional
            The position of the agent to be plotted. The default value is `None`.
        
        adversarial_agent : tuple, optional
            The position of the adversarial agent to be plotted. The default value is `None`.

        adversarial_agent_label : tuple, optional
            The label of the adversarial agent to be plotted. The default value is `None`.

        adversarial_override : bool, optional
            The Boolean value that is `True` if the action of the adversary overrides the action of the agent, and `False` otherwise.
            The default value is `False`.

        save : str, optional
            The name of the file the image will be saved to. The default value is `None`.
        
        hidden : list, optional
            The list of cells for which the actions will not be plotted. The default value is `[]`.

        path: dict, optional
            A dictionary mapping the cells to the fragments of the path to be highlighted. The path fragments should be specified in the form of `'(u|d|l|r)(u|d|l|r)?'`.
            For instance, `path[cell] = 'ur'` means there are two path fragments to be plotted in the cell at the coordinates specified by `cell`: from the top to the middle and from the middle to the right.
            The default value is `{}`.

        title : str, optional
            The title of the figure to be plotted. The default value is `None`.
        
        """

        # Set up the font
        f = FontProperties(weight='bold')
        fontname = 'Times New Roman'
        fontsize = 20

        # Set up the values to be displayed
        if values is None:
            values = self.rewards  # Plot the native rewards if a value matrix is not given
        else:
            values = np.copy(values)
            for h in hidden:  # Hide the values of the cells in `hidden`
                values[h] = 0


        # Plot the figure
        fig = plt.figure(figsize=(self.figsize,self.figsize))
        plt.rc('text', usetex=True)
        threshold = np.nanmax(np.abs(values))*2
        threshold = 1 if threshold==0 else threshold
        plt.imshow(values, interpolation='nearest', cmap=self.cmap, vmax=threshold, vmin=-threshold)
        if title:
            plt.title(title)

        # Get the dimensions
        n_rows, n_cols = self.shape

        # Get the axes
        ax = fig.axes[0]

        # Set the major ticks
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))

        # Set the labels for the major ticks
        ax.set_xticklabels(np.arange(n_cols), fontsize=fontsize)
        ax.set_yticklabels(np.arange(n_rows), fontsize=fontsize)

        # Set the labels for the minor ticks
        ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)

        # Move x axis to the top
        ax.xaxis.tick_top()

        # Show the grid lines based on the minor ticks
        ax.grid(which='minor',color='lightgray',linestyle='-',linewidth=1,alpha=0.5)

        # Hide the spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Hide the ticks on the bottom and the right
        ax.tick_params(bottom='off', left='off')

        # Draw the agent
        if agent:
            circle=plt.Circle((agent[1],agent[0]-0.17),0.26,color='lightblue',ec='darkblue',lw=2)
            ax.add_artist(circle)

        for i, j in self.cells():  # For all cells
            if (i,j) in path:  # Draw the path
                if 'u' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i+0.4),+0.8,-0.9,color='lightsteelblue')
                    plt.gcf().gca().add_artist(rect)
                if 'd' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i-0.4),+0.8,+0.9,color='lightsteelblue')
                    plt.gcf().gca().add_artist(rect)
                if 'l' in path[i,j]:
                    rect=plt.Rectangle((j+0.4,i-0.4),-0.9,+0.8,color='lightsteelblue')
                    plt.gcf().gca().add_artist(rect)
                if 'r' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i-0.4),+0.9,+0.8,color='lightsteelblue')
                    plt.gcf().gca().add_artist(rect)

            cell_type = self.structure[i,j]
            # If there is an obstacle
            if cell_type == 'B':
                circle=plt.Circle((j,i),0.49,color='k',fc='darkgray')
                plt.gcf().gca().add_artist(circle)
                continue
            # If it is a trap cell
            elif cell_type == 'T':
                circle=plt.Circle((j,i),0.49,color='k',fill=False)
                plt.gcf().gca().add_artist(circle)

            # If it is a directional cell (See the description of the class attribute `structure` for details)
            elif cell_type == 'U':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j+0.5,i+0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'D':
                triangle = plt.Polygon([[j,i],[j-0.5,i-0.5],[j+0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'L':
                triangle = plt.Polygon([[j,i],[j+0.5,i+0.5],[j+0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'R':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j-0.5,i-0.5]], color='gray')
                plt.gca().add_patch(triangle)

            # If the background is too dark, make the text and the arrow white
            color = 'snow' if np.abs(values[i, j]) > threshold/2 else 'black'
            intended_arrow_color = 'snow' if np.abs(values[i, j]) > threshold/2 else 'black'
            unintended_arrow_color = 'red' if np.abs(values[i, j]) > threshold/2 else 'red'

            if policy is None:  # Print the values if a policy matrix is not provided
                v = str(int(round(100*values[i,j]))).zfill(3)
                plt.text(j, i, '$'+v[0]+'.'+v[1:]+'$',horizontalalignment='center',color=color,fontname=fontname,fontsize=fontsize+2)  # Value

            # Draw the arrows to visualize the policy
            elif values[i,j] > 0 or values is self.rewards:  # Do not draw the arrows if the value is zero
                if policy[i,j] >= len(self.action_dirs):  # Display the epsilon-actions
                    plt.text(j, i-0.05,r'$\varepsilon_'+str(policy[i,j]-len(self.action_dirs)+1)+'$', horizontalalignment='center',color=color,fontsize=fontsize+5)
                else:
                    action_dir = self.action_dirs[policy[i,j]]
                    pos = j,i  # row,col => y,x
                    if action_dir == 'U':
                        plt.arrow(j,i,0,-0.2,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_dir == 'D':
                        pos = j,i-.3
                        plt.arrow(j,i-.3,0,0.2,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_dir == 'L':
                        pos = j+.15,i-0.15
                        plt.arrow(j+.15,i-0.15,-0.2,0,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_dir == 'R':
                        pos = j-.15,i-0.15
                        plt.arrow(j-.15,i-0.15,0.2,0,head_width=.2,head_length=.15,color=intended_arrow_color)
                    
                    if adversarial_policy is not None:
                        attack_dir = self.action_dirs[adversarial_policy[i,j]]
                        # Draw the adversarial action if the it is perpendicular to or overrides the agent's action 
                        if self.perpendicular(action_dir,attack_dir) or (adversarial_override and action_dir!=attack_dir):
                            if attack_dir == 'U':
                                plt.arrow(pos[0],pos[1],0,-0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_dir == 'D':
                                plt.arrow(pos[0],pos[1],0,0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_dir == 'R':
                                plt.arrow(pos[0],pos[1],0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_dir == 'L':
                                plt.arrow(pos[0],pos[1],-0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                        # Draw two arrows to represent that the agent can move sideways
                        elif self.opposite(action_dir,attack_dir):
                            if attack_dir in ['U','D']:
                                plt.arrow(pos[0],pos[1],0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                                plt.arrow(pos[0],pos[1],-0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            else:
                                plt.arrow(pos[0],pos[1],0,-0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
                                plt.arrow(pos[0],pos[1],0,0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
            
            # Plot circles around the labels
            surplus = 0.2 if (i,j) in hidden else 0
            if len(self.labels[i,j])==1:  # For a single atomic proposition
                l=self.labels[i,j][0]
                if l in self.lcmap:
                    circle=plt.Circle((j, i+0.25-surplus),0.2+surplus/2,color=self.lcmap[l])
                    plt.gcf().gca().add_artist(circle)

            elif len(self.labels[i,j])==2:  # For two atomic propositions
                l1, l2 = self.labels[i,j]
                if l1 in self.lcmap:
                    circle=plt.Circle((j-0.2, i+0.25-surplus),0.2+surplus/2,color=self.lcmap[l1])
                    plt.gcf().gca().add_artist(circle)
                if l2 in self.lcmap:
                    circle=plt.Circle((j+0.2, i+0.25-surplus),0.2+surplus/2,color=self.lcmap[l2])
                    plt.gcf().gca().add_artist(circle)

            # Plot the labels
            if self.labels[i,j]:
                    plt.text(j, i+0.4-surplus,'$'+','.join(self.labels[i,j])+'$',horizontalalignment='center',color=color,fontproperties=f,fontname=fontname,fontsize=fontsize+5+surplus*10)

        # Plot the adversarial agent
        if adversarial_agent:
            circle=plt.Circle((adversarial_agent[1],adversarial_agent[0]-0.17),0.26,color='lightpink',ec='deeppink',lw=2)
            ax.add_artist(circle)
            if adversarial_agent_label:
                plt.text(adversarial_agent[1]+0.01,adversarial_agent[0]-0.09,'$'+','.join(adversarial_agent_label)+'$',horizontalalignment='center',color=color,fontproperties=f,fontname=fontname,fontsize=fontsize+5)
        
        # Save the figure
        if save:
            plt.savefig(save,bbox_inches='tight')


    def plot_list(self,values_list,policy_list=None):
        """Plots the list of cell values with a slider.

        Parameters
        ----------
        values_list : list
            A list of value functions.

        policy_list : list, optional
            A list of policies. The default value is `None`.
        
        """

        # A helper function for the slider
        def plot_values(t):
            if policy_list is not None:
                self.plot(values_list[t],policy_list[t])
            else:
                self.plot(values_list[t])

        T = len(values_list)
        w=IntSlider(values=0,min=0,max=T-1)

        interact(plot_values,t=w)

    def _repr_png_(self):
        self.plot()



    def get_vectorized_transitions_rewards(self):
        """Returns the vectorized transitions and rewards for the grid world environment.
        The transition states are represented as a 4D array where the first two dimensions correspond to the grid shape, the third dimension corresponds to the action directions, and the last dimension corresponds to the destination states.
        The transition probabilities are represented as a 3D array where the first two dimensions correspond to the grid shape, the third dimension corresponds to the action directions, and the last dimension corresponds to the destination states.
        The rewards are represented as a 2D array corresponding to the grid shape.
        The shape of the transition states is `(n_rows, n_cols, n_actions, max_transitions, n_rows, n_cols)`, where `n_rows` and `n_cols` are the dimensions of the grid world, `n_actions` is the number of available
        actions, and `max_transitions` is the maximum number of different transitions for the agent to take in a single step.

        Returns
        -------
        transition_states : array, shape=(n_rows, n_cols, n_actions, max_transitions, n_rows, n_cols)
            The transition states for the grid world environment.
        transition_probs : array, shape=(n_rows, n_cols, n_actions, max_transitions)
            The transition probabilities for the grid world environment.
        rewards : array, shape=(n_rows, n_cols)
            The rewards for the grid world environment.
        """

        transition_shape = self.shape + (len(self.action_dirs), self.max_transitions, len(self.shape))

        transition_states = np.zeros(transition_shape, dtype=int)
        transition_probs = np.zeros(transition_shape[:-1], dtype=float)  # Ignore the last dimension, which is for destination states

        for row, col, action, dst_id in product(*map(range, transition_probs.shape)):
            cell = (row, col)
            action_dir = self.action_dirs[action]
            dst_cells, probs = self.get_transition_probs(cell, action_dir)
            transition_probs[row, col, action, dst_id] = probs[dst_id]
            transition_states[row, col, action, dst_id] = dst_cells[dst_id]

        
        return transition_states, transition_probs, self.rewards




