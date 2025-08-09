"""
Grid World Implementation

"""

import numpy as np
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from ipywidgets.widgets import IntSlider
from ipywidgets import interact


class GridWorld():
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
    action_names : list
        The list of available actions: `['U', 'D', 'R', 'L']`.


    """

    def __init__(self, shape, structure=None, rewards=None, labels=None, prob_intended=0.8, figsize=6, lcmap={}, cmap=plt.cm.RdBu):

        self.action_names = ['U', 'D', 'R', 'L']  # The action list for "Up, Down, Right, Left"

        n_rows, n_cols = self.shape = shape

        # Create the default structure, reward and label if they are not defined.
        self.structure = structure if structure is not None else np.full(shape, 'E')
        self.rewards = rewards if rewards is not None else np.zeros((n_rows, n_cols))
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

    def cells(self):
        """
        The cell generator.

        Yields
        ------
        cell: tuple
            The state coordinates.

        """

        n_rows, n_cols = self.shape
        for cell in product(range(n_rows), range(n_cols)):
            yield cell

    def random_cell(self):
        """
        Generates a random cell coordinate.

        Returns
        -------
        cell: tuple
            A random cell coordinate.

        """

        n_rows, n_cols = self.shape
        cell = np.random.randint(n_rows),np.random.randint(n_cols)
        return cell

    def get_transition_probs(self, cell, action_name, attack_name=None, attack_type=None):
        """
        Returns the list of possible destionation cells with their probabilities `(dst_cells, probs)` when the action with the name `action_name` is taken in the cell at the coordinates specified by `cell`.
        The agent moves in the intented direction with a probability of `self.prob_intended`, and moves sideways with a probability of `(1-self.prob_intended)/2`.
        If exists, the adversarial action with the name `attack_name` can further manipulate the movement:
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

        action_name: str
            The name of the action.

        attack_name: str, optional
            The name of the attack. The default value is `None`.

        Returns
        -------
        output: tuple
            The list of possible destionation cells and their probabilities.

        """

        directions = self.action_names  # Same as the actions: `['U', 'D', 'L', 'R']`

        # Get the default transition probabilities for each direction (without considering obstacles/walls/traps)
        probs = []  # The transition probabilities
        for direction in directions:
            if direction == action_name:  # The intended direction
                probs.append(self.prob_intended)
            elif self.perpendicular(direction,action_name):  # The directions perpendicular to the intended direction
                probs.append((1-self.prob_intended)/2)
            else:
                probs.append(0)
        
        # If there is an adversarial action, update the probabilities accordingly
        if attack_name:
            # If the adversarial action is in the opposite direction to the agent's action, the transition probabilities stay the same.
            if self.opposite(action_name, attack_name):
                pass
            # If the both actions are in the same direction, then the agent surely moves in the intended direction.
            elif action_name == attack_name:
                probs = []  # Reset the default transition probabilities
                for direction in directions:
                    if direction == action_name:  # The intended direction
                        probs.append(1)
                    else:
                        probs.append(0)
            # If the the actions are perpendicular to each other,
            # then the agent moves in the intended direction with a probability of `self.prob_intended`,
            # and moves in the direction of the adversarial action with a probability of `1-self.prob_intended`.
            elif self.perpendicular(action_name,attack_name):
                probs = []  # Reset the default transition probabilities
                for direction in directions:
                    if direction == action_name:  # The intended direction
                        probs.append(self.prob_intended)
                    elif direction == attack_name:  # The direction of the adversarial action
                        probs.append(1-self.prob_intended)
                    else:
                        probs.append(0)
        
        # Get the destination cells for each direction and update the probabilities accordingly if the agent cannot move
        dst_cells = []  # The destination cells
        for direction, prob in zip(directions, probs):
            dst = self.move(cell,direction)
            # If the agent cannot move due to being in a trap cell or trying to move towards an obstacle or a wall,
            # the associated probabilities should be assigned to the cell the agent is in.
            if dst is not None and prob > 0:  # If the agent can move
                dst_cells.append(dst)
            else:  # If the agent cannot move
                dst_cells.append(cell)

        output = (dst_cells, probs)
        return output
    
    def opposite(self, first_direction, second_direction):
        """
        Returns `True` if `first_direction` is in the opposite direction to `second_direction`, and `False` otherwise.

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
        """
        Returns `True` if `first_direction` is in the perpendicular direction to `second_direction`, and `False` otherwise.

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
        """
        Returns the destination cell if the agent can move from the cell `src` towards `direction` and `None` otherwise.
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
            The destination cell if the agent can move, `None` otherwise.
        
        """

        n_rows, n_cols = self.shape

        # Check if `src` is within the boundaries
        if src[0] < 0 or src[0] >= n_rows:  # The row boundaries
            return
        elif src[1] < 0 or src[1] >= n_cols:  # The column boundaries
            return
        
        # Check if `src` is a trap cell or occupied by an obstacle
        src_type = self.structure[src]
        if src_type in ['B', 'T']:
            return
        
        # Get the neighbor cell in the direction
        if direction=='U':
            dst = (src[0]-1, src[1])
        elif direction=='D':
            dst = (src[0]+1, src[1])
        elif direction=='L':
            dst = (src[0], src[1]-1)
        elif direction=='R':
            dst = (src[0], src[1]+1)

        # Check if `dst` is within the boundaries
        if dst[0] < 0 or dst[0] >= n_rows:  # The row boundaries
            return
        elif dst[1] < 0 or dst[1] >= n_cols:  # The column boundaries
            return

        # Check if `dst` is occupied by an obstacle
        dst_type = self.structure[dst]
        if dst_type == 'B':
            return
        
        # Check if the direction is blocked
        if self.opposite(direction,src_type):
            return
            
        return dst

    def plot(self, values=None, policy=None, adversarial_policy=None, agent=None, adversarial_agent=None, adversarial_agent_label=None, adversarial_override=False, save=None, hidden=[], path={}, title=None):
        """
        Plots the values of the cells as a color matrix.

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
        f=FontProperties(weight='bold')
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
                if policy[i,j] >= len(self.action_names):  # Display the epsilon-actions
                    plt.text(j, i-0.05,r'$\varepsilon_'+str(policy[i,j]-len(self.action_names)+1)+'$', horizontalalignment='center',color=color,fontsize=fontsize+5)
                else:
                    action_name = self.action_names[policy[i,j]]
                    pos = j,i  # row,col => y,x
                    if action_name == 'U':
                        plt.arrow(j,i,0,-0.2,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_name == 'D':
                        pos = j,i-.3
                        plt.arrow(j,i-.3,0,0.2,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_name == 'L':
                        pos = j+.15,i-0.15
                        plt.arrow(j+.15,i-0.15,-0.2,0,head_width=.2,head_length=.15,color=intended_arrow_color)
                    elif action_name == 'R':
                        pos = j-.15,i-0.15
                        plt.arrow(j-.15,i-0.15,0.2,0,head_width=.2,head_length=.15,color=intended_arrow_color)
                    
                    if adversarial_policy is not None:
                        attack_name = self.action_names[adversarial_policy[i,j]]
                        # Draw the adversarial action if the it is perpendicular to or overrides the agent's action 
                        if self.perpendicular(action_name,attack_name) or (adversarial_override and action_name!=attack_name):
                            if attack_name == 'U':
                                plt.arrow(pos[0],pos[1],0,-0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_name == 'D':
                                plt.arrow(pos[0],pos[1],0,0.1,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_name == 'R':
                                plt.arrow(pos[0],pos[1],0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                            elif attack_name == 'L':
                                plt.arrow(pos[0],pos[1],-0.1,0,head_width=.13,head_length=.07,color=unintended_arrow_color)
                        # Draw two arrows to represent that the agent can move sideways
                        elif self.opposite(action_name,attack_name):
                            if attack_name in ['U','D']:
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
        """
        Plots the list of cell values with a slider.

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