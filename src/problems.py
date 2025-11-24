from __future__ import annotations  # needed in order to reference a Class within itself
from random import randint, random, choice, choices
from typing import Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import numpy as np


# makes the state both immutable and hashable
T = TypeVar("T", bound=tuple)

class Node:
    """
    Represents a node in a search tree that contains a state.

    Attributes:
        count (int): Class-level counter tracking created nodes.
        state (T): Current state of the problem.
        parent (Node | None): Parent node containing the prior state.
        action (str | None): Action taken to reach this state from the parent.
        path_cost (float): Cumulative cost of reaching this node.
        id (int): Unique identifier for the node instance.
        depth (int): Depth of the node in the search tree.
    """
    count: int = 0
    """Class-level counter tracking created nodes."""

    def __init__(self, state: T, parent: Node = None, action: str = None, path_cost: float = 0):
        """
        Initializes a Node.

        Args:
            state (T): Current state.
            parent (Node | None): Parent node. Defaults to None.
            action (str | None): Action taken from the parent. Defaults to None.
            path_cost (float): Path cost from the parent. Defaults to 0.
        """
        self.state:T = state
        self.parent:Node = parent
        self.action:str = action
        self.path_cost:float = path_cost
        self.id:int = Node.count
        if self.parent is None:
            self.depth:int = 0
        else:
            self.depth:int = self.parent.depth + 1
        Node.count += 1

    def __str__(self) -> str:
        """String representation of the node"""
        ret = "ID: " + str(self.id) + "\n"
        ret += "State: \n" + str(self.state) + "\n"
        ret += "ParentID: " + ("None" if self.parent is None else str(self.parent.id)) + "\n"
        ret += "Action: " + str(self.action) + "\n"
        ret += "Cost: " + str(self.path_cost) + "\n"
        ret += "Depth: " + str(self.depth) + "\n"
        return ret

    def __repr__(self): return '<{}>'.format(self.state)

    def __hash__(self):
        """Stubbing this out so if you try to use it, it will throw an error."""
        raise NotImplementedError

    def __eq__(self, other):
        """Stubbing this out so if you try to use it, it will throw an error."""
        raise NotImplementedError

    def __copy__(self):
        """Stubbing this out so if you try to use it, it will throw an error."""
        raise NotImplementedError

    @staticmethod
    def path_actions(node: Node) -> list[str]:
        """
        Returns the sequence of actions from the root to a node.

        Args:
            node (Node): Target node.

        Returns:
            list[str]: Actions leading to the node.
        """

        root = node
        p = []

        while root is not None:
            p.append(root.action)
            root = root.parent

        # removes the last action which is just None
        p.pop()
        p.reverse()
        return p

    @staticmethod
    def expand(node: Node, problem: Problem) -> list[Node]:
        """
        Expands a node into nodes with neighboring states.

        Args:
            node (Node): Current node.
            problem (Problem): Search problem.

        Returns:
            list[Node]: Nodes containing neighboring states.
        """
        curr_state = node.state
        ret = []
        for a in problem.actions(curr_state):
            next_state = problem.result(curr_state, a)
            cost = problem.action_cost(curr_state, a, next_state)
            ret.append(Node(next_state, node, a, node.path_cost + cost))
        return ret

    @staticmethod
    def is_cycle(node: Node, k: int = 30) -> bool:
        """
        Checks if the node forms a cycle within k ancestors.

        Args:
            node (Node): Node to check.
            k (int): Max number of ancestors. Defaults to 30.

        Returns:
            bool: True if a cycle exists, False otherwise.
        """
        def find_cycle(ancestor:Node, _k:int) -> bool:
            if ancestor is None: # you're at root and done
                return False
            elif _k > 0:
                # seen the current state before
                if ancestor.state == node.state:
                    return True
                else:
                    return find_cycle(ancestor.parent, _k - 1)
            else:
                return False
        return find_cycle(node.parent, k)

class Problem(ABC, Generic[T]):
    """
    Interface representing a generic problem formulation.

    This class defines the operations needed for search algorithms.
    """

    def __init__(self, initial_state: T):
        """
        Initializes a problem with an initial state.

        Args:
            initial_state (T): The starting state of the problem.
        """
        self.initial_state = initial_state

    @abstractmethod
    def actions(self, state: T) -> list[int]:
        """
        Returns a list of legal actions available from the given state.

        Args:
            state (T): Current state.

        Returns:
            List[str]: List of actions available from the state.
        """
        pass

    @abstractmethod
    def is_goal(self, curr_state: T) -> bool:
        """
        Determines whether the given state is a goal state.

        Args:
            curr_state (T): State to check.

        Returns:
            bool: True if curr_state is a goal state, False otherwise.
        """
        pass

    @abstractmethod
    def result(self, curr_state: T, action: int) -> T:
        """
        Returns the state that results from applying an action to the current state.

        Args:
            curr_state (T): Current state.
            action (str): Action to apply.

        Returns:
            T: The resulting state after applying the action.
        """
        pass

    @abstractmethod
    def action_cost(self, curr_state: T, action: int, next_state: T) -> float:
        """
        Returns the cost of transitioning from curr_state to next_state via the given action.

        Args:
            curr_state (T): Current state.
            action (str): Action applied.
            next_state (T): Resulting state after applying the action.

        Returns:
            float: Cost associated with the transition.
        """
        pass

    @abstractmethod
    def value(self, curr_state: T) -> float:
        """
        Performance measure of the current state, used for local search.

        Args:
            curr_state (T): Current state.

        Returns:
            float: Fitness or score of the current state.
        """
        pass

    def verify(self, actions: np.array[int]) -> bool:
        """
        Verifies that a sequence of actions leads from the initial state to a goal state.

        Args:
            actions (List[int]): Sequence of actions to test.

        Returns:
            bool: True if the actions reach a goal state, False otherwise.
        """
        new_state = self.initial_state
        for action in actions:
            new_state = self.result(new_state, action)
        return self.is_goal(new_state)
    
class CGOL_Problem(Problem[T]):
    """
    Version of Problem used for Local Search algorithms Hill Climbing, Genetic Algorithm, and Novelty Search.

    Implements for Conway's Game of Life, with a list of values representing the 20x20 space.
    """
    def __init__(self, state_generator: Callable[[], T]):
        super().__init__(state_generator())
        self.state_generator = state_generator

    def actions(self, curr_state: np.array[int,...]) -> list[int]:
        """
        Actions that can be applied to a CGoL board.
        We can technically set multiple cells at a time, but we will limit to one per epoch.

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: List of cells that can be set.

        """
        return [i for i in range(len(curr_state))]

    def result(self, curr_state: np.array[int,...], action:int) -> np.array[int,...]:
        """
        Sets corresponding cell in the state.

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.
            action: the index to set.

        Returns: the next state representation after the given action is performed.

        """
        next_state = curr_state.copy()
        next_state[action] ^= 1     # XOR toggle the cell

        return next_state


    def is_dead(self, curr_state: np.array[int,...]) -> bool:
        """
        Checks if the board has any cells alive.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: True if there are no cells alive, false otherwise.
        
        """
        return np.sum(curr_state) < 1

    def action_cost(self, curr_state: np.array[int,...], action: str, next_state: np.array[int,...]) -> float:
        return 1

    def quality(self, curr_state: np.array[int,...]) -> float:
        """
        Quality of the board.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: the quality value of the board.
        """

        # TODO figure this out lol
    
    def novelty(self, curr_state: np.array[int,...]) -> float:
        """
        Novelty of the board.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: the novelty value of the board

        """

    def fitness_proportionate_selection(self, fitness_values:list[float]) -> int:
        """
        Performs fitness proportionate selection.

        The probability of returning the index of an individual state in the population is
        proportional to its fitness value when compared to the fitness value of the entire population.
        If the fitness values are [2, 3, 1, 3, 1] then the sum is 10, which makes the probability of
        choosing each index to be [2/10, 3/10, 1/10, 3/10, 1/10] -> [0.2, 0.3, 0.1, 0.3, 0.1]

        :param fitness_values: List of fitness values for each individual in the population.
        :return: Index of individual to choose
        """
        probs = [val / sum(fitness_values) for val in fitness_values]
        indeces = [i for i in range(len(fitness_values))]
        return choices(indeces, weights=probs, k=1)[0]


    def selection(self, n:int, population: list[T],
                  weights: list[float]) -> [T,...]:
        """
        Selects n individuals from a population based on fitness proportionate selection method defined above.
        Args:
            n: Number of individuals to select.
            population: List of the current individuals (states) in the population.
            weights: List of the fitness values for each state.

        Returns: N selected individuals.
        """
        indeces = []
        while len(indeces) < n:
            index = self.fitness_proportionate_selection(weights)
            if index not in indeces:    # ensure we are *not* sampling with replacement
                indeces.append(index)
        return [population[i] for i in indeces]


    def crossover(self, parent1:T, parent2:T) -> T:
        """
        Creates a new state by implementing single point crossover.

        Args:
            parent1: Parent state to crossover. Should not be modified.
            parent2: Parent state to crossover. Should not be modified.

        Returns: New state created by using single-point crossover.
        """
        c = randint(1,len(parent1)-1)
        return parent1[0:c] + parent2[c:]