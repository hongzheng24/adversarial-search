from typing import Dict, Tuple, Optional, Union

from adversarial_search_problem import (
    Action,
    State as GameState,
)
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem


def minimax(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth: float = float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the minimax algorithm for adversarial search problems.
    
    Minimax is a recursive algorithm used for finding the optimal move in two-player,
    zero-sum games. It assumes both players play optimally by maximizing their own
    score and minimizing their opponent's score.

    Args:
        asp: A HeuristicAdversarialSearchProblem representing the game.
        cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
                     Uses heuristic evaluation when cutoff is reached.

    Returns:
        Tuple containing:
            - Best action to take from the current state
            - Dictionary with search statistics including 'states_expanded'
    """
    best_action = None
    stats = {
        'states_expanded': 0
    }

    # TODO: Implement the minimax algorithm. Feel free to write helper functions.
    def minimax_value(state, asp, depth, cutoff_depth=float('inf')):
        """
        Returns
        -------
        tuple[float, Action]:
            Tuple containing the value and the action.
        """
        stats['states_expanded'] += 1

        if asp.is_terminal_state(state):
            if not isinstance(asp.get_result(state), (int, float, complex)):
                raise Exception("Minimax terminal state is non numeric")
            return asp.get_result(state), None
        elif depth >= cutoff_depth:
            if not isinstance(asp.heuristic(state), (int, float, complex)):
                raise Exception("Minimax heuristic is non numeric")
            return asp.heuristic(state), None
        
        if state.player_to_move() == 0:
            return max_value(state, asp, depth + 1, cutoff_depth=cutoff_depth)
        elif state.player_to_move() == 1:
            return min_value(state, asp, depth + 1, cutoff_depth=cutoff_depth)
        else:
            raise Exception("Invalid player")

    def max_value(state, asp, depth, cutoff_depth=float('inf')):
        value, best_action = float('-inf'), None
        for action in asp.get_available_actions(state):
            successor = asp.transition(state, action)
            successor_value, _ = minimax_value(successor, asp, depth + 1, cutoff_depth=cutoff_depth)
            value, best_action = max(
                (value, best_action),
                (successor_value, action),
                key=lambda x: x[0]
            )
        return value, best_action

    def min_value(state, asp, depth, cutoff_depth=float('inf')):    

        value, best_action = float('inf'), None
        for action in asp.get_available_actions(state):
            successor = asp.transition(state, action)
            successor_value, _ = minimax_value(successor, asp, depth + 1, cutoff_depth=cutoff_depth)
            value, best_action = min(
                (value, best_action),
                (successor_value, action),
                key=lambda x: x[0]
            )
        return value, best_action

    start_state = asp.get_start_state()
    _, best_action = minimax_value(start_state, asp, 0, cutoff_depth=cutoff_depth)

    return best_action, stats


def alpha_beta(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth: float = float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implements the alpha-beta pruning algorithm for adversarial search.
    
    Alpha-beta pruning is an optimization of the minimax algorithm that eliminates
    branches that cannot possibly influence the final decision. It maintains two
    values (alpha and beta) representing the best options found so far for the
    maximizing and minimizing players respectively.

    Args:
        asp: A HeuristicAdversarialSearchProblem representing the game.
        cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
                     Uses heuristic evaluation when cutoff is reached.

    Returns:
        Tuple containing:
            - Best action to take from the current state
            - Dictionary with search statistics including 'states_expanded'
    """
    best_action = None
    stats = {
        'states_expanded': 0  # Increase by 1 for every state transition
    }
    
    # TODO: Implement the alpha-beta pruning algorithm. Feel free to use helper functions.
    def alpha_beta_pruning(state, asp, alpha, beta, depth, cutoff_depth=float('inf')):
        
        """
        Returns
        -------
        tuple[float, Action]:
            Tuple containing the value and the action.
        """
        stats['states_expanded'] += 1

        if asp.is_terminal_state(state):
            return asp.get_result(state), None
        elif depth >= cutoff_depth:
            return asp.heuristic(state), None
        
        if state.player_to_move() == 0:
            return max_value(state, asp, alpha, beta, depth + 1, cutoff_depth=cutoff_depth)
        elif state.player_to_move() == 1:
            return min_value(state, asp, alpha, beta, depth + 1, cutoff_depth=cutoff_depth)
        else:
            raise Exception("Invalid player")

    def max_value(state, asp, alpha, beta, depth, cutoff_depth=float('inf')):

        value, best_action = float('-inf'), None
        for action in asp.get_available_actions(state):
            successor = asp.transition(state, action)
            successor_value, _ = alpha_beta_pruning(
                successor, asp, alpha, beta, depth + 1, cutoff_depth=cutoff_depth
            )
            value, best_action = max(
                (value, best_action),
                (successor_value, action),
                key=lambda x: x[0]
            )
        if value >= beta:
            return value, best_action
        alpha = max(alpha, value)
        return value, best_action

    def min_value(state, asp, alpha, beta, depth, cutoff_depth=float('inf')):

        value, best_action = float('inf'), None
        for action in asp.get_available_actions(state):
            successor = asp.transition(state, action)
            successor_value, _ = alpha_beta_pruning(
                successor, asp, alpha, beta, depth + 1, cutoff_depth=cutoff_depth
            )
            value, best_action = min(
                (value, best_action),
                (successor_value, action),
                key=lambda x: x[0]
            )
        if value <= alpha:
            return value, best_action
        beta = min(beta, value)
        return value, best_action

    start_state = asp.get_start_state()
    _, best_action = alpha_beta_pruning(start_state, asp, -1, 1, 0, cutoff_depth=cutoff_depth)
    return best_action, stats