import unittest

from asps.game_dag import DAGState, GameDAG
from adversarial_search import alpha_beta, minimax


class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take.
    """

    def _check_result(self, result: object, dag: GameDAG) -> None:
        """
        Tests whether the result is one of the possible actions of the DAG.
        
        Args:
            result: The return value of an adversarial search algorithm (should be an action)
            dag: The GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potential_actions = dag.get_available_actions(start_state)
        self.assertIn(result, potential_actions, "Output should be an available action")

    def _run_dag_test(self, algorithm: callable, dag: GameDAG, correct_results: list[object], correct_stats: list[int]) -> None:
        result, stats = algorithm(dag)
        
        self._check_result(result, dag)

        # Check results
        self.assertIn(result, correct_results)

        # Check stats
        if correct_stats:
            self.assertEqual(stats["states_expanded"], correct_stats[0])
        return

    def test_minimax(self) -> None:
        """
        Test minimax algorithm on a basic GameDAG.
        
        Creates a simple game tree and verifies that minimax returns
        a valid action from the available set.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = minimax(dag)
        self._check_result(result, dag)

    def test_minimax_dag(self) -> None:
        X = True
        _ = False

        # Given case
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1], [7])

        # Cyclic case
        matrix = [
            [_, X, _],
            [_, _, X],
            [X, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(minimax, dag, None, None)

        # Matchstick case in which Player 0 goes first
        matrix = [
            [_, X, X, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, X, X, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, X, X, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, X, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, X, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, X, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, X],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {9: -1., 10: -1., 12: -1., 13: 1., 14: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1, 2], [15])

        # Matchstick case in which Player 2 goes first
        matrix = [
            [_, X, X, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, X, X, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, X, X, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, X, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, X, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, X, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, X],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 1)
        terminal_evaluations = {9: -1., 10: -1., 12: -1., 13: 1., 14: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1, 2], [15])

        # Positive value case
        matrix =[
            [_, X, X, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _],
            [_, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, X, X],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {4: -10, 5: 5, 6: 10, 7: 10, 8: 9}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [2], [9])

        # One node case
        # matrix = [[_]]
        # start_state = DAGState(0, 0)
        # terminal_evaluations = {0: 1}
        # dag = GameDAG(matrix, start_state, terminal_evaluations)
        # self._run_dag_test(minimax, dag, [0], [1])

        # Two node case
        matrix = [[_, X], [_, _]]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1], [2])

        # Self-directed graph case
        matrix = [
            [X, _, _],
            [_, X, _],
            [_, _, X]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {2: 1}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(minimax, dag, None, None)
        
        # Disconnected graph case
        matrix = [
            [_, _, _],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {2: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        with self.assertRaises(Exception):
            self._run_dag_test(minimax, dag, None, None)

        # Simple graph case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1], [3])

        # Multiple solutions case
        my_matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: 1}
        dag = GameDAG(my_matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1, 2], [3])

        # Case in which graph does not start at node 0
        matrix = [
            [_, _, _],
            [X, _, X],
            [_, _, _]
        ]
        start_state = DAGState(1, 0)
        terminal_evaluations = {0: 1, 2: -1}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(minimax, dag, [0], [3])
        
        # Graph is None case
        with self.assertRaises(Exception):
            self._run_dag_test(minimax, None, None, None)
        
        # No start state case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = None
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        with self.assertRaises(Exception):
            self._run_dag_test(minimax, dag, None, None)

        # No terminal states case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = None
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(minimax, dag, None, None)

        # Graph in which terminal nodes have children nodes
        matrix = [
            [_, X, X, _, _],
            [_, _, _, X, _],
            [_, _, _, _, X],
            [_, _, _, _, _],
            [_, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(minimax, dag, [1], [3])

    def test_alpha_beta(self) -> None:
        """
        Test alpha-beta pruning algorithm on a basic GameDAG.
        
        Creates a simple game tree and verifies that alpha-beta returns
        a valid action from the available set.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = alpha_beta(dag)
        self._check_result(result, dag)

    def test_alpha_beta_dag(self) -> None:
        X = True
        _ = False
        
        # Given case
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1], [7])

        # Cyclic case
        matrix = [
            [_, X, _],
            [_, _, X],
            [X, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(alpha_beta, dag, None, None)

        # Matchstick case in which Player 0 goes first
        matrix = [
            [_, X, X, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, X, X, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, X, X, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, X, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, X, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, X, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, X],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {9: -1., 10: -1., 12: -1., 13: 1., 14: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1, 2], [15])

        # Matchstick case in which Player 2 goes first
        matrix = [
            [_, X, X, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, X, X, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, X, X, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, X, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, X, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, X, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, X],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 1)
        terminal_evaluations = {9: -1., 10: -1., 12: -1., 13: 1., 14: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1, 2], [15])

        # Positive value case
        matrix =[
            [_, X, X, _, _, _, _, _, _],
            [_, _, _, X, X, X, _, _, _],
            [_, _, _, _, _, _, X, _, _],
            [_, _, _, _, _, _, _, X, X],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {4: -10, 5: 5, 6: 10, 7: 10, 8: 9}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [2], [9])

        # One node case
        # matrix = [[_]]
        # start_state = DAGState(0, 0)
        # terminal_evaluations = {0: 1}
        # dag = GameDAG(matrix, start_state, terminal_evaluations)
        # self._run_dag_test(alpha_beta, dag, [0], [1])

        # Two node case
        matrix = [[_, X], [_, _]]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1], [2])

        # Self-directed graph case
        matrix = [
            [X, _, _],
            [_, X, _],
            [_, _, X]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {2: 1}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(alpha_beta, dag, None, None)
        
        # Disconnected graph case
        matrix = [
            [_, _, _],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {2: 1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        with self.assertRaises(Exception):
            self._run_dag_test(alpha_beta, dag, None, None)

        # Simple graph case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1], [3])

        # Multiple solutions case
        my_matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: 1}
        dag = GameDAG(my_matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1, 2], [3])

        # Case in which graph does not start at node 0
        matrix = [
            [_, _, _],
            [X, _, X],
            [_, _, _]
        ]
        start_state = DAGState(1, 0)
        terminal_evaluations = {0: 1, 2: -1}
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(alpha_beta, dag, [0], [3])
        
        # Graph is None case
        with self.assertRaises(Exception):
            self._run_dag_test(alpha_beta, None, None, None)
        
        # No start state case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = None
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        with self.assertRaises(Exception):
            self._run_dag_test(alpha_beta, dag, None, None)

        # No terminal states case
        matrix = [
            [_, X, X],
            [_, _, _],
            [_, _, _]
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = None
        with self.assertRaises(Exception):
            dag = GameDAG(matrix, start_state, terminal_evaluations)
            self._run_dag_test(alpha_beta, dag, None, None)

        # Graph in which terminal nodes have children nodes
        matrix = [
            [_, X, X, _, _],
            [_, _, _, X, _],
            [_, _, _, _, X],
            [_, _, _, _, _],
            [_, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        self._run_dag_test(alpha_beta, dag, [1], [3])

if __name__ == "__main__":
    unittest.main()
