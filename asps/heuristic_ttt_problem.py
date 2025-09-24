from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from .ttt_problem import TTTState, TTTProblem

SPACE = " "
X = "X"  # Player 0 is X
O = "O"  # Player 1 is O
PLAYER_SYMBOLS = [X, O]


class HeuristicTTTProblem(TTTProblem, HeuristicAdversarialSearchProblem):
    def heuristic(self, state: TTTState) -> float:
        """
        TODO: Fill this out with your own heuristic function! You should make sure that this
        function works with boards of any size; if it only works for 3x3 boards, you won't be
        able to properly test ab-cutoff for larger board sizes!
        """
        board = state.board
        n = len(board)
        score = 0

        def find_score(line):
            if all(cell in (None, 'X') for cell in line):
                count = line.count('X')
                return count**2
            elif all(cell in (None, 'O') for cell in line):
                count = line.count('O')
                return -count**2
            return 0
        
        for row in board:
            score += find_score(row)
        
        for j in range(n):
            col = [board[i][j] for i in range(n)]
            score += find_score(col)

        score += find_score([board[i][i] for i in range(n)])
        score += find_score([board[i][n-1-i] for i in range(n)])

        return score
