import numpy as np
import copy


class MinimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def evaluate_board(self, board, player):
        # This is a simple heuristic: count the pieces. More sophisticated evaluations can be used.
        return np.sum(board == player) - np.sum(board == -player)

    def minimax(self, board, depth, maximizingPlayer, env, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or env.is_game_over():
            return self.evaluate_board(board, 1 if maximizingPlayer else -1), None

        legal_moves = env.get_legal_moves(1 if maximizingPlayer else -1)
        best_move = None
        if maximizingPlayer:
            maxEval = float('-inf')
            for move in legal_moves:
                new_env = copy.deepcopy(env)  # Make a copy of the environment to simulate the move
                action_index = env.move_to_action_index(move)
                if action_index is not None and action_index >= 0:
                    new_env.step(action_index)
                    eval, _ = self.minimax(new_env.board, depth - 1, False, new_env, alpha, beta)
                    if eval > maxEval:
                        maxEval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                else:
                    # Handle invalid move
                    print("Invalid move attempted: ", move)
                    continue  # Skip this move
            return maxEval, best_move
        else:
            minEval = float('inf')
            for move in legal_moves:
                new_env = copy.deepcopy(env)
                new_env.step(env.move_to_action_index(move))
                eval, _ = self.minimax(new_env.board, depth - 1, True, new_env, alpha, beta)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval, best_move

    def act(self, board, env):
        _, best_move = self.minimax(board, self.depth, True, env)
        action_index = env.move_to_action_index(best_move) if best_move else None

        if action_index is not None and action_index >= 0:
            return action_index
        else:
            print("Fallback to a random valid move.")
            legal_moves = env.get_legal_moves(env.current_player)

            # Ensure there's at least one legal move to choose from
            if legal_moves:
                # When legal_moves is a list of complex objects like tuples, np.random.choice cannot be used directly.
                # Instead, select a random index and then get the move at that index.
                random_index = np.random.randint(len(legal_moves))  # Select a random index
                random_move = legal_moves[random_index]  # Get the move at the randomly selected index
                return env.move_to_action_index(random_move)
            else:
                # Handle the case where there are no legal moves available
                print("No valid moves available.")
                return None


