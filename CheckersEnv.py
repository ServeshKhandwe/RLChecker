import numpy as np


class CheckersEnv:
    def __init__(self):
        self.board_size = 8
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        for i in range(3):
            for j in range((i % 2), self.board_size, 2):
                self.board[i][j] = 1
                self.board[self.board_size - 1 - i][self.board_size - 1 - j] = -1
        self.current_player = 1
        return self.board

    def get_legal_moves(self, player):
        moves = []
        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == player:
                    for d in directions:
                        new_i, new_j = i + d[0], j + d[1]
                        if 0 <= new_i < self.board_size and 0 <= new_j < self.board_size:
                            if self.board[new_i][new_j] == 0:
                                moves.append(((i, j), (new_i, new_j)))
                            elif self.board[new_i][new_j] == -player:
                                capture_i, capture_j = new_i + d[0], new_j + d[1]
                                if 0 <= capture_i < self.board_size and 0 <= capture_j < self.board_size and \
                                        self.board[capture_i][capture_j] == 0:
                                    moves.append(((i, j), (capture_i, capture_j)))
        return moves

    def step(self, action_index):
        legal_moves = self.get_legal_moves(self.current_player)
        if action_index >= len(legal_moves):
            raise ValueError("Invalid action index.")
        action = legal_moves[action_index]
        src, dest = action

        self.board[src[0], src[1]] = 0
        self.board[dest[0], dest[1]] = self.current_player
        if abs(src[0] - dest[0]) == 2:  # Capture move
            mid_i = (src[0] + dest[0]) // 2
            mid_j = (src[1] + dest[1]) // 2
            self.board[mid_i, mid_j] = 0  # Remove the captured piece

        reward = 0
        done = False
        if not self.get_legal_moves(-self.current_player):  # Check if the opponent has no legal moves
            reward = 1
            done = True

        self.current_player = -self.current_player  # Switch players
        return self.board, reward, done

    # Add a method to translate an action index to the corresponding move
    # This might already be part of how you generate legal moves
    def index_to_action(self, action_index):
        legal_moves = self.get_legal_moves(self.current_player)
        if action_index < len(legal_moves):
            return legal_moves[action_index]
        else:
            return None  # Or handle invalid action index differently

    def render(self):
        print("  " + " ".join([str(i) for i in range(self.board_size)]))
        for i in range(self.board_size):
            print(str(i) + " " + " ".join([str(int(x)) for x in self.board[i]]))
