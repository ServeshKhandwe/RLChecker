import numpy as np



class AgentEvaluator:

    def __init__(self, dqn_agent, minimax_agent, env):
        self.dqn_agent = dqn_agent
        self.minimax_agent = minimax_agent
        self.env = env

    def simulate_game(self, num_turns=100):
        state = self.env.reset()
        for turn in range(num_turns):
            # Determine which agent should act based on the current player
            if self.env.current_player == 1:
                action = self.dqn_agent.act(np.reshape(state, [1, -1]))
            else:
                action = self.minimax_agent.act(self.env.board, self.env)

            # Validate the chosen action
            legal_actions = self.env.get_legal_moves(self.env.current_player)
            if action not in [self.env.move_to_action_index(move) for move in legal_actions]:
                print("Invalid action chosen:", action)
                action = np.random.choice(
                    [self.env.move_to_action_index(move) for move in legal_actions])  # Choose a valid action randomly

            next_state, reward, done = self.env.step(action)
            if done:
                return self.env.current_player  # Return the winner
            state = next_state
        return 0  # Return 0 if the game ends in a draw or reaches the turn limit

    def evaluate_agents(self, num_games=10):
        dqn_wins = 0
        minimax_wins = 0
        draws = 0
        for _ in range(num_games):
            winner = self.simulate_game()
            if winner == 1:
                dqn_wins += 1
            elif winner == -1:
                minimax_wins += 1
            else:
                draws += 1

        print(f"DQN Agent Wins: {dqn_wins}")
        print(f"Minimax Agent Wins: {minimax_wins}")
        print(f"Draws: {draws}")