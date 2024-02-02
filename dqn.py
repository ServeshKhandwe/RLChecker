import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from CheckersEnv import CheckersEnv


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)


import os


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).flatten().unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).flatten().unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            state = torch.FloatTensor(state).flatten().unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, num_episodes, batch_size):
        for e in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                valid_actions = env.get_legal_moves(env.current_player)
                if not valid_actions:
                    break  # No valid moves, end the game

                action = self.act(state)
                # Ensure action is within the valid range
                action = action % len(valid_actions)

                # Execute the chosen action
                next_state, reward, done = env.step(action)  # action is now an index
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time}, e: {self.epsilon:.2}")
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            if e % 10 == 0:
                self.save("./checkers_dqn.pth")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    ##############################

    ##############################


def main():
    board_size = 8
    state_size = board_size * board_size
    action_size = board_size * board_size * 2  # Considering a move can be to any of two directions for each piece
    learning_rate = 0.001
    num_episodes = 10
    batch_size = 64

    env = CheckersEnv()

    agent = DQNAgent(state_size, action_size, learning_rate)

    print("Starting training...")
    agent.train(env, num_episodes, batch_size)
    print("Training completed.")

    # Save the final model
    agent.save("./final_checkers_dqn.pth")
    print("Saved the trained model.")

    # def simulate_game(agent1, agent2, env, verbose=False):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         # Agent 1 makes a move
    #         action1 = agent1.act(state)
    #         next_state, reward, done = env.step(action1)
    #         if done:
    #             if verbose:
    #                 print("Agent 1 wins!")
    #             return 1  # Agent 1 wins
    #
    #         # Agent 2 makes a move
    #         action2 = agent2.act(next_state)
    #         state, reward, done = env.step(action2)
    #         if done:
    #             if verbose:
    #                 print("Agent 2 wins!")
    #             return 2  # Agent 2 wins
    #
    #     return 0  # Draw
    #
    # def evaluate_agents(agent1, agent2, env, num_games=100):
    #     agent1_wins = 0
    #     agent2_wins = 0
    #     for _ in range(num_games):
    #         result = simulate_game(agent1, agent2, env)
    #         if result == 1:
    #             agent1_wins += 1
    #         elif result == 2:
    #             agent2_wins += 1
    #     print(f"Agent 1 wins: {agent1_wins} games")
    #     print(f"Agent 2 wins: {agent2_wins} games")
    #
    # trained_agent = DQNAgent(state_size, action_size, learning_rate)
    # trained_agent.model.load_state_dict(torch.load('final_checkers_dqn.pth'))
    # trained_agent.epsilon = 0  # No exploration
    # new_agent = DQNAgent(state_size, action_size, learning_rate)
    # evaluate_agents(trained_agent, new_agent, env, num_games=100)


if __name__ == "__main__":
    main()
