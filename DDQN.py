import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse

# Neural Network Model (Q-network)
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DDQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, epsilon=0.1, gamma=0.99, lr=0.001, batch_size=64, replay_buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)  # Experience replay buffer
        self.model = QNetwork(state_size, action_size).float()
        self.target_model = QNetwork(state_size, action_size).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _to_numpy(self, x):
        # Ensure state is a contiguous float32 numpy array
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return np.array(x, dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))  # Random action (exploration)
        else:
            state_arr = self._to_numpy(state)
            state_tensor = torch.from_numpy(state_arr).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values, dim=1).item()  # Action with max Q-value (exploitation)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to learn

        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of arrays to single contiguous arrays for faster tensor creation
        states = torch.from_numpy(np.array(states, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        # Use float for arithmetic in target calculation
        dones = torch.FloatTensor(dones)

        # Get Q-values from the model and the target model
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # DDQN Target calculation: using the online model to select the next action, and the target model to estimate its Q-value
        next_actions = torch.argmax(self.model(next_states), dim=1)
        target_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        targets = rewards + (1 - dones) * self.gamma * target_q_values

        # Calculate the loss
        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)).squeeze(1), targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self):
        self.update_target_model()

# Main training loop
def train_ddqn(num_episodes: int = 1000, max_t: int = 200):
    env = gym.make('CartPole-v1')
    agent = DDQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    for episode in range(num_episodes):
        # Gymnasium reset returns (obs, info)
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_t):
            action = agent.act(state)
            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update()  # Periodic update of the target model

        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}")

    env.close()
    return agent

def evaluate_ddqn(agent, episodes=10, render=False):
    # Evaluate the greedy policy (epsilon=0) over several episodes
    render_mode = 'human' if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"Eval Episode {ep+1}/{episodes}, Reward: {ep_reward}")
    agent.epsilon = old_epsilon
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"Average Eval Reward over {episodes} episodes: {avg_reward}")
    env.close()

if __name__ == "__main__":
    def load_agent(model_path: str):
        env = gym.make('CartPole-v1')
        agent = DDQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
        state_dict = torch.load(model_path, map_location='cpu')
        agent.model.load_state_dict(state_dict)
        agent.update_target_model()
        agent.epsilon = 0.0
        env.close()
        return agent

    parser = argparse.ArgumentParser(description="DDQN CartPole trainer/evaluator")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation using a saved model without retraining")
    parser.add_argument("--model-path", type=str, default="ddqn_cartpole.pt", help="Path to saved model state_dict")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--train-episodes", type=int, default=1000, help="Number of training episodes")
    args = parser.parse_args()

    if args.eval_only:
        agent = load_agent(args.model_path)
        evaluate_ddqn(agent, episodes=args.episodes, render=args.render)
    else:
        agent = train_ddqn(num_episodes=args.train_episodes)
        torch.save(agent.model.state_dict(), args.model_path)
        evaluate_ddqn(agent, episodes=args.episodes, render=args.render)
