import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Optional

# Optional plotting (install matplotlib)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from racing_env import RacingEnv


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 gamma=0.99, lr=1e-3, batch_size=128, replay_buffer_size=50000, target_update_every=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.model = QNetwork(state_size, action_size).float()
        self.target_model = QNetwork(state_size, action_size).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_model()
        self.learn_steps = 0
        self.target_update_every = target_update_every
        self.loss_history: List[float] = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return np.array(x, dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        s = torch.from_numpy(self._to_numpy(state)).unsqueeze(0)
        with torch.no_grad():
            q = self.model(s)
        return int(torch.argmax(q, dim=1).item())

    def store(self, s, a, r, ns, done):
        self.replay_buffer.append((s, a, r, ns, done))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q = self.model(states)
        next_q_target = self.target_model(next_states)
        next_actions = torch.argmax(self.model(next_states), dim=1)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + (1 - dones) * self.gamma * next_q

        pred = q.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Track loss for plotting
        try:
            self.loss_history.append(float(loss.item()))
        except Exception:
            pass

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.update_target_model()

        # Epsilon decay (episode-level is typical, but step-level also works)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def train(episodes: int = 500, render: bool = False, resume_model_path: str | None = None, save_model_path: str = 'ddqn_racing.pt', render_fps: int = 60,
          outdir: str = 'outputs', save_plots: bool = True):
    env = RacingEnv(render_mode='human' if render else None, render_fps=render_fps)
    obs, _ = env.reset()
    agent = DDQNAgent(state_size=obs.shape[0], action_size=env.action_space.n)
    # Optionally resume from a saved model
    if resume_model_path:
        try:
            state_dict = torch.load(resume_model_path, map_location='cpu')
            agent.model.load_state_dict(state_dict)
            agent.update_target_model()
            print(f"Resumed model from: {resume_model_path}")
        except Exception as e:
            print(f"Warning: could not resume from {resume_model_path}: {e}")

    returns: List[float] = []
    eps_history: List[float] = []
    for ep in range(episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        truncated = False
        terminated = False
        while not (terminated or truncated):
            a = agent.act(s)
            ns, r, terminated, truncated, _ = env.step(a)
            agent.store(s, a, r, ns, float(terminated or truncated))
            agent.learn()
            s = ns
            ep_ret += r
        returns.append(ep_ret)
        print(f"Episode {ep+1}/{episodes} | Return: {ep_ret:.1f} | epsilon={agent.epsilon:.3f}")
        eps_history.append(agent.epsilon)
    env.close()
    torch.save(agent.model.state_dict(), save_model_path)
    # Plot metrics
    if save_plots:
        try:
            os.makedirs(outdir, exist_ok=True)
            if plt is not None:
                _plot_metrics(returns, agent.loss_history, eps_history, outdir)
            else:
                print("matplotlib not installed; skipping metric plots.")
        except Exception as e:
            print(f"Warning: could not save plots: {e}")
    return agent, returns


def evaluate(agent: DDQNAgent, episodes: int = 10, render: bool = False, render_fps: int = 120):
    env = RacingEnv(render_mode='human' if render else None, render_fps=render_fps)
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    scores = []
    for ep in range(episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.act(s)
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
        scores.append(ep_ret)
        print(f"Eval {ep+1}/{episodes} | Return: {ep_ret:.1f}")
    agent.epsilon = old_eps
    env.close()
    print(f"Average Eval Return: {float(np.mean(scores)):.1f}")
    return scores


def _plot_metrics(returns: List[float], losses: List[float], eps_history: List[float], outdir: str):
    # Returns per episode
    plt.figure(figsize=(8, 4))
    plt.plot(returns, label='Episode Return')
    if len(returns) >= 20:
        import numpy as _np
        win = 20
        sm = _np.convolve(returns, _np.ones(win)/win, mode='valid')
        plt.plot(range(win-1, win-1+len(sm)), sm, label=f'MA({win})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'returns.png'))
    plt.close()

    # Loss curve
    if losses:
        plt.figure(figsize=(8, 4))
        plt.plot(losses, color='tab:red', linewidth=0.8)
        plt.xlabel('Learn Step')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'loss.png'))
        plt.close()

    # Epsilon
    if eps_history:
        plt.figure(figsize=(8, 3))
        plt.plot(eps_history, color='tab:green')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'epsilon.png'))
        plt.close()


def render_q_heatmap(agent: DDQNAgent, outdir: str = 'outputs', grid: int = 160, v_ratio: float = 0.7):
    # Build environment geometry
    env = RacingEnv()
    W, H = env.screen_size
    m = env.margin
    tw = env.track_width
    outer = (m - tw, m - tw, W - (m - tw), H - (m - tw))
    inner = (m + tw, m + tw, W - (m + tw), H - (m + tw))

    xs = np.linspace(0, W-1, grid)
    ys = np.linspace(0, H-1, grid)
    heat = np.full((grid, grid), np.nan, dtype=np.float32)

    def on_road(x: float, y: float) -> bool:
        in_outer = (outer[0] <= x <= outer[2]) and (outer[1] <= y <= outer[3])
        in_inner = (inner[0] <= x <= inner[2]) and (inner[1] <= y <= inner[3])
        return in_outer and not in_inner

    def nearest_segment_index(x: float, y: float) -> int:
        # 0: top (y ~ m), 1: right (x ~ W-m), 2: bottom (y ~ H-m), 3: left (x ~ m)
        d_top = abs(y - m)
        d_right = abs(x - (W - m))
        d_bottom = abs(y - (H - m))
        d_left = abs(x - m)
        dists = [d_top, d_right, d_bottom, d_left]
        return int(np.argmin(dists))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if not on_road(x, y):
                continue
            seg = nearest_segment_index(x, y)
            env.wp_idx = seg
            a, t_hat, _ = env._segment_info(env.wp_idx)
            env.pos = np.array([x, y], dtype=np.float32)
            env.theta = float(math.atan2(t_hat[1], t_hat[0]))
            env.v = env.max_speed * v_ratio
            obs = env._obs()
            s = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                q = agent.model(s)
            heat[iy, ix] = float(torch.max(q).item())

    if plt is None:
        print('matplotlib not installed; skipping Q heatmap.')
        return
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(heat, origin='lower', extent=[0, W, 0, H], cmap='viridis')
    # Draw inner/outer rectangles
    import matplotlib.patches as patches
    ax = plt.gca()
    rect_outer = patches.Rectangle((outer[0], outer[1]), outer[2]-outer[0], outer[3]-outer[1], fill=False, edgecolor='white', linewidth=1.5)
    rect_inner = patches.Rectangle((inner[0], inner[1]), inner[2]-inner[0], inner[3]-inner[1], fill=False, edgecolor='white', linewidth=1.0, linestyle='--')
    ax.add_patch(rect_outer)
    ax.add_patch(rect_inner)
    plt.title('Max Q-value Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'q_heatmap.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Eval DDQN on RacingEnv')
    parser.add_argument('--train-episodes', type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--model-path', type=str, default='ddqn_racing.pt', help='Path to save/load model state_dict')
    parser.add_argument('--resume-model', type=str, default=None, help='Optional path to a saved model to resume training from')
    parser.add_argument('--train-fps', type=int, default=60, help='Rendering FPS during training (use 0 for unlimited)')
    parser.add_argument('--eval-fps', type=int, default=120, help='Rendering FPS during evaluation (use 0 for unlimited)')
    parser.add_argument('--plots-outdir', type=str, default='outputs', help='Directory to save plots/figures')
    parser.add_argument('--save-plots', action='store_true', help='Save training/evaluation plots and Q heatmap')
    args = parser.parse_args()

    if args.eval_only:
        dummy_env = RacingEnv()
        obs, _ = dummy_env.reset()
        agent = DDQNAgent(state_size=obs.shape[0], action_size=dummy_env.action_space.n)
        agent.model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        agent.update_target_model()
        scores = evaluate(agent, episodes=args.eval_episodes, render=args.render_eval, render_fps=args.eval_fps)
        if args.save_plots:
            try:
                render_q_heatmap(agent, outdir=args.plots_outdir)
            except Exception as e:
                print(f"Warning: could not render Q heatmap: {e}")
    else:
        agent, returns = train(episodes=args.train_episodes, render=args.render_train,
                         resume_model_path=args.resume_model, save_model_path=args.model_path,
                         render_fps=args.train_fps, outdir=args.plots_outdir, save_plots=args.save_plots)
        scores = evaluate(agent, episodes=args.eval_episodes, render=args.render_eval, render_fps=args.eval_fps)
        if args.save_plots:
            try:
                render_q_heatmap(agent, outdir=args.plots_outdir)
            except Exception as e:
                print(f"Warning: could not render Q heatmap: {e}")
