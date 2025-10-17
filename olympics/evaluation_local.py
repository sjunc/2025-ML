import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from env.chooseenv import make # Assuming this is the correct environment import
from tabulate import tabulate
import argparse
from torch.distributions import Categorical

# === 0. Setup and Device Configuration ===
# GPU 사용 가능 시 cuda를 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path for saving/loading the model
MODEL_PATH = "my_rl_agent.pt" 

# === Action map: refined (safe + aggressive) ===
actions_map = {
    0: [100, 0], 1: [100, -10], 2: [100, 10], 3: [60, 0], 4: [60, -15], 5: [60, 15],
    6: [30, 0], 7: [30, -20], 8: [30, 20], 9: [150, 0], 10: [150, -6], 11: [150, 6],
    12: [0, 0], 13: [-30, 0], 14: [-30, -10], 15: [-30, 10], 16: [80, -5], 17: [80, 5],
    18: [120, -8], 19: [120, 8], 20: [200, 0], 21: [200, -5], 22: [200, 5], 23: [100, -20],
    24: [100, 20], 25: [60, -25], 26: [60, 25], 27: [100, -30], 28: [100, 30],
    29: [80, -15], 30: [80, 15], 31: [40, -10], 32: [40, 10], 33: [60, -10],
    34: [60, 10], 35: [100, 0],
}
N_ACTIONS = len(actions_map)

# === Neural networks for policy and value ===
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(obs_dim, hidden_size)
        self.policy_head = nn.Linear(hidden_size, N_ACTIONS)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.policy_head(h), self.value_head(h)

# === RL agent wrapper (with Persistence and GPU support) ===
class MyAgent:
    def __init__(self, obs_dim, lr=1e-4, gamma=0.99):
        # Move model to the selected device (GPU/CPU)
        self.net = ActorCritic(obs_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def choose_action(self, obs_flat):
        # Convert to tensor and move to GPU/CPU
        x = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
        
        # Gradient tracking is now enabled for loss calculation
        # REMOVE: with torch.no_grad():
        logits, value = self.net(x)
        
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Action must be sampled without gradient tracking, but logp must track
        with torch.no_grad():
            action = dist.sample()
        
        logp = dist.log_prob(action)
        
        # Return necessary items, ensuring value and logp require grad
        return action.item(), value.squeeze(0), logp

    def compute_returns(self, rewards, masks):
        R = 0
        returns = []
        # Calculate returns (CPU-bound loop, minimal overhead)
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        
        # Create final tensor and move to GPU/CPU for fast update
        return torch.tensor(returns, dtype=torch.float32).to(device)

    # File: evaluation_local.py
    # Inside class MyAgent:
    def update(self, logps, values, returns):
        # ... (이전과 동일한 수정된 내용)
        returns = returns.detach() 
        values = torch.cat(values).squeeze(-1)
        logps = torch.stack(logps)
        
        td_error = returns - values
        normalized_advantage = (td_error - td_error.mean()) / (td_error.std() + 1e-8) 
        
        # Actor Loss (logps requires grad, normalized_advantage is detached multiplier)
        actor_loss = -(logps * normalized_advantage.detach()).mean()
        
        # Critic Loss (td_error requires grad via 'values')
        critic_loss = td_error.pow(2).mean() 
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward() # Now this should work!
        self.optimizer.step()
        
        return loss.item()
    # --- Persistence Methods ---
    def save(self, path):
        """Saves the model's state and optimizer state."""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
        }, path)
        print(f"Agent state saved to {path}")

    def load(self, path):
        """Loads the model's state and optimizer state."""
        if os.path.exists(path):
            # Load map_location ensures it loads correctly even if switching between GPU/CPU
            checkpoint = torch.load(path, map_location=device)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.gamma = checkpoint['gamma']
            print(f"Agent state loaded successfully from {path}")
            return True
        else:
            print(f"No saved agent state found at {path}. Starting fresh.")
            return False

# === Action selection helper ===
def get_join_actions_train(state, agent_obj, algo_list):
    joint_actions = []
    saved = []
    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            force = random.uniform(-100, 200)
            angle = random.uniform(-30, 30)
            joint_actions.append([[force], [angle]])
            saved.append(None)
        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs'].flatten()
            action_idx, value, logp = agent_obj.choose_action(obs)
            action = actions_map[action_idx]
            joint_actions.append([[action[0]], [action[1]]])
            saved.append((action_idx, logp, value))
    return joint_actions, saved

# === One training episode ===
def run_episode(env, agent_obj, algo_list, shuffle_map):
    state = env.reset(shuffle_map)
    done = False
    logps, values, rewards, masks = [], [], [], [] 
    step = 0
    total_reward = 0.0

    while not done and step < env.max_step + 10:
        joint_action, saved = get_join_actions_train(state, agent_obj, algo_list)
        next_state, reward, done, _, _ = env.step(joint_action)
        reward = np.array(reward)
        r = reward[1]
        total_reward += r

        if saved[1] is None:
            break
        _, logp, value = saved[1]
        
        logps.append(logp) 
        values.append(value)
        
        rewards.append(r)
        masks.append(0.0 if done else 1.0)
        state = next_state
        step += 1

    if rewards:
        returns = agent_obj.compute_returns(rewards, masks)
        loss = agent_obj.update(logps, values, returns)
    else:
        loss = 0.0
        
    return total_reward, step, loss

# === Training loop ===
def train(agent_obj, env, algo_list, n_episodes=1000, shuffle_map=True, save_interval=100):
    for ep in range(1, n_episodes + 1):
        r, s, l = run_episode(env, agent_obj, algo_list, shuffle_map)
        
        if ep % 50 == 0:
            print(f"[Episode {ep}] Reward: {r:.2f}, Steps: {s}, Loss: {l:.4f}")
            
        if ep % save_interval == 0:
            agent_obj.save(MODEL_PATH)
            
    agent_obj.save(MODEL_PATH)


# === Evaluation (Headless-compatible) ===
def run_game(env, agent_obj, algo_list, episode, shuffle_map, map_num, render_mode=False): 
    total_reward = np.zeros(2)
    num_win = np.zeros(3)
    total_steps = []
    
    agent_obj.net.eval() 

    for i in range(1, episode + 1):
        episode_reward = np.zeros(2)
        state = env.reset(shuffle_map)
        
        if render_mode: 
            env.env_core.render()
            
        step = 0

        while True:
            joint_actions, _ = get_join_actions_train(state, agent_obj, algo_list)
            next_state, reward, done, _, _ = env.step(joint_actions)
            reward = np.array(reward)
            episode_reward += reward
            
            if render_mode: 
                env.env_core.render()
                
            step += 1

            if done:
                if reward[0] != reward[1]:
                    winner = 0 if reward[0] == 100 else 1
                    num_win[winner] += 1
                    if winner == 1:
                        total_steps.append(step)
                        print(f"Win in {step} steps")
                else:
                    num_win[2] += 1 # Draw
                break
            state = next_state

        total_reward += episode_reward

    total_reward /= episode
    avg_steps = np.mean(total_steps) if total_steps else float('nan')
    print(f"\nResults on map {map_num} with {episode} episodes:")
    data = [
        ['Score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
        ['Wins', int(num_win[0]), int(num_win[1])],
        ['Avg Steps', avg_steps, '-']
    ]
    print(tabulate(data, headers=["Metric", algo_list[0], algo_list[1]], tablefmt='pretty'))
    
    agent_obj.net.train() # Restore train mode

# === Main function ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", choices=["rl", "random"], default="rl")
    parser.add_argument("--opponent", choices=["rl", "random"], default="random")
    parser.add_argument("--episodes_train", default=1000, type=int)
    parser.add_argument("--episodes_eval", default=100, type=int)
    parser.add_argument("--map", default='all', help='1/2/3/4/all')
    parser.add_argument("--load", action='store_true', help="Load the saved agent state if it exists.")
    parser.add_argument("--render", action='store_true', help="Enable Pygame rendering for evaluation.")
    args = parser.parse_args()

    env = make("olympics-running", conf=None, seed=1)
    
    if args.map != 'all':
        env.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    algo_list = ['random', 'rl']
    dummy_state = env.reset(shuffle)
    obs_dim = dummy_state[1]['obs'].flatten().shape[0]

    agent_obj = MyAgent(obs_dim, lr=1e-4, gamma=0.99)
    
    if args.load:
        agent_obj.load(MODEL_PATH)

    print("Starting training …")
    train(agent_obj, env, algo_list, n_episodes=args.episodes_train, shuffle_map=shuffle)

    print("Starting evaluation …")
    run_game(env, agent_obj, algo_list, args.episodes_eval, shuffle_map=shuffle, map_num=args.map, render_mode=args.render)