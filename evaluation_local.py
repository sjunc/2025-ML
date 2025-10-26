import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
from env.chooseenv import make # Assuming this is the correct environment import
from tabulate import tabulate
import argparse
from torch.distributions import Categorical

# === 0. Setup and Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = "my_rl_agent_final.pt"

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
        self.net = ActorCritic(obs_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def choose_action(self, obs_flat):
        x = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
        logits, value = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        with torch.no_grad():
            action = dist.sample()
        
        logp = dist.log_prob(action)
        return action.item(), value.squeeze(0), logp

    def compute_returns(self, rewards, masks):
        R = 0
        returns = []
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(device)

    def update(self, logps, values, returns):
        returns = returns.detach() 
        values = torch.cat(values).squeeze(-1)
        logps = torch.stack(logps)
        
        td_error = returns - values
        normalized_advantage = (td_error - td_error.mean()) / (td_error.std() + 1e-8) 
        
        actor_loss = -(logps * normalized_advantage.detach()).mean()
        critic_loss = td_error.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
        }, path)
        print(f"Agent state saved to {path}")

    def load(self, path):
        if os.path.exists(path):
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
            saved.append((action_idx, logp, value, action))
    return joint_actions, saved

# === One training episode with Final Reward Shaping ===
def run_episode(env, agent_obj, algo_list):
    state = env.reset(shuffle_map=False)
    done = False
    logps, values, rewards, masks = [], [], [], [] 
    step = 0
    total_shaped_reward = 0.0

    while not done and step < env.max_step:
        agent_idx = 1 # Our agent
        prev_pos = env.env_core.agent_pos[agent_idx]
        prev_v = env.env_core.agent_v[agent_idx]

        joint_action, saved = get_join_actions_train(state, agent_obj, algo_list)
        next_state, reward, done, _, _ = env.step(joint_action)
        
        # --- Final Reward Shaping Logic ---
        shaping_reward = 0
        current_pos = env.env_core.agent_pos[agent_idx]
        obs = state[agent_idx]['obs']
        
        # 1. Progress Reward / Backward Penalty
        progress = current_pos[0] - prev_pos[0]
        if progress > 0:
            shaping_reward += progress * 1.0  # Strengthened forward reward
        else:
            shaping_reward += progress * 2.0  # Stronger penalty for backward/still

        # 2. Pathfinding Rewards (Goal/Arrow Seeking)
        center_view = obs[:, 10:15]
        if np.any(obs == 7): # Goal is visible
            shaping_reward += 5 # Lowered to balance with other rewards
            if np.any(center_view == 7):
                shaping_reward += 10 # Bonus for centering the goal
        elif np.any(center_view == 4): # Goal not visible, center on arrows
            shaping_reward += 2

        # 3. Collision Penalty
        prev_speed = np.linalg.norm(prev_v)
        current_speed = np.linalg.norm(env.env_core.agent_v[agent_idx])
        if prev_speed > 5 and current_speed < prev_speed * 0.8:
            shaping_reward -= 2 # Reduced penalty

        # 4. Turning Penalty
        if saved[agent_idx]:
            _, _, _, action_taken = saved[agent_idx]
            angle_used = action_taken[1]
            if abs(angle_used) > 15: # Penalize sharp turns
                shaping_reward -= (abs(angle_used) / 30.0) * 0.5

        # 5. Time Penalty
        shaping_reward -= 0.05 # Reduced penalty

        final_reward = reward[agent_idx] + shaping_reward
        total_shaped_reward += final_reward

        if saved[agent_idx] is None: 
            break
        _, logp, value, _ = saved[agent_idx]
        
        logps.append(logp) 
        values.append(value)
        rewards.append(final_reward)
        masks.append(0.0 if done else 1.0)
        state = next_state
        step += 1

    # 6. Timeout Penalty
    if step >= env.max_step and not env.env_core.agent_list[1].finished:
        if rewards:
            rewards[-1] -= 25 # Reduced but still significant penalty

    if rewards:
        returns = agent_obj.compute_returns(rewards, masks)
        loss = agent_obj.update(logps, values, returns)
    else:
        loss = 0.0
        
    return total_shaped_reward, step, loss

# === Training loop with Experience Replay ===
def train_with_replay(agent_obj, env, algo_list, curriculum, replay_chance=0.3):
    mastered_maps = []
    for i, stage in enumerate(curriculum):
        map_id = stage['map_id']
        episodes = stage['episodes']
        lr = stage['lr']
        
        print(f"\n--- Curriculum Stage {i+1}/{len(curriculum)}: Focusing on Map {map_id} for {episodes} episodes (LR: {lr}) ---")
        
        for param_group in agent_obj.optimizer.param_groups:
            param_group['lr'] = lr

        for ep in range(1, episodes + 1):
            if random.random() < replay_chance and mastered_maps:
                replay_map_id = random.choice(mastered_maps)
                env.specify_a_map(replay_map_id)
                is_replay = True
            else:
                env.specify_a_map(map_id)
                is_replay = False

            r, s, l = run_episode(env, agent_obj, algo_list)
            
            if ep % 50 == 0:
                replay_str = "(Replay)" if is_replay else ""
                print(f"[Stage {i+1}, Ep {ep}/{episodes}] Map: {env.env_core.map_num} {replay_str}, Shaped Reward: {r:.2f}, Steps: {s}, Loss: {l:.4f}")
            
            if ep % 200 == 0:
                agent_obj.save(MODEL_PATH)
        
        if map_id not in mastered_maps:
            mastered_maps.append(map_id)
        
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
                        print(f"Win in {step} steps on map {env.env_core.map_num}")
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
    
    agent_obj.net.train()

# === Main function with Curriculum Learning + Experience Replay ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_eval", default=100, type=int)
    parser.add_argument("--load", action='store_true', help="Load a model to continue training or for evaluation.")
    parser.add_argument("--eval_only", action='store_true', help="Skip training and only run evaluation (requires --load).")
    parser.add_argument("--render", action='store_true', help="Enable Pygame rendering for evaluation.")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), 'env', 'config.json')) as f:
        env_conf = json.load(f)['olympics-running']
    TOTAL_MAPS = env_conf['map_num']
    print(f"Environment configured with {TOTAL_MAPS} maps.")

    env = make("olympics-running", conf=env_conf, seed=1)
    
    algo_list = ['random', 'rl']
    dummy_state = env.reset(False)
    obs_dim = dummy_state[1]['obs'].flatten().shape[0]

    agent_obj = MyAgent(obs_dim, lr=1e-4, gamma=0.99)

    if args.load:
        agent_obj.load(MODEL_PATH)

    if not args.eval_only:
        print("Starting training with Curriculum Learning and Experience Replay...")
        
        base_curriculum = [
            {'map_id': 1, 'episodes': 500, 'lr': 1e-4},
            {'map_id': 9, 'episodes': 500, 'lr': 1e-4},
            {'map_id': 8, 'episodes': 500, 'lr': 1e-4},
            {'map_id': 2, 'episodes': 700, 'lr': 5e-5},
            {'map_id': 4, 'episodes': 700, 'lr': 5e-5},
            {'map_id': 3, 'episodes': 800, 'lr': 5e-5},
            {'map_id': 5, 'episodes': 800, 'lr': 2e-5},
            {'map_id': 7, 'episodes': 1000, 'lr': 2e-5},
            {'map_id': 6, 'episodes': 1200, 'lr': 1e-5},
            {'map_id': 10, 'episodes': 1200, 'lr': 1e-5},
            {'map_id': 11, 'episodes': 1500, 'lr': 1e-5}
        ]

        active_curriculum = [stage for stage in base_curriculum if stage['map_id'] <= TOTAL_MAPS]

        train_with_replay(agent_obj, env, algo_list, active_curriculum)

        print("\nCurriculum training completed!")
        agent_obj.save(MODEL_PATH)

    print("\n--- Starting Final Evaluation on ALL maps ---")
    run_game(env, agent_obj, algo_list, args.episodes_eval, shuffle_map=True, map_num='all', render_mode=args.render)