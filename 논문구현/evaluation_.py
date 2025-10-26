# evaluation_local.py (modified to support 'planner' agent)
import numpy as np
import torch
import random
from agents.rl.submission import agent as rl_agent  # 기존 rl 에이전트 유지
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os

# Import planner chooser
from agents.planner.submission import choose_action as planner_choose_action

actions_map = {
    0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
    6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30],
    12: [20, -30], 13: [20, -18], 14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30],
    18: [80, -30], 19: [80, -18], 20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30],
    24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
    30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]
}           # discrete action space

def get_join_actions(state, algo_list):

    joint_actions = []

    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs'].flatten()
            actions_raw = rl_agent.choose_action(obs)
            actions = actions_map[actions_raw]
            joint_actions.append([[actions[0]], [actions[1]]])

        elif algo_list[agent_idx] == 'planner':
            # planner returns action_index compatible with actions_map
            obs = state[agent_idx]['obs'].flatten()
            try:
                action_idx = planner_choose_action(obs, actions_map)
            except Exception as e:
                # fallback to random if planner fails
                # print("Planner error:", e)
                driving_force = random.uniform(-100, 200)
                turing_angle = random.uniform(-30, 30)
                joint_actions.append([[driving_force], [turing_angle]])
                continue

            actions = actions_map[action_idx]
            joint_actions.append([[actions[0]], [actions[1]]])

        else:
            # fallback random
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

    return joint_actions



RENDER = True

def run_game(env, algo_list, episode, shuffle_map, map_num, verbose=False):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)       # agent1 win, agent2 win, draw
    total_steps = []            # store steps when a win occurs
    episode = int(episode)
    
    for i in range(1, int(episode)+1):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)
        if RENDER:
            env.env_core.render()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if RENDER:
                env.env_core.render()

            step += 1  # step increment

            if done:
                if reward[0] != reward[1]:
                    if reward[0] == 100:
                        num_win[0] += 1
                        total_steps.append(step)
                        print(step)
                    elif reward[1] == 100:
                        num_win[1] += 1
                        total_steps.append(step)
                        print(step)                        
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break
            state = next_state
        
        total_reward += episode_reward

    # results
    total_reward /= episode
    average_steps = np.mean(total_steps) if total_steps else float('nan')
    print("total reward: ", total_reward)
    print('Result in map {} within {} episode:'.format(map_num, episode))
    
    header = ['Name', algo_list[0], algo_list[1]]
    data = [
        ['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
        ['win', int(num_win[0]), int(num_win[1])],
        ['avg_steps', average_steps, '-']
    ]
    print(tabulate(data, headers=header, tablefmt='pretty'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default='rl', help='rl/random/planner')
    parser.add_argument("--opponent", default='random', help='rl/random/planner')
    parser.add_argument("--episode", default=20)
    parser.add_argument("--map", default='all', help='1/2/3/4/all')
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed = 1)

    if args.map != 'all':
        game.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    # torch.manual_seed(1)
    # np.random.seed(1)
    # random.seed(1)

    agent_list = [args.opponent, args.my_ai]        # your are controlling agent green
    run_game(game, algo_list=agent_list, episode=args.episode, shuffle_map=shuffle, map_num=args.map, verbose=False)
