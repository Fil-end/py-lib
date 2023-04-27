from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import time
import os


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes, d, replay_buffer, log_path, load_pkl = None, map_location = None): 
    return_list = []
    if load_pkl:
        for i in torch.load(log_path, map_location)['return_list']:
            return_list.append(i)
        replay_buffer = torch.load(log_path, map_location)['replaybuffer']
    rl_stop = False
    c = 0
    for i in range(d):
        with tqdm(total=int(num_episodes / d), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / d)):
                if not os.path.exists('./save_dir/save_model'):
                    os.makedirs('./save_dir/save_model')
                # episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'info': []}
                state = env.reset()
                done = False
                if load_pkl:
                    agent = torch.load(log_path,map_location)['model']
                    
                while not done:
                    action = agent.take_action(state)
                    start_time = time.time()
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    end_time = time.time()
                    cost = end_time - start_time
                    # print(reward , cost, action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['info'].append(info[0])
                    state = next_state
                    episode_return = env.episode_reward
                    # episode_return += reward
                if transition_dict['info'][-1]:
                    c += 1
                else:
                    c = 0
                return_list.append(episode_return)
                agent.update(transition_dict, len(return_list))
                checkpoint = {
                    'model' : agent,
                    'state' : transition_dict,
                    'return_list' : return_list,
                    'replaybuffer': replay_buffer,
                }
                if (i_episode + 1) % 20 == 0:
                    
                    torch.save(checkpoint, log_path)
                # if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/d * i + i_episode+1),
                                        'return': '%.3f' % np.mean(return_list[-1:]),
                                        'count': '%d' % (c)})
                pbar.update(1)
                '''if c >= 2:
                    rl_stop = True
                if rl_stop:
                    break'''
        '''if rl_stop:
            break'''
    return return_list


def train_off_policy_agent(env, agent, num_episodes, d, replay_buffer, minimal_size, batch_size, log_path, load_pkl = None, map_location = None):
    return_list = []
    if load_pkl:
        for i in torch.load(log_path)['return_list']:
            return_list.append(i)
        replay_buffer = torch.load(log_path, map_location)['replay_buffer']
    for i in range(d):
        with tqdm(total=int(num_episodes / d), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / d)):
                if not os.path.exists('./save_dir/save_model'):
                    os.makedirs('./save_dir/save_model')
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                action = 0
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if load_pkl:
                    agent = torch.load(log_path)['model']
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict,len(return_list))
                
                return_list.append(episode_return)
                checkpoint = {
                    'model' : agent,
                    'state' : transition_dict,
                    'return_list' : return_list,
                    'replay_buffer': replay_buffer
                }
                if (i_episode + 1) % 20 == 0:
                    torch.save(checkpoint, log_path)
                pbar.set_postfix({'episode': '%d' % (num_episodes / d * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-1:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


