from xml.sax import default_parser_list
import ray
import pynvml
from configparser import ConfigParser
from argparse import ArgumentParser
import torch.optim as optim
import torch
from PPO import PPO
from dataclasses import dataclass
from Pd_cluster import MCTEnv
from utils import rl_utils
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Queue
import time
from ast import Dict, List
from ActorNetwork import PaiNNPolicyNet, PolicyNet
from CriticNetwork import PaiNNValueNet, ValueNet
import torch.nn.functional as F
from torch.distributions import Normal
import math
from math import sin,cos
import os
from tqdm import tqdm
import numpy as np

def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

class Worker(mp.Process):
    def __init__(self, id, args, agent_args, global_valueNet,global_value_optimizer,
                 global_policyNet,global_policy_optimizer,
                 global_epi,global_epi_rew,rew_queue,):
        super(Worker, self).__init__()
        # define env for individual worker
        self.env = MCTEnv(save_dir = agent_args.save_dir,
                          save_every = agent_args.save_every,
                          timesteps = agent_args.timesteps,
                          reaction_H = agent_args.reaction_H,
                          reaction_n = agent_args.reaction_n,
                          delta_s = agent_args.delta_s,
                          use_DESW = agent_args.use_DESW,
                          use_GNN_description = agent_args.use_GNN_description,)
        
        self.name = str(id)
        # self.env.seed(id)
        self.state_dim = self.env.observation_space['structures'].shape[0]
        self.hidden_dim = agent_args.hidden_dim
        self.action_dim = self.env.action_space.n

        self.gamma = agent_args.gamma
        self.lmbda = agent_args.lmbda
        self.epochs = agent_args.epochs  # 一条序列的数据用来训练轮数
        self.eps = agent_args.eps  # PPO中截断范围的参数
        self.device = device
        self.actor_lr = agent_args.actor_lr
        self.critic_lr = agent_args.critic_lr
        self.max_train_steps = agent_args.num_episodes
        self.use_GNN_description = agent_args.use_GNN_description
        self.node_size = agent_args.node_size

        self.memory=[]

        # passing global settings to worker
        self.global_valueNet, self.global_value_optimizer = global_valueNet, global_value_optimizer
        self.global_policyNet, self.global_policy_optimizer = global_policyNet, global_policy_optimizer
        self.global_epi,self.global_epi_rew = global_epi,global_epi_rew
        self.rew_queue = rew_queue

        # define local net for individual worker
        if self.agent_args.use_GNN_description:
            self.local_policyNet = PaiNNPolicyNet(self.state_dim, 
                                        self.hidden_dim, 
                                        self.action_dim, 
                                        self.node_size).to(device)
            self.local_valueNet = PaiNNValueNet(self.state_dim, 
                                        self.hidden_dim, 
                                        self.node_size).to(device)
        else:
            self.local_policyNet = PolicyNet(self.state_dim, 
                                        self.hidden_dim, 
                                        self.action_dim, 
                                        self.node_size).to(device)
            self.local_valueNet = ValueNet(self.state_dim, 
                                        self.hidden_dim, 
                                        self.node_size).to(device)
            
    def take_action(self, state):
        if self.use_GNN_description:
            node_state_scalar, node_state_vector = torch.tensor(state[0],
                                                                dtype=torch.float).to(self.device), torch.tensor(state[1]
                                                                                                                 ,dtype=torch.float).to(self.device)

            probs = self.global_policyNet(node_state_scalar, node_state_vector)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.global_policyNet(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def sync_global(self):
        self.local_valueNet.load_state_dict(self.global_valueNet.state_dict())
        self.local_policyNet.load_state_dict(self.global_policyNet.state_dict())

    def lr_decay(self, total_steps):

        lr_a_now = sin(self.actor_lr * pow(0.95, total_steps/10))
        lr_c_now = sin(self.critic_lr * pow(0.95, total_steps/10))
        
        for p in self.global_policy_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.global_value_optimizer.param_groups:
            p['lr'] = lr_c_now

    def update_global(self, transition_dict, total_steps):

        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)
        if self.use_GNN_description:
            node_state_scalar, node_state_vector = torch.tensor(np.array(transition_dict['states'][0]),
                                                                dtype=torch.float).to(self.device), torch.tensor(np.array(transition_dict['states'][1]),
                                                                                                                 dtype=torch.float).to(self.device)
            node_next_state_scalar, node_next_state_vector = torch.tensor(np.array(transition_dict['next_states'][0]),
                                                                         dtype=torch.float).to(self.device), torch.tensor(np.array(transition_dict['next_states'][1]),
                                                                                                                          dtype=torch.float).to(self.device)

            td_target = rewards + self.gamma * self.local_valueNet(node_next_state_scalar, 
                                                           node_next_state_vector) * (1 - dones)
            td_delta = td_target - self.local_valueNet(node_state_scalar, node_state_vector)
            # print(f'actor paras:\n {self.actor.state_dict()}')

            old_log_probs = torch.log(self.local_policyNet(node_state_scalar, 
                                                 node_state_vector).gather(1, actions)).detach()
        else:
            states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'],
                              dtype=torch.float).to(self.device)
            
            td_target = rewards + self.gamma * self.local_valueNet(next_states) * (1 -
                                                                       dones)
            td_delta = td_target - self.local_valueNet(states)
            old_log_probs = torch.log(self.local_policyNet(states).gather(1,
                                                            actions)).detach()
        
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        

        for _ in range(self.epochs):
            if self.use_GNN_description:
                log_probs = torch.log(self.local_policyNet(node_state_scalar, node_state_vector).gather(1, actions))
            else:
                log_probs = torch.log(self.local_policyNet(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            policy_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            if self.use_GNN_description:
                value_loss = torch.mean(
                    F.mse_loss(self.local_valueNet(node_state_scalar, node_state_vector), td_target.detach()))
            else:
                value_loss = torch.mean(
                    F.mse_loss(self.local_valueNet(states), td_target.detach()))
            self.global_value_optimizer.zero_grad()
            value_loss.backward()
            # propagate local gradients to global parameters
            for local_params, global_params in zip(self.local_valueNet.parameters(), self.global_valueNet.parameters()):
                global_params._grad = local_params._grad
            self.global_value_optimizer.step()

            self.global_policy_optimizer.zero_grad()
            policy_loss.backward()
            # propagate local gradients to global parameters
            for local_params, global_params in zip(self.local_policyNet.parameters(), self.global_policyNet.parameters()):
                global_params._grad = local_params._grad
            self.global_policy_optimizer.step()

            self.lr_decay(total_steps)

        self.memory=[]


    def run(self, d, log_path, load_pkl = None, map_location = None):
        return_list = []
        if load_pkl:
            for i in torch.load(log_path + 'model.pkl', map_location)['return_list']:
                return_list.append(i)

        while self.global_epi.value < self.max_epi:
            with tqdm(total=int(self.max_epi / d), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.max_epi/ d)):
                    if not os.path.exists('./save_dir/save_model'):
                        os.makedirs('./save_dir/save_model')
                    
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'info': []}
                    # transition_dict = {'states_s': [], 'states_v': [], 'actions': [], 
                     #              'next_states_s': [],'next_states_v': [], 'rewards': [], 'dones': [], 'info': []}
                    state = self.env.reset()
                    done = False

                    if load_pkl:
                        agent = torch.load(log_path + 'model.pkl',map_location)['model']

                    total_reward=0
                    while not done:
                        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                        action = self.take_action(state)  # 离散空间取直接prob，连续空间取log prob
                        next_state, reward, done, _ = self.env.step(action)
                        self.memory.append([state,action,reward,next_state,done])
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        
                        total_reward = self.env.episode_reward
                        state = next_state

                        # recoding global episode and episode reward
                        with self.global_epi.get_lock():
                            self.global_epi.value += 1
                        with self.global_epi_rew.get_lock():
                            if self.global_epi_rew.value == 0.:
                                self.global_epi_rew.value = total_reward
                            else:
                                # Moving average reward
                                self.global_epi_rew.value = self.global_epi_rew.value * 0.99 + total_reward * 0.01
                        self.rew_queue.put(self.global_epi_rew.value)

                        checkpoint = {
                            'model' : agent,
                            'state' : transition_dict,
                            'return_list' : return_list,
                        }
                        if (self.max_epi / d * i + i_episode + 1) % 20 == 0:
                            
                            torch.save(checkpoint, log_path + 'model_{}.pkl'.format(self.max_epi / d * i + i_episode + 1))
                        # if (i_episode + 1) % 10 == 0:
                        # pbar.set_postfix({'episode': '%d' % (self.max_epi/d * i + i_episode+1),
                        #                         'return': '%.3f' % np.mean(return_list[-1:]),
                        #                         'count': '%d' % (c)})
                        
                        print("w{} | episode: {}\t , episode reward:{:.4} \t  "
                                .format(self.name,self.global_epi.value,self.global_epi_rew.value))
                        
                        pbar.update(1)

                    # update and sync with the global net when finishing an episode
                    self.update_global(transition_dict, len(return_list))
                    self.sync_global()

        self.rew_queue.put(None)

@dataclass
class DPPO():
    args: Dict(str)
    agent_args: Dict(str)

    def __post_init__(self):
        self.max_episode = self.agent_args.num_episodes
        self.global_episode = mp.Value('i', 0)  # 进程之间共享的变量
        self.global_epi_rew = mp.Value('d',0)
        self.rew_queue = mp.Queue()
        self.worker_num = self.args.num_workers

        if self.agent_args.use_GNN_description:
            self.global_policyNet = PaiNNPolicyNet(self.agent_args.state_dim, 
                                        self.agent_args.hidden_dim, 
                                        self.agent_args.action_dim, 
                                        self.agent_args.node_size).to(device)
            self.global_valueNet = PaiNNValueNet(self.agent_args.state_dim, 
                                        self.agent_args.hidden_dim, 
                                        self.agent_args.node_size).to(device)
        else:
            self.global_policyNet = PolicyNet(self.agent_args.state_dim, 
                                        self.agent_args.hidden_dim, 
                                        self.agent_args.action_dim, 
                                        self.agent_args.node_size).to(device)
            self.global_valueNet = ValueNet(self.agent_args.state_dim, 
                                        self.agent_args.hidden_dim, 
                                        self.agent_args.node_size).to(device)
        
        self.global_policyNet.share_memory()
        self.global_valueNet.share_memory()
        
        self.global_optimizer_value = torch.optim.Adam(self.global_actor.parameters(),
                                                lr=self.agent_args.actor_lr)
        self.global_optimizer_policy = torch.optim.Adam(self.global_critic.parameters(),
                                                 lr=self.agent_args.critic_lr)
        
        # define the workers
        self.workers=[Worker(i, self.args, self.agent_args,
                             self.global_valueNet,self.global_optimizer_value,
                             self.global_policyNet,self.global_optimizer_policy,
                             self.global_episode,self.global_epi_rew,self.rew_queue)
                      for i in range(self.worker_num)]
        
    def train_worker(self):
        scores=[]
        [worker.start() for worker in self.workers]
        while True:
            r = self.rew_queue.get()
            if r is not None:
                scores.append(r)
            else:
                break
        [worker.join() for worker in self.workers]

        return scores
    
    def save_model(self):
        torch.save(self.global_valueNet.state_dict(), "dppo_value_model.pth")
        torch.save(self.global_policyNet.state_dict(), "dppo_policy_model.pth")



if __name__ == '__main__':

    torch.manual_seed(0)
    # Get num_workers
    pynvml.nvmlInit()   # initialize the pynvml
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()    # get GPU card numbers

    parser = ArgumentParser('parameters')

    parser.add_argument('--RL_Env', type='Pd_cluster', default=str, help='name of RL_Env, (default: str(Pd100_Env))')
    parser.add_argument('--num_workers', type=int, default=gpuDeviceCount, help='number of actors, (default: 3)')
    parser.add_argument("--device", type=device, default = torch.device("cuda"), help = 'torch device(default : True)')
 
    args = parser.parse_args()

    ##Algorithm config parser
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, 'dppo') 

    print(args.num_workers)

    #ray init
    # ray.init()
    agent = DPPO(args, agent_args)
    agent.train_worker()
    # run_DPPO(args, agent_args)

    #ray terminate
    # ray.shutdown()