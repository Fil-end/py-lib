import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from ActorNetwork import PaiNNPolicyNet, PolicyNet
from CriticNetwork import PaiNNValueNet, ValueNet
from utils import rl_utils
from math import sin
from Pd_cluster import MCTEnv
import matplotlib.pyplot as plt
import numpy as np

class PPO:
    ''' PPO2 algorithm'''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, num_episodes, device, node_size, use_GNN_description):
        if use_GNN_description:
            self.actor = PaiNNPolicyNet(state_dim, hidden_dim, action_dim, node_size).to(device)
            self.critic = PaiNNValueNet(state_dim, hidden_dim, node_size).to(device)
        else:
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
            self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  
        self.eps = eps  
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_train_steps = num_episodes
        self.use_GNN_description = use_GNN_description

    def take_action(self, state):
        if self.use_GNN_description:
            node_state_scalar, node_state_vector = torch.tensor(state[0],
                                                                dtype=torch.float).to(self.device), torch.tensor(state[1]
                                                                                                                 ,dtype=torch.float).to(self.device)

            probs = self.actor(node_state_scalar, node_state_vector)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, total_steps):
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        if self.use_GNN_description:
            node_state_scalar, node_state_vector = torch.tensor(np.array(transition_dict['states_s']),
                                                                dtype=torch.float).to(self.device), torch.tensor(np.array(transition_dict['states_v']),
                                                                                                                 dtype=torch.float).to(self.device)
            node_next_state_scalar, node_next_state_vector = torch.tensor(np.array(transition_dict['next_states_s']),
                                                                         dtype=torch.float).to(self.device), torch.tensor(np.array(transition_dict['next_states_v']),
                                                                                                                          dtype=torch.float).to(self.device)

            td_target = rewards + self.gamma * self.critic(node_next_state_scalar, 
                                                           node_next_state_vector) * (1 - dones)
            td_delta = td_target - self.critic(node_state_scalar, node_state_vector)
            print('actor(node_state_scalar, node_state_vector).shape',self.actor(node_state_scalar, node_state_vector).shape)
            print('actions.shape = ', actions.shape)
            old_log_probs = torch.log(self.actor(node_state_scalar, 
                                                 node_state_vector).gather(1, actions)).detach()
        else:
            states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'],
                              dtype=torch.float).to(self.device)
            
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
            td_delta = td_target - self.critic(states)
            old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        

        for _ in range(self.epochs):
            if self.use_GNN_description:
                log_probs = torch.log(self.actor(node_state_scalar, node_state_vector).gather(1, actions))
            else:
                log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # eps
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO loss function
            if self.use_GNN_description:
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(node_state_scalar, node_state_vector), td_target.detach()))
            else:
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):

        lr_a_now = sin(self.actor_lr * pow(0.95, total_steps/10))
        lr_c_now = sin(self.critic_lr * pow(0.95, total_steps/10))
        
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

# model save path
log_dir = './save_dir/save_model/'

# Equivalent model load path
E_model_path = 'my_mace.model'

# Painn paras
cutoff = 4.0
num_interactions = 3
use_GNN_description = True
node_size = 50

# set default torch type 
torch.set_default_dtype(torch.float32)

# reinforcement learning paras
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
buffer_size = 100000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = MCTEnv(save_dir = 'save_dir', save_every= 1 ,timesteps = 1000, reaction_H = 0.790, reaction_n = 10, 
             delta_s = -0.371, use_DESW = False,use_kinetic_penalty = False, use_GNN_description = use_GNN_description,
             calculator_method = 'MACE') 
# env = gym.make('CartPole-v1')
torch.manual_seed(0)
state_dim = env.observation_space['structures'].shape[0]
# state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma,num_episodes, device, node_size, use_GNN_description)

if use_GNN_description:
    return_list = rl_utils.GNN_train_on_policy_agent(env, agent, num_episodes, 10, 
                                             replay_buffer, log_dir, load_pkl = False)
else:
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes, 20, 
                                             replay_buffer, log_dir, load_pkl = False)
# return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, 20, replay_buffer, minimal_size,batch_size, log_dir, load_pkl = True)