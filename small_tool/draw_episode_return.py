import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils.rl_utils as rl_utils
# from Pd100_Env import MCTEnv
import os
import torch.nn as nn
from matplotlib.font_manager import FontProperties
import matplotlib

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

        print("------use_orthogonal_init------")
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2, gain=0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, num_episodes, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_train_steps = num_episodes

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, total_steps):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
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
        lr_a_now = self.actor_lr * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.critic_lr * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

log_dir = './save_dir/save_model/model.pkl'
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 10000
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
buffer_size = 100000
minimal_size = 1000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# env = MCTEnv(save_dir = 'save_dir', save_every= 20 ,timesteps = 600) 
# env = gym.make('CartPole-v1')
# torch.manual_seed(0)
# state_dim = env.observation_space['structures'].shape[0]
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma, num_episodes, device)

# return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes, log_dir, load_pkl = True)
# episodes_list = list(range(len(return_list)))

# save_path_1 = os.path.join('./1.png')



'''dict = torch.load('./model.pkl', map_location = torch.device('cpu'))['state']
return_list = torch.load('./model.pkl', map_location = torch.device('cpu'))['return_list']
episodes_list = list(range(len(return_list)))
save_path_1 = os.path.join('./1.png')

plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format('MCT'))
plt.savefig(save_path_1, bbox_inches='tight')

mv_return = rl_utils.moving_average(return_list, 9)
save_path_2 = os.path.join('./2.png')
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format('MCT'))
plt.savefig(save_path_2, bbox_inches='tight')'''

dict = torch.load('./model.pkl', map_location = torch.device('cpu'))['state']
return_list = torch.load('./model.pkl', map_location = torch.device('cpu'))['return_list']

'''return_list_n = []
for i in range(len(return_list)):
    if i <= 620 or (i >= 760 and i <= 1340) or i >=1360:
        return_list_n.append(return_list[i])'''

episodes_list = list(range(len(return_list)))

'''font = {'family' : 'Arial',
'weight' : 'medium',
'size' : 20,
'style' : 'normal'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial'
matplotlib.rc('font', **font)'''

'''save_path_1 = os.path.join('./1.png')

plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format('MCT'))
plt.savefig(save_path_1, bbox_inches='tight')

mv_return = rl_utils.moving_average(return_list, 9)
save_path_2 = os.path.join('./2.png')
plt.plot(episodes_list, mv_return)
plt.xticks(fontsize=11, fontfamily='Arial')
plt.yticks(fontsize=11, fontfamily='Arial')
plt.xlabel('Episodes', fontsize=16, fontweight='bold', fontstyle='italic', fontfamily='Arial')
plt.ylabel('Returns', fontsize=16, fontweight='bold', fontstyle='italic', fontfamily='Arial')
plt.title('PPO on {}'.format('MCT'), fontsize=22, fontweight='bold', fontfamily='Arial')
plt.savefig(save_path_2, bbox_inches='tight')
'''
mv_return = rl_utils.moving_average(return_list, 9)
upper,below = rl_utils.get_RMSE_err_list(return_list,mv_return, 9)

save_path_2 = os.path.join('3.png')
plt.plot(episodes_list, mv_return, color = 'darkorange')
plt.fill_between(episodes_list, below, upper, alpha = 0.5, color = 'darkorange')
plt.xticks(fontsize=11, fontfamily='Arial')
plt.yticks(fontsize=11, fontfamily='Arial')
plt.xlabel('Episodes', fontsize=16, fontweight='bold', fontstyle='italic', fontfamily='Arial')
plt.ylabel('Returns', fontsize=16, fontweight='bold', fontstyle='italic', fontfamily='Arial')
plt.title('PPO on {}'.format('MCT'), fontsize=22, fontweight='bold', fontfamily='Arial')
plt.savefig(save_path_2, bbox_inches='tight')