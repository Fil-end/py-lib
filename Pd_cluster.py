# This code is written by Filend, chen shiyang at 2023/4/20
# The function of the code is to offer a simulation environment
# for the deep reinforcement learning(PPO).
import os
import json
import itertools
import sys
from typing import List, Optional, Tuple
import math
from math import cos, sin
import copy
from copy import deepcopy

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

from scipy.spatial import Delaunay

import gym
from gym import spaces

import torch
import torch.nn as nn

from einops import rearrange

import ase
from ase import Atom, Atoms, units
from ase.visualize.plot import plot_atoms
from ase.io import read
from ase.io.lasp_PdO import write_arc, read_arc
from ase.calculators.lasp_bulk import LASP
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase.geometry.analysis import Analysis
from ase.cluster.wulff import wulff_construction
from mace.calculators import MACECalculator

# use Painn description
import GNN_utils.Painn_utils as Painn
from tools.periodic_table import ELERADII, ELEDICT
from tools.calc import Calculator
from tools.cluster_actions import ClusterActions
from tools.utils import to_pad_the_array


'''
    Actions:
        Type: Discrete(8)
        Num   Action
        0     ADS
        1     Translation
        2     R_Rotation
        3     L_Rotation
        4     Min
        5     Diffusion
        6     Drill
        7     Dissociation

        '''
# slab = images[0]
DIRECTION = [
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, -1, 0]),
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
]
# 设定动作空间
ACTION_SPACES = ['ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']

r_O = ELERADII[7]
r_Si = ELERADII[14]
r_Pd = ELERADII[45]

d_O_Pd = r_O + r_Pd
d_O_O = 2 * r_O
d_Pd_Pd = 2 * r_Pd
d_Si_Pd = r_Si + r_Pd

# 创建MCT环境
class MCTEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self,
                 save_dir=None,
                 timesteps=None,
                 temperature=473,
                 k=8.6173324e-05,
                 max_episodes=None,
                 step_size=0.1,
                 max_energy_profile = 0.5,
                 convergence = 0.005,
                 save_every=None,
                 save_every_min=None,
                 plot_every=None,
                 reaction_H = None,         #/(KJ/mol)
                 reaction_n = None,
                 delta_s = None,            #/eV
                 use_DESW = None,
                 use_GNN_description = None,    # use Painn description
                 use_kinetic_penalty = None,
                 cutoff = 4.0,  # Painn paras
                 hidden_state_size = 50,
                 embedding_size = 50,
                 num_interactions = 3,
                 calculate_method = 'MACE',
                 model_path = None,
                 in_zeolite: bool = False,
                 cluster_metal = "Pd",
                 max_MCT_step = None,
                 max_save_atoms = 2500,
                 max_observation_atoms = 400): 
        # initialize parameters
        # Define system
        self.in_zeolite = in_zeolite
        self.cluster_metal = cluster_metal
        self.calculate_method = calculate_method
        self.model_path = model_path
        
        # Env tricks
        self.convergence = convergence
        self.step_size = step_size
        self.timesteps = timesteps
        self.max_MCT_step = max_MCT_step
        self.max_episodes = max_episodes

        # Saving paramaters
        self.save_every = save_every
        self.plot_every = plot_every
        self.save_every_min = save_every_min
        
        # Switches
        self.use_GNN_description = use_GNN_description
        self.use_kinetic_penalty = use_kinetic_penalty
        self.use_DESW = use_DESW

        # Set temperature = 473 K, and define the thermal properties
        self.temperature_K = temperature
        self.k = k  # eV/K
        self.reaction_H = reaction_H
        self.reaction_n = reaction_n
        self.delta_s = delta_s
        # self.thermal_energy = k * temperature * self.len_atom
        # self.H = 112690 * 32/ 96485

        # Painn paras
        self.cutoff = cutoff
        self.hidden_state_size = hidden_state_size  # embedding_output_dim and the hidden_dim overall the Painn
        self.num_interactions = num_interactions
        self.embedding_size = embedding_size    # embedding_hidden_dim

        # ---------------------------------- Here we start post_init ---------------------------------------------------
        # initialize the initial slab
        self.initial_state = self._generate_initial_slab(self.in_zeolite)

        # initialize Calculator and ClusterActions class
        self.calculator = Calculator(calculate_method=calculate_method, model_path = self.model_path)
        self.cluster_actions = ClusterActions(metal_ele = self.cluster_metal)

        # initialize the thermal properties
        self.E_OO = self.add_mole(self.initial_state, 'OO', 1.21)
        self.E_OOP = self.add_mole(self.initial_state, 'OOO', 1.28)
        print(f"E_OO is {self.E_OO}")

        self.H = self.reaction_H * self.reaction_n

        # pre_processing the Zeolite system
        if self.in_zeolite:
            self.system = self.initial_state.copy()
            self.zeolite = self.system.copy()
            del self.zeolite[[a.index for a in self.zeolite if a.symbol == self.cluster_metal]]
            self.cluster = self.system.copy()
            del self.cluster[[a.index for a in self.cluster if a.symbol != self.cluster_metal]]
            self.system = self.zeolite + self.cluster

            write_arc([self.zeolite], name = "zeolite.arc")
        else:
            self.system = self.cluster_actions.rectify_atoms_positions(self.initial_state)
        

        # get all surfaces
        self.total_surfaces = self._mock_cluster().get_surfaces()

        # 标记可以自由移动的原子
        write_arc([self._get_free_atoms(self.system)], name = "Free_atoms.arc")
        self.free_list = self._get_free_atoms_list(self.system)
        self.fix_list = [atom_idx for atom_idx in range(len(self.system)) if atom_idx not in self.free_list]
        
        self.len_atom = len(self.free_list)
        
        # initialize episode and timestep = 0
        self.episode = 0
        self.timestep = 0

        if max_observation_atoms:
            self.max_observation_atoms = max_observation_atoms
        else:
            self.max_observation_atoms = 2 * len(self.system)

        if max_save_atoms:
            self.max_save_atoms = max_save_atoms
        else:
            self.max_save_atoms = 2 * len(self.system)

        self.max_energy_profile = max_energy_profile
        self.range = [0.9, 1.1]
        self.reward_threshold = 0
        # 保存history,plots,trajs,opt
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history_dir = os.path.join(save_dir, 'history')
        self.plot_dir = os.path.join(save_dir, 'plots')

        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # 初始化history字典
        self.history = {}

        # 记录不同吸附态（Pd64O,Pd64O2......Pd64O16）时的能量以及结构
        self.adsorb_history = {}

        # 定义动作空间
        self.action_space = spaces.Discrete(len(ACTION_SPACES))
        self.observation_space = self.get_observation_space()

        # 一轮过后重新回到初始状态
        self.reset()

        return

    def step(self, action):
        barrier = 0
        reward = 0  # 定义初始奖励为0

        action = action
        
        RMSD_similar = False
        kickout = False

        action_done = True
        target_get = False

        self.done = False  # 开关，决定episode是否结束
        episode_over = False  # 与done作用类似

        self.atoms, previous_structure, previous_energy = self.state
        atoms = self.atoms.copy()

        self.center_point = self.cluster_actions.get_center_point(self._get_cluster(atoms, with_zeolite=self.in_zeolite))

        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        
        write_arc([self.atoms], name = "chk_pt_pre.arc")

        if action in [0, 1, 2, 3, 5, 6, 7]:
            self.facet_selection = self.total_surfaces[np.random.randint(len(self.total_surfaces))]
            self.atoms = self.cluster_actions.cluster_rotation(self.atoms, self.facet_selection, self.center_point)
        
        # Define ACTION_SPACES = 
        # 'ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion'

        # Currently, all actions can coordinate with zeolite env
        '''——————————————————————————————————————————以下是动作选择————————————————————————————————————————————————————————'''
        if action in [1,2,3,8]:
            self.atoms = self._get_cluster(self.atoms, with_zeolite=self.in_zeolite)

        if action == 0:
            self.atoms = self._adsorption(self.atoms)

        elif action == 1:
            self._to_translate(self.atoms)

        elif action == 2:
            self._to_rotate(self.atoms, 3)

        elif action == 3:
            self._to_rotate(self.atoms, -3)

        elif action == 4:
            self._md(self.atoms)
            '''------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------'''

        elif action == 5:  # 表面上氧原子的扩散，单原子行为
            self.atoms, action_done = self._diffusion(self.atoms)

        elif action == 6:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
            self.atoms, action_done = self._drill(self.atoms)

        elif action == 7:  # 氧气解离
            self.atoms, action_done = self._dissociation(self.atoms)

        elif action == 8:
            self.atoms, action_done = self._desorption(self.atoms)
            
        else:
            print('No such action')

        self.timestep += 1
        
        if action in [0, 1, 2, 3, 5, 6, 7]:
            self.cluster_actions.recover_rotation(self.atoms, self.facet_selection, self.center_point)
        
        # print(f"The atoms to optimize is {self.atoms}")
        if action in [1,2,3,8]:
            self.atoms = self._get_system(self.atoms, with_zeolite=self.in_zeolite)

        previous_atom = self.trajectories[-1]
        # 优化该state的末态结构以及next_state的初态结构
        self.to_constraint(self.atoms)
        write_arc([self.atoms], name = "chk_pt_2.arc")
        print(f"The action is {action}, num O2 is {self.n_O2}, num O3 is {self.n_O3}")
        self.atoms, current_energy, current_force = self.calculator.to_calc(self.atoms) 
        current_energy = current_energy + self.n_O2 * self.E_OO + self.n_O3 * self.E_OOP

        if not self.in_zeolite:
            self.atoms = self.cluster_actions.rectify_atoms_positions(self.atoms)
        else:
            self.zeolite = self._get_zeolite(self.atoms)

        if action in [1, 2, 3, 5, 6, 7]:
            barrier = self.check_TS(previous_atom, self.atoms, previous_energy, current_energy, action) 
            if not self.use_kinetic_penalty:
                if barrier > 5:
                    reward += -5.0 / ((self.H+10) * self.k * self.temperature_K)
                    barrier = 5.0
                else:
                    reward +=  -barrier / ((self.H+10) * self.k * self.temperature_K)
            else:
                if barrier > 5:
                    reward += -math.exp(5.0 / (1.5 * 8.314 * self.k * self.temperature_K)) / (self.H+10)
                    barrier = 5.0
                else:
                    reward += -math.exp(barrier / (1.5 * 8.314 * self.k * self.temperature_K)) /(self.H+10)

        # kickout the structure if too similar
        if self.timestep > 11:
            if self.RMSD(self.atoms, self.trajectories[-10])[0] and (current_energy - self.history['energies'][-10]) > 0: 
                self.atoms = previous_atom
                current_energy = previous_energy
                self.n_O2, self.n_O3 = self.history['adsorbates'][-1]
                RMSD_similar = True
                reward -= 1
        
        if RMSD_similar:
            kickout = True

        if not action_done:
            reward += -2

        if self.to_get_bond_info(self.atoms):   # 如果结构过差，将结构kickout
            self.atoms = previous_atom
            current_energy = previous_energy
            self.n_O2, self.n_O3 = self.history['adsorbates'][-1]
            kickout = True
            reward += -5

        if action == 0 and not self.atoms == previous_atom:
            current_energy = current_energy - self.delta_s

        if action == 8 and not self.atoms == previous_atom:
            current_energy = current_energy + self.delta_s

        relative_energy = current_energy - previous_energy
        if relative_energy > 5:
            reward += -1
        else:
            # reward += math.tanh(-relative_energy/(self.H * 8.314 * self.temperature_K)) * (math.pow(10.0, 5))
            reward += self.get_reward_sigmoid(relative_energy)
        
        if relative_energy >= 0:
            reward -= 0.5
            
        self.RMSD_list.append(self.RMSD(self.atoms, previous_atom)[1])

        if self.timestep > 6:
            current_action_list = self.history['actions'][-5:]
            result = all(x == current_action_list[0] for x in current_action_list)
            if result and action == current_action_list[0] and (RMSD_similar and relative_energy >= 0):
                self.repeat_action += 1
                reward -= self.repeat_action * 1
            elif result and action != current_action_list[0]:
                self.repeat_action = 0
      
        current_structure = self.atoms.get_positions()

        self.energy = current_energy
        self.force = current_force

        self.free_list = [idx for idx in range(len(self.atoms)) if idx not in self.fix_list]
        self.pd_save = nn.ZeroPad2d(padding = (0,0,0,self.max_save_atoms-len(self.atoms.get_positions())))
        self.pd_obs = nn.ZeroPad2d(padding = (0,0,0,self.max_observation_atoms-len(self.atoms[self.free_list].get_positions())))

        # if action == 0:
        #     self.adsorb_history['traj'] = self.adsorb_history['traj'] + [self.atoms.copy()]
        #     self.adsorb_history['structure'] = self.adsorb_history['structure'] + [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions())).flatten())]
        #     self.adsorb_history['energy'] = self.adsorb_history['energy'] + [current_energy - previous_energy]
        #     self.adsorb_history['timesteps'].append(self.history['timesteps'][-1] + 1)

        observation = self.get_obs()  # 能观察到该state的结构与能量信息

        self.state = self.atoms, current_structure, current_energy

        # Update the history for the rendering

        self.history, self.trajectories = self.update_history(action, kickout)
        
        exist_too_short_bonds = self.exist_too_short_bonds(self.atoms)

        if exist_too_short_bonds or self.energy - self.initial_energy > len(self.atoms) * self.max_energy_profile or relative_energy > self.max_RE:
            # reward += self.get_reward_sigmoid(1) * (self.timesteps - self.history['timesteps'][-1])
            reward -= 0.5 * self.timesteps
            self.done = True
        
        elif self.timestep > 11:
            if self.atoms == self.trajectories[-10]:
                self.done = True
                reward -= 0.5 * self.timesteps
                
        if -1.5 * relative_energy > self.max_RE:
            self.max_RE = -1.5 * relative_energy
            
        if len(self.history['actions']) - 1 >= self.total_steps:    # 当步数大于时间步，停止，且防止agent一直选取扩散或者平动动作
            self.done = True

        reward -= 0.5 # 每经历一步timesteps, -0.5

        # if len(self.history['real_energies']) > 11:
        #     RMSE_energy = self.RMSE(self.history['real_energies'][-10:])
        #     RMSE_RMSD = self.RMSE(self.RMSD_list[-10:])
        #     if RMSE_energy < 0.5 and RMSE_RMSD < 0.5:
        #         done_similar = True

        # and done_similar
        if ((current_energy - self.initial_energy) <= -self.H and (abs(current_energy - previous_energy) < self.min_RE_d \
                and abs(current_energy - previous_energy) > 0.0001))  and self.RMSD_list[-1] < 0.5 \
                and (current_energy - self.initial_energy) < self.lowest_energy:   
        # if abs(current_energy - previous_energy) < self.min_RE_d and abs(current_energy - previous_energy) > 0.001:    
            self.done = True
            self.lowest_energy = current_energy - self.initial_energy
            self.mct_step += 1

            reward -= (1 + self.reaction_n/100) * (self.energy - self.initial_energy + self.H) /(self.k * self.temperature_K)
            target_get = True
            # self.min_RE_d = abs(current_energy - previous_energy)
        
        self.history['reward'] = self.history['reward'] + [reward]
        self.episode_reward += reward
        
        # 设置惩罚下限
        if self.episode_reward <= self.reward_threshold:   
            self.done = True

        if self.done:
            episode_over = True
            self.episode += 1
            if self.episode % self.save_every == 0 or target_get:
                self.save_episode()
                self.plot_episode()

        return observation, reward, episode_over, [target_get, action_done]

    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d.npz' % self.episode)
        # traj = self.trajectories,
        # adsorb_traj=self.adsorb_history['traj'],
        # adsorb_structure=self.adsorb_history['structure'],
        # adsorb_energy=self.adsorb_history['energy'],
        # adsorb_timesteps = self.adsorb_history['timesteps'],
        # forces = self.history['forces'],
        np.savez_compressed(
            save_path,
            
            initial_energy=self.initial_energy,
            energies=self.history['energies'],
            actions=self.history['actions'],
            structures=self.history['structures'],
            timesteps=self.history['timesteps'],
            symbols = self.history['symbols'],
            reward = self.history['reward'],

            ts_energy = self.TS['energies'],
            ts_timesteps = self.TS['timesteps'],
            barriers = self.TS['barriers'],

            episode_reward = self.episode_reward,

        )
        return

    def plot_episode(self):
        save_path = os.path.join(self.plot_dir, '%d.png' % self.episode)

        energies = np.array(self.history['energies'])
        actions = np.array(self.history['actions'])

        plt.figure(figsize=(30, 30))
        plt.xlabel('steps')
        plt.ylabel('Energies')
        plt.plot(energies, color='blue')

        for action_index in range(len(ACTION_SPACES)):
            action_time = np.where(actions == action_index)[0]
            plt.plot(action_time, energies[action_time], 'o',
                     label=ACTION_SPACES[action_index])

        #plt.scatter(self.TS['timesteps'], self.TS['energies'], label='TS', marker='x', color='g', s=180)
        #plt.scatter(self.adsorb_history['timesteps'], self.adsorb_history['energy'], label='ADS', marker='p', color='black', s=180)
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        return plt.close('all')

    def reset(self):

        print(f'---------------The current reaction_n = {self.reaction_n}, whether use kinetic penalty is {self.use_kinetic_penalty}--------------------')
        self.H = self.reaction_H * self.reaction_n
        
        if os.path.exists('input.arc'):
            os.remove('input.arc')
        if os.path.exists('all.arc'):
            os.remove('all.arc')
        # if os.path.exists('sella.log'):
        #     os.remove('sella.log')

        self.n_O2 = 2000
        self.n_O3 = 0
        self.facet_selection = np.array([0,0,1])
        
        if self.in_zeolite:
            self.atoms = self.zeolite + self.cluster
        else:
            self.atoms = self._generate_initial_slab(self.in_zeolite)

        # center_point = [self.atoms.get_cell()[0][0]/2, self.atoms.get_cell()[1][1]/2, self.atoms.get_cell()[2][2] - 2]
        self.free_list = self._get_free_atoms_list(self.atoms)
        self.fix_list = [atom_idx for atom_idx in range(len(self.atoms)) if atom_idx not in self.free_list]

        # The actual initial state and initial energy   
        self.atoms, self.initial_energy, _ = self.calculator.to_calc(self.atoms)
        self.initial_energy = self.initial_energy + self.n_O2 * self.E_OO + self.n_O3 * self.E_OOP

        if not self.in_zeolite:
            self.atoms = self.cluster_actions.rectify_atoms_positions(self.atoms)
            
        self.pd_save = nn.ZeroPad2d(padding = (0,0,0,self.max_save_atoms-len(self.atoms.get_positions())))
        self.pd_obs = nn.ZeroPad2d(padding = (0,0,0,self.max_observation_atoms-len(self.atoms[self.free_list].get_positions())))

        self.episode_reward = 0.5 * self.timesteps
        self.timestep = 0

        self.total_steps = self.timesteps
        self.max_RE = 3
        self.min_RE_d = self.convergence * self.len_atom
        self.repeat_action = 0

        self.ads_list = []
        for _ in range(self.n_O2):
            self.ads_list.append(2)

        # self.atoms = self.choose_ads_site(self.atoms)
        # self.atoms, self.energy, self.force= self.calculator.to_calc(self.atoms, with_zeolite=self.in_zeolite)
        # self.energy = self.energy + self.n_O2 * self.E_OO + self.n_O3 * self.E_OOP

        # if not self.in_zeolite:
        #     self.atoms = self.cluster_actions.rectify_atoms_positions(self.atoms)
        self.energy = self.initial_energy

        self.trajectories = []
        self.RMSD_list = []
        self.trajectories.append(self.atoms.copy())

        self.mct_step = 0
        self.lowest_energy = 0

        self.TS = {}
        self.TS['energies'] = [0.0]
        self.TS['timesteps'] = [0]
        self.TS['barriers'] = [0.0]

        # self.adsorb_history = {}
        # self.adsorb_history['traj'] = [self.atoms]
        # self.adsorb_history['structure'] = [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions())).flatten())]
        # self.adsorb_history['energy'] = [0.0]
        # self.adsorb_history['timesteps'] = [0]

        results = ['energies', 'actions', 'structures', 'timesteps', 'forces', 'scaled_structures', 'real_energies', 'reward', 'symbols']
        for item in results:
            self.history[item] = []
        self.history['energies'] = [0.0]
        self.history['real_energies'] = [0.0]
        self.history['actions'] = []
        # self.history['forces'] = [np.array(self.pd(torch.tensor(self.force)))]

        # free_atoms = self._get_free_atoms(self.atoms)
        self.history['structures'] = [np.array(self.pd_save(torch.tensor(self.atoms.get_positions())).flatten())]
        self.history['scaled_structures'] = [np.array(self.pd_save(torch.tensor(self.atoms.get_scaled_positions())).flatten())]
        self.history['symbols'] = [to_pad_the_array(np.array(self.atoms.get_chemical_symbols()), max_len = self.max_save_atoms, 
                                                            position = False, symbols = True)]

        self.history['timesteps'] = [0]
        self.history['reward'] = [0]
        self.history['adsorbates'] = [(self.n_O2, self.n_O3)]

        self.state = self.atoms, self.atoms.positions, self.initial_energy

        observation = self.get_obs()

        return observation

    def render(self, mode='rgb_array'):

        if mode == 'rgb_array':
            # return an rgb array representing the picture of the atoms

            # Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(self.atoms.get_scaled_positions(),
                       ax1,
                       rotation='48x,-51y,-144z',
                       show_unit_cell=0)

            ax1.set_ylim([-1, 2])
            ax1.set_xlim([-1, 2])
            ax1.axis('off')
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])

            # Add a subplot for the energy history overlay
            ax2.plot(self.history['timesteps'],
                     self.history['energies'])

            #if len(self.TS['timesteps']) > 0:
            #    ax2.plot(self.TS['timesteps'],
            #             self.TS['energies'], 'o', color='g')

            ax2.set_ylabel('Energy [eV]')

            # Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()

            # return the rendered array (but not the alpha channel)
            return img_array[:, :, :3]

        else:
            return

    def close(self):
        return

    def get_observation_space(self):
        if self.use_GNN_description:
            observation_space = spaces.Dict({'structures':
            spaces.Box(
                low=-1,
                high=2,
                shape=(self.max_observation_atoms, ),
                dtype=float
            ),
            'energy': spaces.Box(
                low=-50.0,
                high=5.0,
                shape=(1,),
                dtype=float
            ),
            'force':spaces.Box(
                low=-2,
                high=2,
                shape=(self.max_observation_atoms, ),
                dtype=float
            ),
            'TS': spaces.Box(low = -0.5,
                                    high = 1.5,
                                    shape = (1,),
                                    dtype=float),
        })
        else:
            observation_space = spaces.Dict({'structures':
                spaces.Box(
                    low=-1,
                    high=2,
                    shape=(self.max_observation_atoms * 3, ),
                    dtype=float
                ),
                'energy': spaces.Box(
                    low=-50.0,
                    high=5.0,
                    shape=(1,),
                    dtype=float
                ),
                'force':spaces.Box(
                    low=-2,
                    high=2,
                    shape=(self.max_observation_atoms * 3, ),
                    dtype=float
                ),
                'TS': spaces.Box(low = -0.5,
                                        high = 1.5,
                                        shape = (1,),
                                        dtype=float),
            })
        return observation_space

    def get_obs(self):
        observation = {}
        obs_atoms = self.atoms[self.free_list]
        if self.use_GNN_description:
            observation['structure_scalar'], observation['structure_vector'] = self._use_Painn_description(obs_atoms)
            observation['energy'] = np.array([self.energy - self.initial_energy]).reshape(1, )
            return observation['structure_scalar'], observation['structure_vector']
        else:
            observation['structure'] = self._use_MLP(obs_atoms)
            observation['energy'] = np.array([self.energy - self.initial_energy]).reshape(1, )
            return observation['structure']

    def update_history(self, action_idx, kickout):
        self.trajectories.append(self.atoms.copy())
        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.energy - self.initial_energy]
        # self.history['forces'] = self.history['forces'] + [np.array(self.pd(torch.tensor(self.force)))]
        self.history['actions'] = self.history['actions'] + [action_idx]

        self.history['structures'] = self.history['structures'] + [np.array(self.pd_save(torch.tensor(self.atoms.get_positions())).flatten())]
        self.history['scaled_structures'] = self.history['scaled_structures'] \
                                            + [np.array(self.pd_save(torch.tensor(self.atoms.get_scaled_positions())).flatten())]
        self.history['symbols'] = self.history['symbols'] + [to_pad_the_array(np.array(self.atoms.get_chemical_symbols()), 
                                                                              max_len = self.max_save_atoms, position = False, symbols = True)]

        self.history['adsorbates'] = self.history['adsorbates']+ [(self.n_O2, self.n_O3)]
        if not kickout:
            self.history['real_energies'] = self.history['real_energies'] + [self.energy - self.initial_energy]

        return self.history, self.trajectories

    def transition_state_search(self, previous_atom, current_atom, previous_energy, current_energy, action):
        layerlist = self.get_layer_atoms(previous_atom)
        layer_O = []
        for i in layerlist:
            if previous_atom[i].symbol == 'O':
                layer_O.append(i)

        if self.use_DESW:
            self.to_constraint(previous_atom)
            write_arc([previous_atom])

            write_arc([previous_atom, current_atom])
            previous_atom = self.calculator.to_calc(previous_atom, calc_type = 'ts')

            if previous_atom.get_potential_energy() == 0:  #没有搜索到过渡态
                ts_energy = previous_energy
            else:
                barrier, ts_energy = previous_atom.get_potential_energy()

        else:
            if current_energy - previous_energy > 5.0:
                print(f"The current action_idx is {action}, relative_energy is \
                       {current_energy - previous_energy}, and the structure may broken!!!!!")
                write_arc([self.atoms], name = "broken.arc")
                current_energy = previous_energy + 5.0
            if action == 1:
                if current_energy - previous_energy < -1.0:
                    barrier = 0
                elif current_energy - previous_energy >= -1.0 and current_energy - previous_energy <= 1.0:
                    barrier = np.random.normal(2, 2/3)
                else:
                    barrier = 4.0

            elif action == 2 or action == 3:
                barrier = math.log(1 + pow(math.e, current_energy-previous_energy), 10)
            elif action == 5:
                barrier = math.log(0.5 + 1.5 * pow(math.e, 2 *(current_energy - previous_energy)), 10)
            elif action == 6:
                barrier = 0.93 * pow(math.e, 0.615 * (current_energy - previous_energy)) - 0.16
            elif action == 7:
                barrier = 0.65 + 0.84 * (current_energy - previous_energy)
            else:
                barrier = 1.5
            
            if barrier > 5.0:
                barrier = 5.0
            elif barrier < -1.0:
                barrier = -1.0

            ts_energy = previous_energy + barrier

        return barrier, ts_energy

    def check_TS(self, previous_atom, current_atom, previous_energy, current_energy, action):
        barrier, ts_energy = self.transition_state_search(previous_atom, current_atom, previous_energy, current_energy, action)

        self.record_TS(ts_energy, barrier)

        return barrier

    def record_TS(self, ts_energy, barrier):
        self.TS['energies'].append(ts_energy - self.initial_energy)
        self.TS['timesteps'].append(self.history['timesteps'][-1] + 1)
        self.TS['barriers'].append(barrier)
        return

    def get_ads_d(self, ads_site):
        if ads_site[3] == 1:
            d = 1.5
        elif ads_site[3] == 2:
            d = 1.3
        else:
            d = 1.0
        return d
    
    def _adsorption(self, state):

        new_state = state.copy()
        layerList = self.get_layer_atoms(new_state)
        add_total_sites = []
        layer_ele = []

        # total_layer_O_list, total_surf_O_list = self.get_O_info(new_state)

        # surfList = self.get_surf_atoms(new_state)
        # surf_metal_list = self.get_surf_metal_atoms(new_state, surfList)

        # if len(surf_metal_list) > 3:
        surf_sites = self.get_surf_sites(new_state)
        # else:
        #     new_state = self.cluster_actions.recover_rotation(new_state, self.facet_selection, self.center_point)
        #     addable_facet_list = []
        #     prior_ads_list = []
        #     for facet in self.total_surfaces:
        #         new_state = self.cluster_actions.cluster_rotation(new_state, facet, self.center_point)
        #         list = self.get_surf_atoms(new_state)
        #         surf_metal_list_tmp = self.get_surf_metal_atoms(new_state, list)
        #         layer_list = self.get_layer_atoms(new_state)

        #         if len(surf_metal_list_tmp) > 3:
        #             for i in layer_list + list:
        #                 if i not in total_layer_O_list and i not in total_surf_O_list:
        #                     prior_ads_list.append(facet)

        #             addable_facet_list.append(facet)
        #         new_state = self.cluster_actions.recover_rotation(new_state, facet, self.center_point)
        #     if prior_ads_list:
        #         self.facet_selection = prior_ads_list[np.random.randint(len(prior_ads_list))]
        #     else:
        #         self.facet_selection = addable_facet_list[np.random.randint(len(addable_facet_list))]
        #     new_state = self.cluster_actions.cluster_rotation(new_state, self.facet_selection, self.center_point)

            # print(f'The adsorb facet is {self.facet_selection}')
        #     surf_sites = self.get_surf_sites(new_state)


        for ads_sites in surf_sites:
            for i in layerList:
                if state[i].symbol in ['Si', 'O', 'H']:
                    layer_ele.append(i)
            to_other_ele_distance = []
            if layer_ele:
                for i in layer_ele:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, state.get_positions()[i][0],
                                           state.get_positions()[i][1], state.get_positions()[i][2])
                    to_other_ele_distance.append(distance)
                if min(to_other_ele_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                add_total_sites.append(ads_sites)
        
        if add_total_sites:
            ads_site = add_total_sites[np.random.randint(len(add_total_sites))]
            choosed_adsorbate = np.random.randint(len(self.ads_list))
            ads = self.ads_list[choosed_adsorbate]
            
            del self.ads_list[choosed_adsorbate]

            if ads:
                if ads == 2:
                    self.n_O2 -= 1
                    d = self.get_ads_d(ads_site)
                    O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + d))
                    O2 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + d + 1.21))
                    new_state = new_state + O1
                    new_state = new_state + O2

                elif ads == 3:
                    self.n_O3 -= 1
                    O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + d))
                    O2 = Atom('O', (ads_site[0], ads_site[1] + 1.09, ads_site[2] + d + 0.67))
                    O3 = Atom('O', (ads_site[0], ads_site[1] - 1.09, ads_site[2] + d + 0.67))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    new_state = new_state + O3

        return new_state
    
    def _desorption(self, state):
        action_done = True
        new_state = state.copy()

        ana = Analysis(new_state)
        OOBonds = ana.get_bonds('O','O',unique = True)

        desorblist = []

        if OOBonds[0]:
            desorb,  _ = self.to_desorb_adsorbate(new_state)
            if len(desorb):
                if len(desorb) == 2:
                    self.ads_list.append(2)
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                elif len(desorb) == 3:
                    self.ads_list.append(3)
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                    desorblist.append(desorb[2])

                del new_state[[i for i in range(len(new_state)) if i in desorblist]]

                if len(desorb) == 2:
                    self.n_O2 += 1

                elif len(desorb) == 3:
                    self.n_O3 += 1
            else:
                action_done = False
        action_done = False
        return new_state, action_done
    
    def _to_translate(self, atoms):
        layer_atom, surf_atom, sub_atom, deep_atom = self.get_atom_info(atoms)
        # if abs(facet[0]) + abs(facet[1]) + abs(facet[2]) == 1:
        #     muti_movement = 2.791 * np.array([math.sqrt(2)/2, math.sqrt(2)/2, 0])
        # elif abs(facet[0]) + abs(facet[1]) + abs(facet[2]) == 2:
        #     muti_movement = 2.791 * np.array([0, 1, 0])
        # elif abs(facet[0]) + abs(facet[1]) + abs(facet[2]) == 3:
        #     muti_movement = 2.791 * np.array([1, 0, 0])
        # else:
        #     raise ValueError(f"Currently, we only consider the cluster with low Miller surface, \
        #                      including ([1 0 0], [1 1 0], [1 1 1] surface)")

        # initial_positions = atoms.positions

        # a = np.random.choice([-1, 1])
        # for atom in initial_positions:
        #     if atom in self.surf_atom:
        #         atom += a * muti_movement
        #     elif atom in self.layer_atom:
        #         atom += a * muti_movement
        # atoms.positions = initial_positions

        lamada_d = 0.2
        lamada_s = 0.4
        lamada_layer = 0.6
        initial_positions = atoms.positions

        muti_movement = np.array([np.random.normal(0.25,0.25), np.random.normal(0.25,0.25), np.random.normal(0.25,0.25)])

        for atom in initial_positions:
            if atom in deep_atom:
                atom += lamada_d * muti_movement
            if atom in sub_atom:
                atom += lamada_s * muti_movement
            if atom in surf_atom:
                atom += lamada_layer * muti_movement
            if atom in layer_atom:
                atom += lamada_layer * muti_movement
        atoms.positions = initial_positions

    
    def _get_rotate_matrix(self, zeta):
        matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
        matrix = np.array(matrix)

        return matrix
    
    def _to_rotate(self, atoms, zeta):
        initial_state = atoms.copy()

        zeta = math.pi * zeta / 180
        surf_matrix = self._get_rotate_matrix(zeta * 3)
        sub_matrix = self._get_rotate_matrix(zeta * 2)
        deep_matrix = self._get_rotate_matrix(zeta)

        rotation_surf_list = []

        surf_list = self.get_surf_atoms(atoms)
        layer_list = self.get_layer_atoms(atoms)
        rotation_sub_list = self.get_sub_atoms(atoms)
        rotation_deep_list = self.get_deep_atoms(atoms)

        for i in surf_list:
            rotation_surf_list.append(i)
        for j in layer_list:
            rotation_surf_list.append(j)

        rotation_surf_list = [i for n, i in enumerate(rotation_surf_list) if i not in rotation_surf_list[:n]]

        central_point = self.mid_point(atoms, surf_list)

        for atom in initial_state:

            if atom.index in rotation_surf_list:
                atom.position += np.array(
                        (np.dot(surf_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
            elif atom.index in rotation_sub_list:
                atom.position += np.array(
                        (np.dot(sub_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
            elif atom.index in rotation_deep_list:
                atom.position += np.array(
                        (np.dot(deep_matrix, (np.array(atom.position.tolist()) - central_point).T).T + central_point).tolist()) - atom.position
                
        atoms.positions = initial_state.get_positions()

    def _md(self, atoms:Atoms) -> None:
        self.to_constraint(atoms, all_zeolite=self.in_zeolite)
        if self.calculate_method in ['MACE', 'mace', 'Mace']:
            atoms = self.calculator.to_calc(atoms, calc_type = 'md')
        elif self.calculate_method  in ['LASP', 'Lasp', 'lasp']:
            atoms = self.calculator.to_calc(atoms, calc_type = 'ssw')
    
    def _diffusion(self, slab):
        action_done = True
        total_layer_O, _ = self.get_O_info(slab)
        
        # If there exists single O on the surf layer in the system, then it can diffuse
        if total_layer_O:
            to_diffuse_O_list = []
            diffuse_sites = []

            single_layer_O = self.layer_O_atom_list(slab)

            # If there not exists single O on the surf layer on the selected facet, then
            if not single_layer_O:
                slab = self.cluster_actions.recover_rotation(slab, self.facet_selection, self.center_point)
                diffusable_facet_list = []
                for facet in self.total_surfaces:
                    slab = self.cluster_actions.cluster_rotation(slab, facet, self.center_point)

                    single_layer_O_tmp = self.layer_O_atom_list(slab)

                    if single_layer_O_tmp:
                        diffusable_facet_list.append(facet)
                    slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)
            
                #找到了可以扩散的O所在的facet
                if diffusable_facet_list:
                    self.facet_selection = diffusable_facet_list[np.random.randint(len(diffusable_facet_list))]
                else:
                    action_done = False

                slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)

            c_layer_List = self.get_layer_atoms(slab)

            c_layer_ele = []
            for i in slab:
                if i.index in c_layer_List and i.symbol in ['Si', 'O', 'H']:
                    c_layer_ele.append(i.index)

            single_layer_O_c = self.layer_O_atom_list(slab)

            if single_layer_O_c: # 防止氧原子被trap住无法diffuse
                for i in single_layer_O_c:
                    to_other_O_distance = []
                    for j in c_layer_ele:
                        if j != i:
                            distance = self.distance(slab.get_positions()[i][0],
                                                    slab.get_positions()[i][1], slab.get_positions()[i][2],slab.get_positions()[j][0],
                                                    slab.get_positions()[j][1], slab.get_positions()[j][2])
                            to_other_O_distance.append(distance)
                                    
                    if self.to_get_min_distances(to_other_O_distance,4):
                        d_min_4 = self.to_get_min_distances(to_other_O_distance, 4)
                        if d_min_4 > 1.5 * d_O_O:
                            to_diffuse_O_list.append(i)
                    else:
                        to_diffuse_O_list.append(i)

            neigh_facets = self.neighbour_facet(slab)

            diffusable_facet = []
            for neigh_facet in neigh_facets:
                slab = self.cluster_actions.cluster_rotation(slab, neigh_facet, self.center_point)
                neigh_surf_list = self.get_surf_atoms(slab)
                neigh_surf_metal_list = self.get_surf_metal_atoms(slab, neigh_surf_list)
                if len(neigh_surf_metal_list) > 3:
                    sites = self.get_surf_sites(slab)
                    if sites.size:
                        diffusable_facet.append(neigh_facet)

                slab = self.cluster_actions.recover_rotation(slab, neigh_facet, self.center_point)

            if diffusable_facet:
                to_diffuse_facet = diffusable_facet[np.random.randint(len(diffusable_facet))]
                self.facet_selection = to_diffuse_facet

            slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)

            diffuse_sites = self.get_surf_sites(slab)

            diffusable_sites = []
            interference_O_distance = []
            diffusable = True

            for ads_sites in diffuse_sites:    # 寻找可以diffuse的位点
                to_other_O_distance = []
                if single_layer_O_c:
                    for i in single_layer_O_c:
                        distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.5, slab.get_positions()[i][0],
                                                    slab.get_positions()[i][1], slab.get_positions()[i][2])
                        to_other_O_distance.append(distance)
                    if min(to_other_O_distance) > 1.5 * d_O_O:
                        ads_sites[4] = 1
                    else:
                        ads_sites[4] = 0
                else:
                    ads_sites[4] = 1
                if ads_sites[4]:
                    diffusable_sites.append(ads_sites)
            
            if self.in_zeolite:
                to_diffuse_O_list = [atom_idx for atom_idx in range(len(to_diffuse_O_list)) if atom_idx in total_layer_O]

            if to_diffuse_O_list and diffusable_sites:
                selected_O_index = to_diffuse_O_list[np.random.randint(len(to_diffuse_O_list))]
                diffuse_site = diffusable_sites[np.random.randint(len(diffusable_sites))]

                interference_O_list = [i for i in c_layer_ele if i != selected_O_index]

                for j in interference_O_list:
                    d = self.atom_to_traj_distance(slab.positions[selected_O_index], diffuse_site, slab.positions[j])
                    interference_O_distance.append(d)

                if interference_O_distance:
                    if min(interference_O_distance) < 0.3 * d_O_O:
                        diffusable = False
                    
                if diffusable:
                    for atom in slab:
                        if atom.index == selected_O_index:
                            d = self.get_ads_d(diffuse_site)
                            atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + d])
                else:
                    action_done = False
            else:
                action_done = False
        else:
            action_done = False
        
        return slab, action_done

    def to_expand_lattice(self, slab, expand_layer, expand_surf, expand_lattice):


        layerList = self.get_layer_atoms(slab)
        surfList = self.get_surf_atoms(slab)
        subList = self.get_sub_atoms(slab)

        if layerList:
            mid_point_layer = self.mid_point(slab, layerList)
        else:
            mid_point_layer = self.mid_point(slab,surfList)
        mid_point_surf = self.mid_point(slab, surfList)
        mid_point_sub = self.mid_point(slab, subList)

        for i in slab:
            slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_layer[0]) * expand_lattice + \
                                             mid_point_layer[0]
            slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_layer[1]) * expand_lattice + \
                                             mid_point_layer[1]
            if i.index in layerList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_layer[0]) * expand_layer + \
                                             mid_point_layer[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_layer[1]) * expand_layer + \
                                             mid_point_layer[1]
            if i.index in surfList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_surf[0]) * expand_surf + \
                                             mid_point_surf[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_surf[1]) * expand_surf + \
                                             mid_point_surf[1]
            if i.index in subList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_sub[0]) * expand_lattice + \
                                             mid_point_sub[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_sub[1]) * expand_lattice + \
                                             mid_point_sub[1]
        return slab

    def get_reward_tanh(self, relative_energy):
        reward = math.tanh(-relative_energy/(self.H * self.k * self.temperature_K))
        return reward
    
    def get_reward_sigmoid(self, relative_energy):
        return 2 * (0.5 - 1 / (1 + np.exp(-relative_energy/(self.H * self.k * self.temperature_K))))
    
    def _drill(self, slab):
        action_done = True

        total_layer_O, total_sub_O = self.get_O_info(slab)

        if total_layer_O or total_sub_O:
            selected_drill_O_list = []

            layer_O_atom_list = self.layer_O_atom_list(slab)
            sub_O_atom_list = self.sub_O_atom_list(slab)

            if layer_O_atom_list:
                for i in layer_O_atom_list:
                    selected_drill_O_list.append(i)
            if sub_O_atom_list:
                for j in sub_O_atom_list:
                    selected_drill_O_list.append(j)

            if not selected_drill_O_list:
                slab = self.cluster_actions.recover_rotation(slab, self.facet_selection, self.center_point)
                drillable_facet_list = []
                for facet in self.total_surfaces:
                    slab = self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
                    layer_O_atom_list_p = self.layer_O_atom_list(slab)
                    sub_O_atom_list_p = self.sub_O_atom_list(slab)
                    selected_drill_O_list_p = []

                    if layer_O_atom_list_p:
                        for i in layer_O_atom_list_p:
                            selected_drill_O_list_p.append(i)
                    if sub_O_atom_list_p:
                        for j in sub_O_atom_list_p:
                            selected_drill_O_list_p.append(j)
                    if selected_drill_O_list_p:
                        drillable_facet_list.append(facet)
                    slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)
                    
                if drillable_facet_list:
                    self.facet_selection = drillable_facet_list[np.random.randint(len(drillable_facet_list))]
                else:
                    action_done = False
                slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)

            c_layer_O_atom_list = self.layer_O_atom_list(slab)
            c_sub_O_atom_list = self.sub_O_atom_list(slab)

            c_selected_drill_O_list = []
            if c_layer_O_atom_list:
                for i in c_layer_O_atom_list:
                    c_selected_drill_O_list.append(i)
            if c_sub_O_atom_list:
                for j in c_sub_O_atom_list:
                    c_selected_drill_O_list.append(j)

            if c_selected_drill_O_list:
                selected_O = c_selected_drill_O_list[np.random.randint(len(c_selected_drill_O_list))]
                
                if selected_O in c_layer_O_atom_list:
                    slab, action_done = self.to_drill_surf(slab, selected_O)
                elif selected_O in c_sub_O_atom_list:
                    slab, action_done = self.to_drill_deep(slab, selected_O)

            else:
                action_done = False
        else:
            action_done = False

        return slab, action_done

    def lifted_distance(self, drill_site, pos):

        r = self.distance(drill_site[0], drill_site[1], drill_site[2] +1.3,
                                    pos[0], pos[1], pos[2])
        
        lifted_d = math.exp(- r * r / (2 * 2.5 ** 2))

        return min(lifted_d, 0.5)
    
    def to_drill_surf(self, slab, selected_drill_atom):
        action_done = True

        layer_O = []
        to_distance = []
        drillable_sites = []
        layer_List = self.get_layer_atoms(slab)

        sub_sites = self.get_sub_sites(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        for ads_sites in sub_sites:
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)

        position = slab.get_positions()[selected_drill_atom]
        for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = sub_sites[to_distance.index(min(to_distance))]
            
            for atom in slab:
                if atom.index == selected_drill_atom:
                    atom.position = np.array([drill_site[0], drill_site[1], drill_site[2] +1.0])

            lifted_atoms_list = []
            current_surfList = self.get_surf_atoms(slab)
            current_layerList = self.get_layer_atoms(slab)

            for layer_atom in current_layerList:
                lifted_atoms_list.append(layer_atom)

            for surf_atom in current_surfList:
                lifted_atoms_list.append(surf_atom)

            for lifted_atom in lifted_atoms_list:
                slab.positions[lifted_atom][2] += self.lifted_distance(drill_site, slab.get_positions()[lifted_atom])
        else:
            action_done = False
        return slab, action_done
    
    def to_drill_deep(self, slab, selected_drill_atom):
        action_done = True
        to_distance = []
        drillable_sites = []
        sub_O_atom_list = self.sub_O_atom_list(slab)

        deep_sites = self.get_deep_sites(slab)
        for ads_sites in deep_sites:
            to_other_O_distance = []
            if sub_O_atom_list:
                for i in sub_O_atom_list:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)

        position = slab.get_positions()[selected_drill_atom]
        for drill_site in drillable_sites:
            to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = deep_sites[to_distance.index(min(to_distance))]
            for atom in slab:
                if atom.index == selected_drill_atom:
                    atom.position = np.array([drill_site[0], drill_site[1], drill_site[2] +1.0])

            lifted_atoms_list = []
            current_surfList = self.get_surf_atoms(slab)
            current_layerList = self.get_layer_atoms(slab)
            current_subList = self.get_sub_atoms(slab)

            for surf_atom in current_surfList:
                lifted_atoms_list.append(surf_atom)

            for sub_atom in current_subList:
                lifted_atoms_list.append(sub_atom)

            for layer_atom in current_layerList:
                lifted_atoms_list.append(layer_atom)

            for lifted_atom in lifted_atoms_list:
                slab.positions[lifted_atom][2] += self.lifted_distance(drill_site, slab.get_positions()[lifted_atom])

        else:
            action_done = False
        return slab, action_done

    def _dissociation(self, slab):
        action_done = True
                
        dissociate_O2_list = self.get_dissociate_O2_list(slab)

        if not dissociate_O2_list:
            # TODO: some tricks here
            slab = self.cluster_actions.recover_rotation(slab, self.facet_selection, self.center_point)
            dissociable_facet_list = []
            for facet in self.total_surfaces:
                slab = self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
                dissociate_O2_list_tmp = self.get_dissociate_O2_list(slab)

                if dissociate_O2_list_tmp:
                    dissociable_facet_list.append(facet)
                slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)
            if dissociable_facet_list:    
                self.facet_selection = dissociable_facet_list[np.random.randint(len(dissociable_facet_list))]
            else:
                action_done = False
            slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)

        dissociate_O2_list_c = self.get_dissociate_O2_list(slab)

        if dissociate_O2_list_c:
            OO = dissociate_O2_list_c[np.random.randint(len(dissociate_O2_list_c))]
            # print(OO)

            zeta = self.get_angle_with_z(slab, OO) * 180/ math.pi -5
            fi = 0
            slab = self.oxy_rotation(slab, OO, zeta, fi)
            slab, action_done = self.to_dissociate(slab, OO)
        else:
            action_done = False

        # print(f'Whether dissociate done is {action_done}')

        return slab, action_done
    
    def get_dissociate_O2_list(self, slab:Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O','O',unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique = True)

        Pd_O_list = []
        dissociate_O2_list = []

        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])

        if OOBonds[0]:
            layerList = self.get_layer_atoms(slab)
            for j in OOBonds[0]:
                if (j[0] in layerList or j[1] in layerList) and (j[0] in Pd_O_list or j[1] in Pd_O_list):
                    dissociate_O2_list.append([(j[0],j[1])])

        dissociate_O2_list = self._map_zeolite(dissociate_O2_list, with_zeolite=self.in_zeolite)
        return dissociate_O2_list

    def label_atoms(self, atoms, zRange):
        myPos = atoms.get_positions()
        return [
            i for i in range(len(atoms)) \
            if min(zRange) < myPos[i][2] < max(zRange)
        ]

    def distance(self, x1, y1, z1, x2, y2, z2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
        return dis

    def _generate_initial_slab(self, re_read = False) -> ase.atoms:
        if os.path.exists('./initial_input.arc') and re_read and self.in_zeolite:
            atoms = read_arc('initial_input.arc')[0]
        else:
            atoms = self._mock_cluster()
        return atoms    
    
    def _mock_cluster(self) -> ase.Atoms:
        if os.path.exists('./mock.xyz'):
            atoms = read('mock.xyz')
        else:
            surfaces = [(1, 0, 0),(1, 1, 0), (1, 1, 1)]
            esurf = [1.0, 1.0, 1.0]   # Surface energies.
            lc = 3.89
            size = 147 # Number of atoms
            atoms = wulff_construction(self.cluster_metal, surfaces, esurf,
                                    size, 'fcc',
                                    rounding='closest', latticeconstant=lc)

            uc = np.array([[30.0, 0, 0],
                            [0, 30.0, 0],
                            [0, 0, 30.0]])

            atoms.set_cell(uc)
        return atoms

    
    def RMSD(self, current_atoms:Atoms, previous_atoms:Atoms):
        similar = False

        len_atom_p = len(previous_atoms.get_positions())
        len_atom_c = len(current_atoms.get_positions())

        RMSD = 0
        cell_x = current_atoms.cell[0][0]
        cell_y = current_atoms.cell[1][1]
        if len_atom_p == len_atom_c:
            for i in range(len_atom_p):
                d = self.distance(previous_atoms.positions[i][0], previous_atoms.positions[i][1], previous_atoms.positions[i][2],
                                  current_atoms.positions[i][0], current_atoms.positions[i][1], current_atoms.positions[i][2])
                if d > max(cell_x, cell_y) / 2:
                    d = self._get_pbc_min_dis(previous_atoms, current_atoms, i)
                    
                RMSD += d * d
            RMSD = math.sqrt(RMSD / len_atom_p)
            if RMSD <= 0.5:
                similar = True

        return [similar, RMSD]
    
    def mid_point(self, slab:Atoms, List:List) -> List:
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for i in slab:
            if i.index in List:
                sum_x += slab.get_positions()[i.index][0]
                sum_y += slab.get_positions()[i.index][1]
                sum_z += slab.get_positions()[i.index][2]
        mid_point = [sum_x/len(List), sum_y/len(List), sum_z/len(List)]
        return mid_point
    
    def get_atom_info(self, atoms:Atoms) -> Tuple:
        layerList = self.get_layer_atoms(atoms)
        surfList = self.get_surf_atoms(atoms)
        subList = self.get_sub_atoms(atoms)
        deepList = self.get_deep_atoms(atoms)
        
        layer = atoms.copy()
        del layer[[i for i in range(len(layer)) if i not in layerList]]
        layer_atom = layer.get_positions()

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]
        surf_atom = surf.get_positions()

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if i not in subList]]
        sub_atom = sub.get_positions()

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if i not in deepList]]
        deep_atom = deep.get_positions()

        return layer_atom, surf_atom, sub_atom, deep_atom

    def get_total_surf_sites(self,atoms):
        total_surf_sites = []

        for facet in self.total_surfaces:
            self.cluster_actions.cluster_rotation(atoms, facet, self.center_point)
            total_surf_sites.append(self.get_surf_sites(atoms))
            self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
        # self.facet_sites_dict['sites'] = total_surf_sites

        return total_surf_sites

    def get_total_sub_sites(self,atoms):
        total_sub_sites = []
        
        for facet in self.total_surfaces:
            self.cluster_actions.cluster_rotation(atoms, facet, self.center_point)
            total_sub_sites.append(self.get_sub_sites(atoms))
            self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
        
        return total_sub_sites
    
    
    def get_surf_sites(self, atoms):
        surfList = self.get_surf_atoms(atoms)

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if (i not in surfList) or surf[i].symbol != self.cluster_metal]]
        

        surf_sites = self.get_sites(surf)

        return surf_sites
    
    def get_sub_sites(self, atoms):
        subList = self.get_sub_atoms(atoms)

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if (i not in subList) or sub[i].symbol != self.cluster_metal]]

        sub_sites = self.get_sites(sub)
        return sub_sites
    
    def get_deep_sites(self, atoms):
        deepList = self.get_deep_atoms(atoms)

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if (i not in deepList) or deep[i].symbol != self.cluster_metal]]

        deep_sites = self.get_sites(deep)

        return deep_sites
    
    def get_sites(self, atoms):
        if len(atoms) == 1:
            sites = []
            for _ in range(2):
                sites.append([atoms.get_positions()[0][0],atoms.get_positions()[0][1],atoms.get_positions()[0][2], 1, 0])
            return np.array(sites)
        elif len(atoms) == 2:
            sites = []
            for atom in atoms:
                sites.append(np.append(atom.position, [1, 0]))
                
            sites.append(np.array([(atoms.get_positions()[0][0] + atoms.get_positions()[1][0]) / 2,
                                   (atoms.get_positions()[0][1] + atoms.get_positions()[1][1]) / 2,
                                   (atoms.get_positions()[0][2] + atoms.get_positions()[1][2]) / 2,
                                   2, 0]))
            return np.array(sites)

        elif len(atoms) >= 3:
            atop = atoms.get_positions()
            pos_ext = atoms.get_positions()
            tri = Delaunay(pos_ext[:, :2])
            pos_nodes = pos_ext[tri.simplices]

            bridge_sites = []
            hollow_sites = []

            for i in pos_nodes:
                if (self.distance(i[0][0], i[0][1], i[0][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[0] + i[1]) / 2)
                else:
                    hollow_sites.append((i[0] + i[1]) / 2)
                if (self.distance(i[2][0], i[2][1], i[2][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[2] + i[1]) / 2)
                else:
                    hollow_sites.append((i[2] + i[1]) / 2)
                if (self.distance(i[0][0], i[0][1], i[0][2], i[2][0], i[2][1], i[2][2])) < 3.0:
                    bridge_sites.append((i[0] + i[2]) / 2)
                else:
                    hollow_sites.append((i[0] + i[2]) / 2)

            top_sites = np.array(atop)
            hollow_sites = np.array(hollow_sites)
            bridge_sites = np.array(bridge_sites)

            sites_1 = []
            total_sites = []

            for i in top_sites:
                sites_1.append(np.transpose(np.append(i, 1)))
            for i in bridge_sites:
                sites_1.append(np.transpose(np.append(i, 2)))
            for i in hollow_sites:
                sites_1.append(np.transpose(np.append(i, 3)))
            for i in sites_1:
                total_sites.append(np.append(i, 0))

            total_sites = np.array(total_sites)

            return total_sites
        
    def _get_constraint(self, atoms:ase.Atoms, all_zeolite: Optional[bool] = False) -> FixAtoms:
        if all_zeolite:
            constrain_list = list(range(len(self.zeolite)))
        else:
            constrain_list = self.fix_list
        constraint = FixAtoms(mask=[a.index in constrain_list for a in atoms]) 
        return constraint
    
    def to_constraint(self, atoms:Atoms, all_zeolite: Optional[bool] = False) -> None: # depending on such type of atoms
        constraint = self._get_constraint(atoms, all_zeolite)
        atoms.set_constraint(constraint)

    def exist_too_short_bonds(self,slab:Atoms):
        exist = False
        ana = Analysis(slab)
        PdPdBonds = ana.get_bonds(self.cluster_metal,self.cluster_metal,unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique=True)
        PdPdBondValues = ana.get_values(PdPdBonds)[0]
        minBonds = []
        minPdPd = min(PdPdBondValues)
        minBonds.append(minPdPd)
        if OOBonds[0]:
            OOBondValues = ana.get_values(OOBonds)[0]
            minOO = min(OOBondValues)
            minBonds.append(minOO)
        if PdOBonds[0]:    
            PdOBondValues = ana.get_values(PdOBonds)[0]
            minPdO = min(PdOBondValues)
            minBonds.append(minPdO)

        if min(minBonds) < 0.4:
            exist = True
        return exist
    
    def atom_to_traj_distance(self, atom_A, atom_B, atom_C):
        d_AB = self.distance(atom_A[0], atom_A[1], atom_A[2], atom_B[0], atom_B[1], atom_B[2])
        d = abs((atom_C[0]-atom_A[0])*(atom_A[0]-atom_B[0])+
                (atom_C[1]-atom_A[1])*(atom_A[1]-atom_B[1])+
                (atom_C[2]-atom_A[2])*(atom_A[2]-atom_B[2])) / d_AB
        return d

    def to_get_bond_info(self, slab):
        ana = Analysis(slab)
        PdPdBonds = ana.get_bonds(self.cluster_metal,self.cluster_metal,unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique=True)
        PdPdBondValues = ana.get_values(PdPdBonds)[0]
        if OOBonds[0]:
            OOBondValues = ana.get_values(OOBonds)[0]
            if PdOBonds[0]:
                PdOBondValues = ana.get_values(PdOBonds)[0]
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(OOBondValues) < d_O_O * 0.80 or min(PdOBondValues) < d_O_Pd * 0.80 or max(OOBondValues) > 1.15 * d_O_O:
                    return True
                else:
                    return False
            else:
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(OOBondValues) < d_O_O * 0.80  or max(OOBondValues) > 1.15 * d_O_O:
                    return True
                else:
                    return False
        else:
            if PdOBonds[0]:
                PdOBondValues = ana.get_values(PdOBonds)[0]
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(PdOBondValues) < d_O_Pd * 0.80:
                    return True
                else:
                    return False
            else:
                if min(PdPdBondValues) < d_Pd_Pd * 0.80:
                    return True
                else:
                    return False
    
    def ball_func(self,pos1, pos2, zeta, fi):	# zeta < 36, fi < 3
        d = self.distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
        zeta = -math.pi * zeta / 180
        fi = -math.pi * fi / 180
        '''如果pos1[2] > pos2[2],atom_1旋转下来'''
        pos2_position = pos2
        # pos1_position = [pos2[0]+ d*sin(zeta)*cos(fi), pos2[1] + d*sin(zeta)*sin(fi),pos2[2]+d*cos(zeta)]
        pos_slr = pos1 - pos2

        pos_slr_square = math.sqrt(pos_slr[0] * pos_slr[0] + pos_slr[1] * pos_slr[1])
        pos1_position = [pos2[0] + d * pos_slr[0]/pos_slr_square, pos2[1] + d * pos_slr[1]/pos_slr_square, pos2[2]]

        return pos1_position, pos2_position

    def oxy_rotation(self, slab, OO, zeta, fi):
        if slab.positions[OO[0][0]][2] > slab.positions[OO[0][1]][2]:
            a,b = self.ball_func(slab.get_positions()[OO[0][0]], slab.get_positions()[OO[0][1]], zeta, fi)
        else:
            a,b = self.ball_func(slab.get_positions()[OO[0][1]], slab.get_positions()[OO[0][0]], zeta, fi)
        slab.positions[OO[0][0]] = a
        slab.positions[OO[0][1]] = b
        return slab
    
    def to_dissociate(self, slab, atoms):
        action_done = True

        expanding_index = 2.0
        # print(f'Before dissociate, the position of atom_1 = {slab.positions[atoms[0][0]]}, the position of atom_2 = {slab.positions[atoms[0][1]]}')
        central_point = np.array([(slab.get_positions()[atoms[0][0]][0] + slab.get_positions()[atoms[0][1]][0])/2, 
                                  (slab.get_positions()[atoms[0][0]][1] + slab.get_positions()[atoms[0][1]][1])/2, 
                                  (slab.get_positions()[atoms[0][0]][2] + slab.get_positions()[atoms[0][1]][2])/2])
        slab.positions[atoms[0][0]] += np.array([expanding_index*(slab.get_positions()[atoms[0][0]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][2]-central_point[2])])
        slab.positions[atoms[0][1]] += np.array([expanding_index*(slab.get_positions()[atoms[0][1]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][2]-central_point[2])])
        
        addable_sites = []
        layer_interfere_ele = []
        layerlist = self.get_layer_atoms(slab)

        surfList = self.get_surf_atoms(slab)
        surf_metal_list = self.get_surf_metal_atoms(slab, surfList)

        if len(surf_metal_list) > 3:
            surf_sites = self.get_surf_sites(slab)
        else:
            neigh_facets = self.neighbour_facet(slab)
            # now the facet has been recovered
            to_dissociate_facet_list = []
            for facet in neigh_facets:
                slab = self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
                neigh_surf_list = self.get_surf_atoms(slab)
                neigh_surf_metal_list = self.get_surf_metal_atoms(slab, neigh_surf_list)
                if len(neigh_surf_metal_list) > 3:
                    to_dissociate_facet_list.append(facet)
                slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)
            
            if to_dissociate_facet_list:
                self.facet_selection = to_dissociate_facet_list[np.random.randint(len(to_dissociate_facet_list))]

            slab = self.cluster_actions.cluster_rotation(slab, self.facet_selection, self.center_point)
            surf_sites = self.get_surf_sites(slab)

        for ads_site in surf_sites:
            for atom_index in layerlist:
                if slab[atom_index].symbol in ['Si', 'O', 'H']:
                    layer_interfere_ele.append(atom_index)
            to_other_ele_distance = []
            if layer_interfere_ele:
                for i in layer_interfere_ele:
                    to_distance = self.distance(ads_site[0], ads_site[1], ads_site[2] + 1.5, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_ele_distance.append(to_distance)
                if min(to_other_ele_distance) > 1.5 * d_O_O:
                    ads_site[4] = 1
            else:
                ads_site[4] = 1
            if ads_site[4]:
                addable_sites.append(ads_site)

        O1_distance = []
        for add_1_site in addable_sites:
            distance_1 = self.distance(add_1_site[0], add_1_site[1], add_1_site[2] + 1.3, slab.get_positions()[atoms[0][0]][0],
                                           slab.get_positions()[atoms[0][0]][1], slab.get_positions()[atoms[0][0]][2])
            O1_distance.append(distance_1)

        if O1_distance:
            O1_site = addable_sites[O1_distance.index(min(O1_distance))]
            
            ad_2_sites = []
            for add_site in addable_sites:
                d = self.distance(add_site[0], add_site[1], add_site[2] + 1.3, O1_site[0], O1_site[1], O1_site[2])
                if d > 2.0 * d_O_O:
                    ad_2_sites.append(add_site)

            O2_distance = []
            for add_2_site in ad_2_sites:
                distance_2 = self.distance(add_2_site[0], add_2_site[1], add_2_site[2] + 1.3, slab.get_positions()[atoms[0][1]][0],
                                            slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][1]][2])
                O2_distance.append(distance_2)
            
            if O2_distance:
                O2_site = ad_2_sites[O2_distance.index(min(O2_distance))]
            else:
                O2_site = O1_site
                
            d_1 = self.get_ads_d(O1_site)
            d_2 = self.get_ads_d(O2_site)

            print(f'site_1 = {O1_site}, site_2 = {O2_site}')
            if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
                    
                O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1])
                O_2_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1 + 1.21])
                action_done = False
            else:
                O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + d_1])
                O_2_position = np.array([O2_site[0], O2_site[1], O2_site[2] + d_2])

            for atom in slab:
                if atom.index == atoms[0][0]:
                    atom.position = O_1_position
                elif atom.index == atoms[0][1]:
                    atom.position = O_2_position

            # print(f'And after modified, the position of atom_1 = {slab.positions[atoms[0][0]]}, the position of atom_2 = {slab.positions[atoms[0][1]]}')
        else:
            action_done = False
        return slab, action_done
    
    def get_angle_with_z(self,slab, atoms):
        if slab.positions[atoms[0][0]][2] > slab.positions[atoms[0][1]][2]:
            a = np.array([slab.get_positions()[atoms[0][0]][0] - slab.get_positions()[atoms[0][1]][0], slab.get_positions()[atoms[0][0]][1] - slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][0]][2] - slab.get_positions()[atoms[0][1]][2]])
        else:
            a = np.array([slab.get_positions()[atoms[0][1]][0] - slab.get_positions()[atoms[0][0]][0], slab.get_positions()[atoms[0][1]][1] - slab.get_positions()[atoms[0][0]][1], slab.get_positions()[atoms[0][1]][2] - slab.get_positions()[atoms[0][0]][2]])
        z = np.array([0,0,1])
        zeta = math.asin(np.dot(a,z)/math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]))
        return zeta
    
    def _get_pbc_min_dis(self, atoms_1, atoms_2, i):
        atom_x = atoms_1.cell[0][0]
        atom_y = atoms_1.cell[1][1]

        d = []
        atom_1_x = atoms_1.get_positions()[i][0]
        atom_1_y = atoms_1.get_positions()[i][1]
        atom_1_z = atoms_1.get_positions()[i][2]

        atom_2_x = [atoms_2.get_positions()[i][0], atoms_2.get_positions()[i][0] - atom_x, atoms_2.get_positions()[i][0] + atom_x]
        atom_2_y = [atoms_2.get_positions()[i][1], atoms_2.get_positions()[i][1] - atom_y, atoms_2.get_positions()[i][1] + atom_y]
        z = atoms_2.get_positions()[i][2]

        for x in atom_2_x:
            for y in atom_2_y:
                d.append(self.distance(atom_1_x, atom_1_y, atom_1_z, x, y, z))
        
        dis = min(d)
        return dis
    
    def to_get_min_distances(self, a, min_point):
        for i in range(len(a) - 1):
            for j in range(len(a)-i-1):
                if a[j] > a[j+1]:
                    a[j], a[j+1] = a[j+1], a[j]
        if len(a):
            if len(a) < min_point:
                return a[-1]
            else:
                return a[min_point - 1]
        else:
            return False

    def RMSE(self, a:list):
        mean = np.mean(a)
        b = mean * np.ones(len(a))
        diff = np.subtract(a, b)
        square = np.square(diff)
        MSE = square.mean()
        RMSE = np.sqrt(MSE)
        return RMSE

    def to_ads_adsorbate(self, slab):
        ads = ()
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique=True)

        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        Pd_O_list = []
        ads_list = []
        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] not in Pd_O_list and i[1] not in Pd_O_list:
                    ads_list.append(i)

        if OOOangles[0]:
            for j in OOOangles[0]:
                if j[0] not in Pd_O_list and j[1] not in Pd_O_list and j[2] not in Pd_O_list:
                    ads_list.append(i)

        if ads_list:
            ads = ads_list[np.random.randint(len(ads_list))]
        return ads, ads_list
    
    def to_desorb_adsorbate(self, slab):
        desorb = ()
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.cluster_metal, 'O', unique=True)

        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        Pd_O_list = []
        desorb_list = []
        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] in Pd_O_list or i[1] in Pd_O_list:
                    desorb_list.append(i)

        if OOOangles[0]:
            for j in OOOangles[0]:
                if j[0] in Pd_O_list or j[1] in Pd_O_list or j[2] in Pd_O_list:
                    desorb_list.append(j)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
        return desorb, desorb_list
    
    def _2D_distance(self, x1,x2, y1,y2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        return dis

    def get_layer_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        layerlist = self.label_atoms(atoms, [z_max - 1.0, z_max + 6.0])

        sum_z = 0
        if layerlist:
            for i in layerlist:
                sum_z += atoms.get_positions()[i][2]

            modified_z = sum_z / len(layerlist)
            modified_layer_list = self.label_atoms(atoms, [modified_z - 1.0, modified_z + 1.0])
        else:
            modified_layer_list = layerlist
        

        return modified_layer_list
    
    def modify_slab_layer_atoms(self, atoms, list):
        sum_z = 0

        if list:
            for i in list:
                sum_z += atoms.get_positions()[i][2]

            modified_z = sum_z / len(list)
            modified_list = self.label_atoms(atoms, [modified_z - 1.0, modified_z + 1.0])
            return modified_list
        else:
            return list
    
    def get_surf_metal_atoms(self, atoms, surfList):
        surf_metal_list = []

        if surfList:
            for index in surfList:
                if atoms[index].symbol == self.cluster_metal:
                    surf_metal_list.append(index)

        return surf_metal_list
    
    def get_surf_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)
        surf_z = z_max - r_Pd / 2

        surflist = self.label_atoms(atoms, [surf_z - 1.0, surf_z + 1.0])
        modified_surflist = self.modify_slab_layer_atoms(atoms, surflist)

        return modified_surflist
    
    def get_sub_atoms(self, atoms:Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        sub_z = z_max - r_Pd/2 - 2.0

        sublist = self.label_atoms(atoms, [sub_z - 1.0, sub_z + 1.0])
        modified_sublist = self.modify_slab_layer_atoms(atoms, sublist)

        return modified_sublist
    
    def get_deep_atoms(self, atoms:Atoms) -> List:
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == self.cluster_metal:
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)
        deep_z = z_max - r_Pd/2 - 4.0

        deeplist = self.label_atoms(atoms, [deep_z - 1.0, deep_z + 1.0])
        modified_deeplist = self.modify_slab_layer_atoms(atoms, deeplist)

        return modified_deeplist

    def neighbour_facet(self, atoms:Atoms) -> List:
        facet = self.facet_selection
        surface_list = self.get_surf_atoms(atoms)
        atoms = self.cluster_actions.recover_rotation(atoms, facet, self.center_point)
        neighbour_facet = []
        neighbour_facet.append(facet)
        for selected_facet in self.total_surfaces:
            if selected_facet[0] != facet[0] or selected_facet[1] != facet[1] or selected_facet[2] != facet[2]:
                atoms = self.cluster_actions.cluster_rotation(atoms, selected_facet, self.center_point)
                selected_surface_list = self.get_surf_atoms(atoms)
                atoms = self.cluster_actions.recover_rotation(atoms, selected_facet, self.center_point)
                repeat_atoms = [i for i in selected_surface_list if i in surface_list]
                if len(repeat_atoms) >= 2:
                    neighbour_facet.append(selected_facet)
        return neighbour_facet
        
    def layer_O_atom_list(self, slab:Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        layer_O = []
        layer_O_atom_list = []
        layer_OObond_list = []
        layer_List = self.get_layer_atoms(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        if layer_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_O or i[1] in layer_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in layer_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)
        layer_O_atom_list = self._map_zeolite(layer_O_atom_list,with_zeolite=True)
        return layer_O_atom_list
    
    def sub_O_atom_list(self, slab:ase.Atoms) -> List:
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)
        sub_O = []
        sub_O_atom_list = []
        sub_OObond_list = []
        sub_List = self.get_sub_atoms(slab)

        for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)
        
        if sub_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_O or i[1] in sub_O:
                        sub_OObond_list.append(i[0])
                        sub_OObond_list.append(i[1])

            for j in sub_O:
                if j not in sub_OObond_list:
                    sub_O_atom_list.append(j)
        sub_O_atom_list = self._map_zeolite(sub_O_atom_list, with_zeolite=True)
        return sub_O_atom_list
    
    def add_mole(self, atom:Atoms, mole:Atoms, d:float) -> float:
        new_state = atom.copy()
        energy_1  = self.calculator.to_calc(new_state, calc_type = 'single')
        if len(mole) == 2:
            ele_1 = Atom(mole[0], (atom.get_cell()[0][0] / 2, atom.get_cell()[1][1] / 2, atom.get_cell()[2][2] - 5.0))
            ele_2 = Atom(mole[1], (atom.get_cell()[0][0] / 2, atom.get_cell()[1][1] / 2, atom.get_cell()[2][2] - 5.0 + d))
            new_state = new_state + ele_1
            new_state = new_state + ele_2
        elif len(mole) == 3:
            ele_1 = Atom(mole[0], (atom.get_cell()[0][0] / 2, atom.get_cell()[1][1] / 2, atom.get_cell()[2][2] - 5.0))
            ele_2 = Atom(mole[1], (atom.get_cell()[0][0] / 2 - 0.6 * d, atom.get_cell()[1][1] / 2, atom.get_cell()[2][2] - 5.0 + 0.8 * d))
            ele_3 = Atom(mole[1], (atom.get_cell()[0][0] / 2 + 0.6 * d, atom.get_cell()[1][1] / 2, atom.get_cell()[2][2] - 5.0 + 0.8 * d))
            new_state = new_state + ele_1
            new_state = new_state + ele_2
            new_state = new_state + ele_3
        energy_2 = self.calculator.to_calc(new_state, calc_type = 'single')
        energy = energy_2 - energy_1
        return energy
    
    def get_O_info(self, slab:Atoms):
        slab = self._get_cluster(slab, with_zeolite=self.in_zeolite)

        layer_O_total = []
        sub_O_total = []

        total_O_list = []
        total_layer_atom_list = []
        total_sub_atom_list = []

        for atom in slab:
            if atom.symbol == 'O':
                total_O_list.append(atom.index)

        for facet in self.total_surfaces:
            slab= self.cluster_actions.cluster_rotation(slab, facet, self.center_point)
            layer_list = self.get_layer_atoms(slab)
            sub_list = self.get_sub_atoms(slab)

            for i in layer_list:
                if i not in total_layer_atom_list:
                    total_layer_atom_list.append(i)

            for i in sub_list:
                if i not in total_sub_atom_list:
                    total_sub_atom_list.append(i)

            slab = self.cluster_actions.recover_rotation(slab, facet, self.center_point)

        for j in total_layer_atom_list:
            if j in total_O_list:
                layer_O_total.append(j)
        
        for j in total_sub_atom_list:
            if j in total_O_list:
                sub_O_total.append(j)
        
        layer_O_total = self._map_zeolite(layer_O_total, with_zeolite=self.in_zeolite)
        sub_O_total = self._map_zeolite(sub_O_total, with_zeolite=self.in_zeolite)
        return layer_O_total, sub_O_total
    
    def get_observation_ele_positions(self, atoms:Atoms) -> List:
        ele_positions_list = []
        for atom in atoms:
            ele_position = atom.position.tolist()
            ele_position.append(atom.symbol)
            ele_positions_list.append(ele_position)
        ele_positions_list = np.array(ele_positions_list)
        return ele_positions_list
    
    def get_observation_ele_squence_positions(self, atoms:Atoms) -> List:
        ele_positions_list = []
        for atom in atoms:
            ele_position = atom.position.tolist()
            ele_position.append(ELEDICT[atom.symbol])
            ele_positions_list.append(ele_position)
        ele_positions_list = np.array(ele_positions_list)
        return ele_positions_list
    
    def _use_MLP(self, atoms:Atoms):
        nodes_scalar = torch.tensor(self.to_pad_the_array(atoms.get_atomic_numbers(), 
                                    self.max_observation_atoms, position = False))
        atom_embeddings = nn.Embedding(min(118,max(len(atoms), 50)),50)
        nodes_scalar = atom_embeddings(nodes_scalar)

        pos = self.pd_obs(torch.tensor(atoms.get_positions()))
        pos = rearrange(pos.unsqueeze(1), 'a b c ->a c b')

        # print(torch.mul(nodes_scalar.unsqueeze(1), pos).detach().numpy().shape)
        return torch.mul(nodes_scalar.unsqueeze(1), pos).detach().numpy()
    
    def _use_Painn_description(self, atoms:Atoms) -> List[np.ndarray]:
        input_dict = Painn.atoms_to_graph_dict(atoms, self.cutoff)
        atom_model = Painn.PainnDensityModel(
            num_interactions = self.num_interactions,
            hidden_state_size = self.hidden_state_size,
            cutoff = self.cutoff,
            atoms = atoms,
            embedding_size = self.embedding_size,
        )
        atom_representation_scalar, atom_representation_vector = atom_model(input_dict)

        atom_representation_scalar = np.array(self.pd_obs(torch.tensor(np.array(atom_representation_scalar[0].tolist()))))

        # print(atom_representation_vector[0].shape)
        atom_representation_vector = rearrange(atom_representation_vector[0], "a b c -> b a c")
        # print(atom_representation_vector.shape)

        atom_representation_vector = np.array(self.pd_obs(torch.tensor(np.array(atom_representation_vector.tolist()))))
        # print(atom_representation_vector.shape)
        atom_representation_vector = rearrange(atom_representation_vector, "b a c -> a b c")
        # print(atom_representation_vector.shape)

        return [atom_representation_scalar, atom_representation_vector]

    def to_pad_the_array(self, array, max_len:int = None, position:bool = True) -> np.ndarray:
        if max_len is None:
            max_len = self.max_observation_atoms
        if position:
            array = np.append(array, [0.0, 0.0, 0.0] * (max_len - array.shape[0]))
            array = array.reshape(int(array.shape[0]/3), 3)
        else:
            array = np.append(array, [0] * (max_len - array.shape[0]))
        return array
    
    '''----------------Zeolite functions-------------------'''
    def _get_zeolite(self, system:ase.Atoms) -> ase.Atoms:
        return system[[a.index for a in system if a.index in range(len(self.zeolite))]]

    def _get_cluster(self, system:ase.Atoms, with_zeolite:Optional[bool]=False) -> ase.Atoms:
        if with_zeolite:
            return system[[a.index for a in system if a.index not in range(len(self.zeolite))]]
        else:
            return system
    
    def _get_system(self, atoms:ase.Atoms, with_zeolite:Optional[bool] = False) -> ase.Atoms:
        if with_zeolite:
            return self.zeolite + atoms
        else:
            return atoms
    
    def _get_free_atoms(self, system: ase.Atoms) -> ase.Atoms:
        free_list = self._get_free_atoms_list(system)
        return system[free_list]
    
    def _get_free_atoms_list(self, system: ase.Atoms, center_point:List = None, d_max:float = None) -> List:
        """
        The function is to get the free atoms on zeolite
        """
        if self.in_zeolite:
            central_cluster = self._get_cluster(system, with_zeolite=self.in_zeolite)
            # print(f"central_cluster is {central_cluster}")
            if not center_point:
                center_point = self.cluster_actions.get_center_point(central_cluster)
            # print(f"The current center point is {center_point}")
            d_max = np.max(np.sqrt(np.diagonal(np.inner(central_cluster .get_positions() - center_point,
                                                central_cluster .get_positions() - center_point)))) + d_Si_Pd
            d_array = np.sqrt(np.diagonal(np.inner(system.get_positions() - center_point,
                                                system.get_positions() - center_point)))
            # print(f"d max is {d_max}")
            free_list = np.where(d_array < d_max)[0].tolist()
            print(f"The length of free list is {len(free_list)}")
            return free_list
        else:
            atoms = system.copy()
            free_list = []
            # for facet in atoms.get_surfaces():
            for facet in self.total_surfaces:
                atoms = self.cluster_actions.cluster_rotation(atoms, facet, center_point)
                list = self.get_surf_atoms(atoms)
                for i in list:
                    free_list.append(i)
                atoms = self.cluster_actions.recover_rotation(atoms, facet, center_point)
            return free_list
        
    def _map_zeolite(self, idx_list: List, with_zeolite: Optional[bool] = False) -> List:
        if with_zeolite:
            return (np.array(idx_list) + len(self.zeolite)).tolist()
        else:
            return idx_list