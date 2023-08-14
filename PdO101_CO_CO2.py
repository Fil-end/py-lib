import array
import sys

import numpy as np
import matplotlib
from scipy.spatial import Delaunay

matplotlib.use("agg")
import torch
import matplotlib.pyplot as plt
import ase
from ase import Atom
from ase.io.lasp_PdO import write_arc, read_arc
import os
from copy import deepcopy
import json
import itertools
from ase.units import Hartree
from ase.build import fcc100, fcc111, add_adsorbate, add_vacuum, surface
from ase.spacegroup import crystal
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.calculators.LASP_reaction import LASP
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase import Atoms
from ase.io import read, write
from ase.optimize import QuasiNewton
import gym
import math
import copy
from gym import spaces
from ase.md.langevin import Langevin
from ase import units
from math import cos, sin
import random
from random import sample
from ase.geometry.analysis import Analysis
from rdkit import Chem
import warnings
import torch
import torch.nn as nn
import GNN_utils.Painn_utils as painn
from einops import rearrange

ACTION_SPACES = ['ADS', 'Trans', 'L-Rotation', 'R-Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']  # + ‘Reaction’ 动作
# ACTION_SPACES = ['ADS', 'Trans', 'L-Rotation', 'R-Rotation','MD', 'Diffusion', 'Drill', 'Dissociation', 'Synthesis', 'Desportion'] 
# TODO:
'''1. ADS(doing), Diffusion(initial_finished), Drill(doing), Dissociation(initial_finished), Desorption(initial_finished)
'''


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
                 max_energy_profile = 0.05,
                 convergence = 0.01,
                 save_every=None,
                 save_every_min=None,
                 plot_every=None,
                 reaction_H = None,         #/(KJ/mol)
                 reaction_n = None,
                 delta_s = None,            #/eV
                 use_DESW = None,
                 lr = 0.01,
                 max_observation_atoms = 250,   # to reinforcement learning ———— deep network 
                 target_system = ['[Pd]#CO', 'PdCO'],
                 pot = 'PdCHO',
                 metal_ele = 'Pd',
                 use_GNN_description = None,    # use Painn description
                 cutoff = 4.0,  # Painn paras
                 hidden_state_size = 50,
                 embedding_size = 50,
                 num_interactions = 3):
        
        self.metal_ele = metal_ele
        self.target_system = target_system
        self.pot = pot
        
        self.initial_state = self.generate_initial_slab_clean_metal_surface()  # 设定初始结构
        self.to_constraint(self.initial_state)
        self.initial_state, self.energy, self.force = self.lasp_calc(self.initial_state)  # 由于使用lasp计算，设定初始能量

        self.ads_list = ['CO', 'OO']  # if the initial_slab is PdO, ['CO'] is enough, elif the initial_slab is pure metal ['CO', 'O2']
        # self.ads_list = ['CO', 'OO']
        self.desorb_list = ['CO', 'OO', 'OCO']
        self.diffuse_list = ['O', 'CO', '[{}]#CO'.format(self.metal_ele)]
        self.dissociation_list = ['OO', 'CO']
        self.sythesis_list = ['O-O', 'C-O', 'O-CO']
        self.drill_list = ['O', 'O', 'C', 'C']    # 两个氧指代不同，第一个氧原子为surf层的氧原子，第二个氧原子为sub层的氧原子
                                                  # 同理，两个碳原子也指代不同，第一个为surf层碳原子， 第二个为sub层碳原子  
        # self.reaction_list = [['CO', 'O'], ['CO', 'OO']]

        self.episode = 0  # 初始化episode为0
        self.max_episodes = max_episodes

        self.save_every = save_every
        self.save_every_min = save_every_min
        self.plot_every = plot_every
        self.use_DESW = use_DESW

        self.step_size = step_size  # 设定单个原子一次能移动的距离
        self.timesteps = timesteps  # 定义episode的时间步上限

        # self.episode_reward = 0  # 初始化一轮epsiode所获得的奖励
        self.timestep = 0  # 初始化时间步数为0

        # 获得氧气分子和一氧化碳分子的能量
        self.E_OO = self.add_mole(self.initial_state, 'OO', 1.21)
        self.E_CO = self.add_mole(self.initial_state, 'CO', 1.13)
        self.E_OCO = self.add_mole(self.initial_state, 'OCO', 1.16)
        
        # self.H = 112690 * 32/ 96485   # 没有加入熵校正, 单位eV
        if reaction_H:
            self.reaction_H = reaction_H
        else:
            self.reaction_H = self.E_OCO - self.E_CO - 0.5 * self.E_OO
            
        self.reaction_n = reaction_n
        self.delta_s = delta_s
        self.H = self.reaction_H * self.reaction_n

        # Painn paras
        self.cutoff = cutoff
        self.hidden_state_size = hidden_state_size  # embedding_output_dim and the hidden_dim overall the Painn
        self.num_interactions = num_interactions
        self.embedding_size = embedding_size    # embedding_hidden_dim

        self.lr = lr    # 对diffuse、dissociation、ADS、Desportion这些动作的prob_list更新的参数
        self.ele_list = self.split_element(self.target_system[0])   # 分辨总体的元素种类
        

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

        # 标记可以自由移动的原子
        self.bottomList = self.label_atoms(self.initial_state, [bottom_z - fluct_d_metal, bottom_z + fluct_d_metal])
        self.free_atoms = list(set(range(len(self.initial_state))) - set(self.bottomList))
        self.len_atom = len(self.free_atoms)
        self.convergence = convergence

        # 设定环境温度为473 K，定义热力学能
        # T = 473.15
        self.temperature_K = temperature
        self.k = k  # eV/K
        self.thermal_energy = k * temperature * self.len_atom

        self.action_space = spaces.Discrete(len(ACTION_SPACES))

        # 设定动作空间，‘action_type’为action_space中的独立动作,atom_selection为三层Pd layer和环境中的16个氧
        # movement设定为单个原子在空间中的运动（x,y,z）
        # 定义动作空间
        self.use_GNN_description = use_GNN_description
        self.max_observation_atoms = max_observation_atoms
        self.observation_space = self.get_observation_space()

        # 一轮过后重新回到初始状态
        self.reset()

        return

    def step(self, action):
        pro = 1  # 定义该step完成该动作的概率，初始化为1
        barrier = 0
        self.steps = 100  # 定义优化的最大步长
        reward = 0  # 定义初始奖励为0

        action_done = True
        self.action_idx = action
        RMSD_similar = False
        kickout = False
        RMSE = 10

        desorb_info = []

        selected_mol_index = None

        self.done = False  # 开关，决定episode是否结束
        done_similar = False
        episode_over = False  # 与done作用类似

        self.atoms, previous_structure, previous_energy = self.state

        # 定义表层、次表层、深层以及环境层的平动范围
        self.lamada_d = 0.2
        self.lamada_s = 0.4
        self.lamada_layer = 0.6
        self.lamada_env = 0

        previous_n_OO, previous_n_CO, previous_n_OCO = self.n_OO, self.n_CO, self.n_OCO

        assert self.action_space.contains(self.action_idx), "%r (%s) invalid" % (
            self.action_idx,
            type(self.action_idx),
        )

        self.muti_movement = np.array([np.random.normal(0.25,0.25), np.random.normal(0.25,0.25), np.random.normal(0.25,0.25)])
        # 定义层之间的平动弛豫

        #   定义保存ts，min和md动作时的文件路径
        save_path_ts = None
        save_path_ads = None
        save_path_md = None

        # self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint, self.layer_atom, self.surf_atom, self.sub_atom, self.deep_atom, self.envList = self.get_surf_sites(self.atoms)
        self.layer_atom, self.surf_atom, self.sub_atom, self.deep_atom, constraint, self.layerList, self.surfList, self.subList, self.deepList, self.bottomList = self.get_layer_info(self.atoms)
        self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint = self.get_surf_sites(self.atoms)

        # env_list = self.label_atoms(self.atoms, [2.0- fluct_d_layer, 2.0 + fluct_d_layer])  # 判断整个slab中是否还存在氧气分子，若不存在且动作依旧执行吸附，则强制停止
        '''_,  ads_exist = self.to_ads_adsorbate(self.atoms)
        if not ads_exist and action == 0:
            # self.done = True
            self.action_idx = 1'''
        # in the GCMC simulation, the ads num is enough

        layerList = self.label_atoms(self.atoms, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        layer_O = []
        for i in layerList:
            if self.atoms[i].symbol == 'O':
                layer_O.append(i)

        subList = self.label_atoms(self.atoms, [sub_z - fluct_d_metal , surf_z])
        sub_O = []
        for i in subList:
            if self.atoms[i].symbol == 'O':
                sub_O.append(i)
                        
        '''——————————————————————————————————————————以下是动作选择————————————————————————————————————————————————————————'''
        if self.action_idx == 0:
            self.atoms, selected_mol_index, action_done = self.choose_ads_site(self.atoms, self.total_s)
            # return new_state,new_state_energy

        elif self.action_idx == 1:
            self.translation()

        elif self.action_idx == 2:
            self.rotation(9)

        elif self.action_idx == 3:
            self.rotation(-9)


        elif self.action_idx == 4:
            self.atoms.set_constraint(constraint)
            self.atoms.calc = EMT()
            dyn = Langevin(self.atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory=save_path_md,
                           logfile='MD.log')
            dyn.run(self.steps)

            '''------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------'''

        elif self.action_idx == 5:  # 表面上O,CO,PdCO的扩散，原子或者分子行为
            self.atoms, selected_mol_index, action_done = self.to_diffuse_mol(self.atoms, self.total_s)

        elif self.action_idx == 6:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
            self.atoms, selected_mol_index, action_done = self.to_drill(self.atoms)
        
        elif self.action_idx == 7:  # OO龟裂，CO龟裂+异裂
            self.atoms, selected_mol_index, action_done = self.mol_dissociation(self.atoms)

        elif self.action_idx == 8:  #CO、O2、CO2的脱附
            # _,  desorblist, selected_mol_index = self.to_desorb_adsorbate(self.atoms)
            # if desorblist:
            self.atoms, selected_mol_index, action_done, desorb_info = self.choose_ads_to_desorb(self.atoms)
            if selected_mol_index == 2:
                self.reaction_occuration = True
                reward += self.reaction_occuration_reward   
        # elif self.action_idx == 9:
        # self.atoms, selected_mol_index, action_done = self.reaction(self.atoms)     
        else:
            print('No such action')

        self.timestep += 1
        if not action_done:
            reward -= 5
        
        self.to_constraint(self.atoms)
   
        # 优化该state的末态结构以及next_state的初态结构
        self.atoms, current_energy, current_force = self.lasp_calc(self.atoms)  # 优化分子结构并且计算体系能量与力
        
        # inspect which part problem
        current_energy_tmp = current_energy
        # print('To inspect problem in potential or my_python_env current_energy_tmp = ', current_energy_tmp)

        # 获得氧气分子和一氧化碳分子的能量
        self.E_OO_tmp = self.add_mole(self.initial_state, 'OO', 1.21)
        self.E_CO_tmp = self.add_mole(self.initial_state, 'CO', 1.13)
        self.E_OCO_tmp = self.add_mole(self.initial_state, 'OCO', 1.16)

        print(f'delta E_OO = {self.E_OO_tmp - self.E_OO}, delta E_CO = {self.E_CO_tmp - self.E_CO}, delta_E_CO2 = {self.E_OCO_tmp - self.E_OCO}')


        current_energy = current_energy + self.E_CO * self.n_CO + self.E_OO * self.n_OO + self.E_OCO * self.n_OCO

        self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint = self.get_surf_sites(
            self.atoms)
        
        # get_penalty_items
        previous_atom = self.trajectories[-1]

        # kickout the structure if too similar
        '''if self.RMSD(self.atoms, previous_atom)[0] and (current_energy - previous_energy) > 0:
            self.atoms = previous_atom
            current_energy = previous_energy
            RMSD_similar = True
            
        if self.timestep > 3:
            if self.RMSD(self.atoms, self.trajectories[-2])[0] and (current_energy - self.history['energies'][-2]) > 0:
                self.atoms = previous_atom
                current_energy = previous_energy
                RMSD_similar = True
                reward -= 1'''

        if self.timestep > 11:
            if self.RMSD(self.atoms, self.trajectories[-10])[0] and (current_energy - self.history['energies'][-10]) > 0: 
                self.atoms = previous_atom
                current_energy = previous_energy
                RMSD_similar = True
                reward -= 1
        
        if RMSD_similar:
            kickout = True

        '''if self.to_get_bond_info(self.atoms):   # 如果结构过差，将结构kickout
            self.atoms = previous_atom
            current_energy = previous_energy
            kickout = True
            # current_force = self.history['forces'][-1]
            reward += -5
'''
        self.free_atoms = []
        for i in range(len(self.atoms)):
            if i not in constraint.index:
                self.free_atoms.append(i)
        self.len_atom = len(self.free_atoms)

        self.pd = nn.ZeroPad2d(padding = (0,0,0,250-len(self.atoms.get_positions())))

        if self.action_idx == 0:
            current_energy = current_energy - self.delta_s
            self.adsorb_history['traj'] = self.adsorb_history['traj'] + [self.atoms.copy()]
            self.adsorb_history['structure'] = self.adsorb_history['structure'] + [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions()))).flatten()]
            self.adsorb_history['energy'] = self.adsorb_history['energy'] + [current_energy - previous_energy]
            self.adsorb_history['timesteps'].append(self.history['timesteps'][-1] + 1)

        if self.action_idx == 5:
            current_energy = current_energy + self.delta_s

        relative_energy = current_energy - previous_energy
        if relative_energy > 5:
            reward += -1
        else:
            # reward += math.tanh(-relative_energy/(self.H * 8.314 * self.temperature_K)) * (math.pow(10.0, 5))
            reward += self.get_reward_sigmoid(relative_energy)
        
        if relative_energy >= 0:
            reward -= 0.5

        print("action_done = ", action_done)
        if abs(current_energy - previous_energy) > 10.0:
            print("while {}, the number of molecules is:(n_OO, n_CO, n_OCO)".format(ACTION_SPACES[action]), self.n_OO, self.n_CO, self.n_OCO)
            print("And the previous number of molecules is: (previous_OO, previous_CO, previous_OCO)", previous_n_OO, 
                      previous_n_CO, previous_n_OCO)
            print("The energy change is:", current_energy - previous_energy)
            # print("action_done = ", action_done)
            

        # update prob list
        if self.action_idx == 0:
            self.prob_ads_list = self.update_eprob_list(self.prob_ads_list, selected_mol_index, relative_energy)
            
        elif self.action_idx == 5:
            self.prob_diffuse_list = self.update_eprob_list(self.prob_diffuse_list, selected_mol_index, relative_energy)
        elif self.action_idx == 6:
            self.prob_drill_list = self.update_eprob_list(self.prob_drill_list, selected_mol_index, relative_energy)
        elif self.action_idx == 7:
            self.prob_dissociation_list = self.update_eprob_list(self.prob_dissociation_list, selected_mol_index, relative_energy)
        elif self.action_idx == 8:
            self.prob_desorb_list = self.update_eprob_list(self.prob_desorb_list, selected_mol_index, relative_energy)

        # elif self.action_idx == 9:
        # self.prob_reaction_list_1 = self.update_eprob_list(self.prob_reaction_list_1, selected_mol_index[0], relative_energy)
        # self.prob_reaction_list_2 = self.update_eprob_list(self.prob_reaciton_list_2, selected_mol_index[1] ,relative_energy)

        # store the RMSD every step    
        self.RMSD_list.append(self.RMSD(self.atoms, previous_atom)[1])

        if self.timestep > 6:
            current_action_list = self.history['actions'][-5:]
            result = all(x == current_action_list[0] for x in current_action_list)
            if result and self.action_idx == current_action_list[0] and (RMSD_similar and relative_energy >= 0):
                self.repeat_action += 1
                reward -= self.repeat_action * 1
            elif result and self.action_idx != current_action_list[0]:
                self.repeat_action = 0

        current_structure = self.atoms.get_positions()

        self.energy = current_energy
        self.force = current_force

        observation = self.get_obs()  # 能观察到该state的结构与能量信息

        self.state = self.atoms, current_structure, current_energy

        # Update the history for the rendering

        self.history, self.trajectories = self.update_history(self.action_idx, kickout)

        if self.action_idx in [2,3,4]: 
            barrier = self.check_TS(previous_atom, self.atoms, previous_energy, current_energy, self.action_idx)    # according to Boltzmann probablity distribution
            if barrier > 5:
                reward += -5.0 / (self.H * self.k * self.temperature_K)
                barrier = 5.0
            else:
                # reward += math.tanh(-relative_energy /(self.H * 8.314 * self.temperature_K)) * (math.pow(10.0, 5))
                reward += -barrier/ (self.H * self.k * self.temperature_K)

        env_metal_list = []
        env_list = self.label_atoms(self.atoms, [43.33, 45.83])
        for i in self.atoms:    #查找是否Pd原子游离在环境中
            if i.index in env_list and i.symbol == self.metal_ele:
                env_metal_list.append(i.index)
        
        exist_too_short_bonds = self.exist_too_short_bonds(self.atoms)

        if exist_too_short_bonds or env_metal_list or self.energy - self.initial_energy > self.len_atom * self.max_energy_profile  or relative_energy > self.max_RE or self.len_atom > self.max_observation_atoms - 1:
            # reward += self.get_reward_sigmoid(1) * (self.timesteps - self.history['timesteps'][-1])
            reward -= 0.5 * (self.timesteps - self.timestep)
            self.done = True
        
        elif self.timestep > 11:
            if self.atoms == self.trajectories[-10]:
                self.done = True
                reward -= 0.5 * self.timesteps
                
        if -2.0 * relative_energy > self.max_RE:
            self.max_RE = -2.0 * relative_energy
            
        if len(self.history['actions']) - 1 >= self.total_steps:    # 当步数大于时间步，停止，且防止agent一直选取扩散或者平动动作
            self.done = True

        # _,  exist = self.to_ads_adsorbate(self.atoms)
        if len(self.history['real_energies']) > 11:
            RMSE = self.RMSE(self.history['real_energies'][-10:])
            if RMSE < 1.0:
                done_similar = True

        if (((current_energy - self.initial_energy) <= -0.95 * self.H and (current_energy - self.initial_energy) >= -1.1 * self.H) and (abs(current_energy - previous_energy) < self.min_RE_d and abs(current_energy - previous_energy) > 0.0001)) or (((current_energy - self.initial_energy) <= -0.9 * self.H and (current_energy - self.initial_energy) >= -1.2 * self.H) and done_similar):   # 当氧气全部被吸附到Pd表面，且两次相隔的能量差小于一定阈值，达到终止条件
        # if abs(current_energy - previous_energy) < self.min_RE_d and abs(current_energy - previous_energy) > 0.001:    
            self.done = True
            reward -= (self.energy - self.initial_energy)/(self.H * self.k * self.temperature_K)
            # self.min_RE_d = abs(current_energy - previous_energy)
        
        self.history['reward'] = self.history['reward'] + [reward]
        self.episode_reward += reward
        
        if self.episode_reward <= self.reward_threshold:   # 设置惩罚下限
            self.done = True

        if self.done:
            episode_over = True
            self.episode += 1
            if self.episode % self.save_every or self.done == 0:
                self.save_episode()
                self.plot_episode()

        
        return observation, reward, episode_over, [action_done]


    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d.npz' % self.episode)
        # traj = self.trajectories,
        # adsorb_traj=self.adsorb_history['traj'],
        np.savez_compressed(
            save_path,
            cell = self.cell,
            
            initial_energy=self.initial_energy,
            energies=self.history['energies'],
            actions=self.history['actions'],
            structures=self.history['structures'],  # 1
            timesteps=self.history['timesteps'],    
            forces = self.history['forces'],    # 2
            reward = self.history['reward'],

            adsorb_structure=self.adsorb_history['structure'],  # 3
            adsorb_energy=self.adsorb_history['energy'],
            adsorb_timesteps = self.adsorb_history['timesteps'],

            ts_energy = self.TS['energies'],
            ts_timesteps = self.TS['timesteps'],

            episode_reward = self.episode_reward,
            element_list = self.history['index'],

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

        plt.scatter(self.TS['timesteps'], self.TS['energies'], label='TS', marker='x', color='g', s=180)
        plt.scatter(self.adsorb_history['timesteps'], self.adsorb_history['energy'], label='ADS', marker='p', color='black', s=180)
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        return plt.close('all')

    def reset(self):
        if os.path.exists('input.arc'):
            os.remove('input.arc')
        if os.path.exists('all.arc'):
            os.remove('all.arc')
        if os.path.exists('sella.log'):
            os.remove('sella.log')

        self.n_CO = 2000
        self.n_OCO = 0
        self.n_OO = 4000    # need to discuss in the situation of oxide-metal

        self.atoms = self.generate_initial_slab_clean_metal_surface()

        self.to_constraint(self.atoms)
        self.atoms, self.initial_energy, self.initial_force= self.lasp_calc(self.atoms)
        self.initial_energy = self.initial_energy + self.E_CO * self.n_CO + self.E_OO * self.n_OO + self.E_OCO * self.n_OCO
        self.cell = self.atoms.get_cell()

        self.layer_atom, self.surf_atom, self.sub_atom, self.deep_atom, constraint, self.layerList, self.surfList, self.subList, self.deepList, self.bottomList = self.get_layer_info(self.atoms)
        # self.top_s, self.bridge_s, self.hollow_s, self.total_s, self.surf_O_s, self.surf_Pd_s, constraint= self.get_surf_sites(self.atoms)
        self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint= self.get_surf_sites(self.atoms)

        self.action_idx = 0
        self.episode_reward = 0.5 * self.timesteps
        self.timestep = 0

        self.total_steps = self.timesteps
        self.max_RE = 3
        self.min_RE_d = self.convergence * self.len_atom
        self.repeat_action = 0

        self.reaction_occuration = False
        self.reaction_occuration_reward = 10

        # maybe we can initialize the prob list according to the principle of kMC for seperate situation?

        self.prob_ads_list = [0.50, 0.50, 0.0]
        self.prob_diffuse_list = [0.55, 0.40, 0.05, 0.0]  # self.diffuse_list = ['O', 'CO', '[Pd]#CO']
        self.prob_dissociation_list = [0.50, 0.50, 0.0]  # self.dissociation_list = ['OO', 'CO']
        self.prob_desorb_list = [0.20, 0.20, 0.60, 0.0]    # self.desorb_list = ['OO', 'h-CO', 'tr-CO']   
        self.prob_drill_list = [0.40, 0.10, 0.40, 0.10, 0.0]
        # self.prob_reaction_list_1 = [0, 0]
        # self.prob_reaction_list_2 = [0, 0, 0]

        self.num_mole_adsorb_list = [self.n_CO, self.n_OO, 0]
        self.num_mole_diffuse_list = [0, 0, 0, 0]
        self.num_mole_dissociation_list = [0, 0, 0]
        self.num_mole_desorb_list = [0, 0, 0, 0]
        self.num_mole_drill_list = [0, 0, 0, 0, 0]
        # self.num_reaction_list_1 = [0, 0] 
        # self.num_reaction_list_2 = [0, 0, 0]

        self.atoms, _, _ = self.choose_ads_site(self.atoms, self.total_s)

        self.to_constraint(self.atoms)
        self.atoms, self.energy, self.force= self.lasp_calc(self.atoms)
        self.energy = self.energy + self.E_CO * self.n_CO + self.E_OO * self.n_OO + self.E_OCO * self.n_OCO

        self.trajectories = []
        self.RMSD_list = []
        self.trajectories.append(self.atoms.copy())

        self.TS = {}
        # self.TS['structures'] = [slab.get_scaled_positions()[self.free_atoms, :]]
        self.TS['energies'] = [0.0]
        self.TS['timesteps'] = [0]

        self.pd = nn.ZeroPad2d(padding = (0,0,0,250-len(self.atoms.get_positions())))

        self.adsorb_history = {}
        self.adsorb_history['traj'] = [self.atoms]
        self.adsorb_history['structure'] = [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions()))).flatten()]
        self.adsorb_history['energy'] = [0.0]
        self.adsorb_history['timesteps'] = [0]

        results = ['energies', 'actions', 'structures', 'timesteps', 'forces', 'scaled_structures', 'real_energies', 'reward', 'index']
        for item in results:
            self.history[item] = []
        self.history['energies'] = [0.0] 
        self.history['real_energies'] = [0.0] 
        self.history['actions'] = [0]
        self.history['forces'] = [np.array(self.pd(torch.tensor(self.force)))]
        self.history['structures'] = [np.array(self.pd(torch.tensor(self.atoms.get_positions()))).flatten()]
        self.history['scaled_structures'] = [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions()))).flatten()]
        self.history['timesteps'] = [0]
        self.history['reward'] = []
        self.history['index'] = [self.to_pad_the_array(np.array(self.get_atoms_symbol_list(self.atoms)), 
                                                       self.max_observation_atoms, position = False)]

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

            if len(self.TS['timesteps']) > 0:
                ax2.plot(self.TS['timesteps'],
                         self.TS['energies'], 'o', color='g')

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
                shape=(250, ),
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
                shape=(250, ),
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
        if self.use_GNN_description:
            observation['structure_scalar'], observation['structure_vector'] = self._use_Painn_description(self.atoms)
            observation['energy'] = np.array([self.energy - self.initial_energy]).reshape(1, )
            return observation['structure_scalar'], observation['structure_vector']
            # return observation['energy']
        else:
        # observation['structure'] = self.to_pad_the_array(self.atoms.get_scaled_positions()[self.free_atoms, :], self.max_observation_atoms).flatten()
            observation['structure'] = self.to_pad_the_array(self.atoms.get_scaled_positions(), self.max_observation_atoms).flatten()
            observation['energy'] = np.array([self.energy - self.initial_energy]).reshape(1, )
            # observation['force'] = self.force[self.free_atoms, :].flatten()
            # return observation['structure']
            return observation['structure']

    def update_history(self, action_idx, kickout):
        self.trajectories.append(self.atoms.copy())
        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.energy - self.initial_energy]
        self.history['forces'] = self.history['forces'] + [np.array(self.pd(torch.tensor(self.force)))]
        self.history['actions'] = self.history['actions'] + [action_idx]
        self.history['structures'] = self.history['structures'] + [np.array(self.pd(torch.tensor(self.atoms.get_positions()))).flatten()]
        # [np.array(self.pd(torch.tensor(self.atoms.get_positions()))).flatten()]
        self.history['scaled_structures'] = self.history['scaled_structures'] + [np.array(self.pd(torch.tensor(self.atoms.get_scaled_positions()))).flatten()]
        self.history['index'] = self.history['index'] + [self.to_pad_the_array(np.array(self.get_atoms_symbol_list(self.atoms)), 
                                                                               self.max_observation_atoms, position = False)]
        if not kickout:
            self.history['real_energies'] = self.history['real_energies'] + [self.energy - self.initial_energy]

        return self.history, self.trajectories

    def transition_state_search(self, previous_atom, current_atom, previous_energy, current_energy, action):
        if self.use_DESW:
            self.to_constraint(previous_atom)
            write_arc([previous_atom])

            write_arc([previous_atom, current_atom])
            previous_atom.calc = LASP(task='TS', pot='PdO', potential='NN D3')

            if previous_atom.get_potential_energy() == 0:  #没有搜索到过渡态
                ts_energy = previous_energy
            else:
                # ts_atom = read_arc('TSstr.arc',index = -1)
                barrier, ts_energy = previous_atom.get_potential_energy()
            # barrier = ts_energy - previous_energy

        else:
            if action == 1:
                if current_energy - previous_energy < -1.0:
                    barrier = 0
                elif current_energy - previous_energy >= -1.0 and current_energy - previous_energy <= 1.0:
                    barrier = np.random.normal(2, 2/3)
                else:
                    barrier = 4.0

            if action == 2 or action == 3:
                barrier = math.log(1 + pow(math.e, current_energy-previous_energy), 10)
            if action == 5:
                barrier = math.log(0.5 + 1.5 * pow(math.e, 2 *(current_energy - previous_energy)), 10)
            elif action == 6:
                barrier = 0.93 * pow(math.e, 0.615 * (current_energy - previous_energy)) - 0.16
            elif action == 7:
                barrier = 0.65 + 0.84 * (current_energy - previous_energy)
            else:
                barrier = 1.5
            ts_energy = previous_energy + barrier

        return barrier, ts_energy

    def check_TS(self, previous_atom, current_atom, previous_energy, current_energy, action):
        barrier, ts_energy = self.transition_state_search(previous_atom, current_atom, previous_energy, current_energy, action)

        self.record_TS(ts_energy)
        # pro = math.exp(-barrier / self.k * self.temperature_K)

        return barrier

    def record_TS(self, ts_energy):
        self.TS['energies'].append(ts_energy - self.initial_energy)
        self.TS['timesteps'].append(self.history['timesteps'][-1] + 1)
        return

    def choose_ads_site(self, state, surf_sites):
        action_done = True
        new_state = state.copy()

        add_total_sites = []
        layer_mol = []

        # self.num_mole_adsorb_list = [self.n_CO, self.n_OO, 0]
        prob_list = self.get_prob_list(self.prob_ads_list, self.num_mole_adsorb_list)

        prob = np.random.rand()
        ads_index = -1

        for i in self.layerList:
            if state[i].symbol == 'C' or state[i].symbol == 'O':
                layer_mol.append(i)

        for ads_sites in surf_sites:
            to_other_mol_distance = []
            if layer_mol:
                for i in layer_mol:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, state.get_positions()[i][0],
                                           state.get_positions()[i][1], state.get_positions()[i][2])
                    to_other_mol_distance.append(distance)
                if min(to_other_mol_distance) > 1.5 * d_O_C:
                    ads_sites[4] = 1
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                add_total_sites.append(ads_sites)
        
        if bool(sum(prob_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    ads_index = i
            to_ads_mole = self.ads_list[ads_index]  # select a target type of mole to desorb['CO', 'OO', 'OCO']
            ele_list = self.split_element(to_ads_mole)

            if add_total_sites:
                ads_site = add_total_sites[np.random.randint(len(add_total_sites))]
                new_state = state.copy()
                if len(ele_list):
                    if len(ele_list) == 2:
                        ele_1 = Atom(ele_list[0], (ads_site[0], ads_site[1], ads_site[2] + 1.3))
                        ele_2 = Atom(ele_list[1], (ads_site[0], ads_site[1], ads_site[2] + 2.51))
                        new_state = new_state + ele_1
                        new_state = new_state + ele_2

                # print('ads_index = ', ads_index)
                
                if ads_index == 0:
                    self.n_CO -= 1
                elif ads_index == 1:
                    self.n_OO -= 1 
                else:
                    action_done = False

                print("ads index is:", ads_index)
                print("whether action done:", action_done)
                print("The number of CO and O2 is {}, {}", self.n_CO, self.n_OO)
                
            else:
                action_done = False
        else:
            action_done = False

        print("ADS action_done:" , action_done)
        
        return new_state, ads_index, action_done
    
    def choose_ads_to_desorb(self, state):
        new_state = state.copy()

        # add_total_sites = []
        layer_mol = []
        mol_position = []
        desorblist = []

        action_done = True
        done = None

        selected_mol_index = []

        layerList = self.label_atoms(new_state, [surf_z, layer_z + fluct_d_layer])
        self.num_mole_desorb_list, mol_info_list = self.update_molecule_list(new_state, self.desorb_list, self.num_mole_desorb_list, identify_single_atom = True)
        prob_list = self.get_prob_list(self.prob_desorb_list, self.num_mole_desorb_list)
        prob = np.random.rand()
        desorb_index = -1
        if bool(sum(prob_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    desorb_index = i
            to_desorb_mole = self.desorb_list[desorb_index]  # select a target type of mole to desorb['CO', 'OO', 'OCO']
            ele_list = self.split_element(to_desorb_mole)

            if len(ele_list) == 1:
                if mol_info_list[0]:
                    selected_mol_index = mol_info_list[0][np.random.randint(len(mol_info_list[0]))]
                else:
                    action_done = False
                    # print('no len(mol) == 1')
            elif len(ele_list) == 2:
                if mol_info_list[1]:
                    selected_mol_index = mol_info_list[1][np.random.randint(len(mol_info_list[1]))]
                else:
                    action_done = False  
                    # print('no len(mol) == 2')
            elif len(ele_list) == 3:
                if mol_info_list[2]:
                    selected_mol_index = mol_info_list[2][np.random.randint(len(mol_info_list[2]))]
                else:
                    action_done = False
                    # print('no len(mol) == 3')

            for i in layerList:
                if state[i].symbol == 'O' or state[i].symbol == 'C':
                    layer_mol.append(i)
            for i in selected_mol_index:
                desorblist.append(i)

            if desorblist and action_done:
                del new_state[[i for i in range(len(new_state)) if i in desorblist]]

                # print('desorb_index:', desorb_index)
                
                if desorb_index == 0:
                    self.n_CO += 1
                elif desorb_index == 1:
                    self.n_OO += 1
                elif desorb_index == 2:
                    self.n_OCO += 1
                else:
                    action_done = False
            else:
                action_done = False
        else:
            action_done = False

        print("Desportion action_done:" , action_done)
            
        return new_state, desorb_index, action_done, [mol_info_list, prob_list, desorblist, selected_mol_index]
    
    def rotation(self, zeta):
        initial_state = self.atoms.copy()
        zeta = math.pi * zeta / 180
        central_point = np.array([initial_state.cell[0][0] / 2, initial_state.cell[1][1] / 2, 0])
        matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
        matrix = np.array(matrix)

        for atom in initial_state.positions:
            if surf_z < atom[2] < 15.0 :
                atom += np.array(
                        (np.dot(matrix, (np.array(atom.tolist()) - central_point).T).T + central_point).tolist()) - atom
        self.atoms.positions = initial_state.get_positions()

    def translation(self):
        initial_positions = self.atoms.positions

        for atom in initial_positions:
            if atom in self.deep_atom:
                atom += self.lamada_d * self.muti_movement
            if atom in self.sub_atom:
                atom += self.lamada_s * self.muti_movement
            if atom in self.surf_atom:
                atom += self.lamada_layer * self.muti_movement
            if atom in self.layer_atom:
                atom += self.lamada_layer * self.muti_movement
        self.atoms.positions = initial_positions

    def get_surf_sites(self, slab):
        state = slab.copy()

        layerList = self.label_atoms(state, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        surfList = self.label_atoms(state, [surf_z - fluct_d_metal, surf_z + fluct_d_metal * 2])
        '''for i in surfList:
            if state[i].symbol == 'O':
                surfList.remove(i)'''

        subList = self.label_atoms(state, [sub_z - fluct_d_metal, sub_z + fluct_d_metal])
        deepList = self.label_atoms(state, [deep_z - fluct_d_metal, deep_z + fluct_d_metal])
        bottomList = self.label_atoms(state, [bottom_z - fluct_d_metal, bottom_z + fluct_d_metal])

        layer = state.copy()
        del layer[[i for i in range(len(layer)) if i not in layerList]]
        layer_atom = layer.get_positions()

        surf = state.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]
        surf_atom = surf.get_positions()
        atop = surf.get_positions()

        sub_layer = state.copy()
        del sub_layer[[i for i in range(len(sub_layer)) if i not in subList]]
        sub_atom = sub_layer.get_positions()

        deep_layer = state.copy()
        del deep_layer[[i for i in range(len(deep_layer)) if i not in deepList]]
        deep_atom = deep_layer.get_positions()

        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        fix = state.set_constraint(constraint)

        top_sites, bridge_sites, hollow_sites, total_surf_sites = self.get_sites(surf)

        surf_O = state.copy()
        del surf_O[[i for i in range(len(surf_O)) if i not in surfList or surf_O.symbols[i] != 'O']]
        surf_metal_ele = state.copy()
        del surf_metal_ele[[i for i in range(len(surf_metal_ele)) if i not in surfList or surf_metal_ele.symbols[i] != self.metal_ele]]

        return top_sites, bridge_sites, hollow_sites, total_surf_sites, constraint

    def to_diffuse_mol(self, slab, surf_sites):
        layer_mol = []
        to_diffuse_mol_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        diffusable_sites = []
        interference_mol_distance = []
        diffusable = True
        selected_mol_index = None
        action_done = True

        # prob_list = [0.3, 0.6, 0.9, 1.0, 0.0]
        self.num_mole_diffuse_list, mol_info_list = self.update_molecule_list(slab, self.diffuse_list, self.num_mole_diffuse_list, identify_single_atom = True)
        prob_list = self.get_prob_list(self.prob_diffuse_list, self.num_mole_diffuse_list)
        prob = np.random.rand()
        diff_index = 0
        if bool(sum(prob_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    diff_index = i
            to_diffuse_mole = self.diffuse_list[diff_index]  # select a target type of mole to diffuse['O', 'CO', 'PdCO']
            ele_list = self.split_element(to_diffuse_mole)

            if len(ele_list) == 1:
                r_mol_1 = Eleradii[Eledict[ele_list[0]] - 1]
                if mol_info_list[0]:
                    selected_mol_index = mol_info_list[0][np.random.randint(len(mol_info_list[0]))]
                else:
                    action_done = False
                    # print('no len(mol) == 1')
            elif len(ele_list) == 2:
                r_mol_1 = Eleradii[Eledict['C'] - 1]
                if mol_info_list[1]:
                    selected_mol_index = mol_info_list[1][np.random.randint(len(mol_info_list[1]))]
                else:
                    action_done = False  
                    # print('no len(mol) == 2')
            elif len(ele_list) == 3:
                r_mol_1 = Eleradii[Eledict[self.metal_ele]- 1]
                if mol_info_list[2]:
                    selected_mol_index = mol_info_list[2][np.random.randint(len(mol_info_list[2]))]
                else:
                    action_done = False
                    # print('no len(mol) == 3')

            for i in slab:  # 寻找layer层所有的碳原子和氧原子
                if i.index in layer_List and (i.symbol == 'C' or i.symbol == 'O'):
                    layer_mol.append(i.index)
            
            for ads_sites in surf_sites:    # 寻找可以diffuse的位点
                to_other_mol_symbol_list = []
                to_other_mol_distance = []
                if layer_mol:
                    for i in layer_mol:
                        distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.5, slab.get_positions()[i][0],
                                            slab.get_positions()[i][1], slab.get_positions()[i][2])
                        to_other_mol_distance.append(distance)
                        to_other_mol_symbol_list.append(slab.symbols[i])
                    
                    r_mol_2 = Eleradii[Eledict[to_other_mol_symbol_list[to_other_mol_distance.index(min(to_other_mol_distance))]] - 1]
                    if min(to_other_mol_distance) > 1.5 * (r_mol_1 + r_mol_2):
                        ads_sites[4] = 1
                    else:
                        ads_sites[4] = 0
                else:
                    ads_sites[4] = 1
                if ads_sites[4]:
                    diffusable_sites.append(ads_sites)

            '''if layer_mol: # 防止选定的mole被trap住无法diffuse
                for i in layer_mol:
                    to_other_mol_distance = []
                    for j in layer_mol:
                        if j != i:
                            distance = self.distance(slab.get_positions()[i][0],
                                            slab.get_positions()[i][1], slab.get_positions()[i][2],slab.get_positions()[j][0],
                                            slab.get_positions()[j][1], slab.get_positions()[j][2])
                            to_other_mol_distance.append(distance)
                            
                    if self.to_get_min_distances(to_other_mol_distance,4):
                        d_min_4 = self.to_get_min_distances(to_other_mol_distance, 4)
                        if d_min_4 > 1.5:
                            to_diffuse_mol_list.append(i)
                    else:
                        to_diffuse_mol_list.append(i)

            if to_diffuse_mol_list:
                # selected_mol_index = layer_mol[np.random.randint(len(to_diffuse_mol_list))]
                if diffusable_sites:
                    diffuse_site = diffusable_sites[np.random.randint(len(diffusable_sites))]
                else:
                    if len(ele_list) == 1:
                        diffuse_site = slab.get_positions()[selected_mol_index]
                    else:
                        diffuse_site = slab.get_positions()[selected_mol_index[0]]
                if len(ele_list) == 1:
                    interference_O_list = [i for i in layer_mol if i != selected_mol_index]
                else:
                    interference_O_list = [i for i in layer_mol if i not in selected_mol_index]
                for j in interference_O_list:
                    if len(ele_list) == 1:
                        d = self.atom_to_traj_distance(slab.positions[selected_mol_index], diffuse_site, slab.positions[j])
                    else:
                        d = self.atom_to_traj_distance(slab.positions[selected_mol_index[0]], diffuse_site, slab.positions[j])
                    interference_mol_distance.append(d)
                if interference_mol_distance:
                    if min(interference_mol_distance) < 0.3 * (r_mol_1 + r_mol_2):
                        diffusable = False
            
                if diffusable:'''
            if diffusable_sites and action_done:
                diffuse_site = diffusable_sites[np.random.randint(len(diffusable_sites))]
                '''if len(ele_list) == 1:
                    del slab[[j for j in range(len(slab)) if j == selected_mol_index]]
                else:
                    del slab[[j for j in range(len(slab)) if j in selected_mol_index]]'''
                    # O = Atom('O', (diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.5))
                    # slab = slab + O
                for atom in slab:
                    if len(ele_list) == 1:
                        if atom.index == selected_mol_index:
                            atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.5])

                    elif len(ele_list) == 2:
                        if atom.index == selected_mol_index[0]:
                            atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.3])
                        elif atom.index == selected_mol_index[1]:
                            atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + 2.5])

                    elif len(ele_list) == 3:
                        if atom.index == selected_mol_index[0]:
                            atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.3])
                        elif atom.index == selected_mol_index[1]:
                            atom.position = np.array([diffuse_site[0] + 0.6, diffuse_site[1], diffuse_site[2] + 2.0])
                        elif atom.index == selected_mol_index[2]:
                            atom.position = np.array([diffuse_site[0] - 0.6, diffuse_site[1], diffuse_site[2] + 2.0])

            else:
                action_done = False
                # print('no diffuse_sites')
        else:
            action_done = False
        
        print("Diffusion action_done:" , action_done)

        return slab, diff_index, action_done

    def to_expand_lattice(self, slab, expand_layer, expand_surf, expand_lattice):

        slab.cell[0][0] = slab.cell[0][0] * expand_lattice
        slab.cell[1][1] = slab.cell[1][1] * expand_lattice

        mid_point_x = slab.cell[0][0] / 2
        mid_point_y = slab.cell[1][1] / 2

        def mid_point_z(List):
            sum = 0
            for i in slab:
                if i.index in List:
                    sum += slab.get_positions()[i.index][2]
            mid_point_z = sum / len(List)
            return mid_point_z
        
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        if layer_List:
            mid_point_layer = [mid_point_x, mid_point_y, mid_point_z(layer_List)]
        else:
            mid_point_layer = [mid_point_x, mid_point_y, mid_point_z(self.surfList)]
        mid_point_surf = [mid_point_x, mid_point_y, mid_point_z(self.surfList)]
        mid_point_sub = [mid_point_x, mid_point_y, mid_point_z(self.subList)]

        for i in slab:
            slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_layer[0]) * expand_lattice + \
                                             mid_point_layer[0]
            slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_layer[1]) * expand_lattice + \
                                             mid_point_layer[1]
            if i.index in self.layerList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_layer[0]) * expand_layer + \
                                             mid_point_layer[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_layer[1]) * expand_layer + \
                                             mid_point_layer[1]
            if i.index in self.surfList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_surf[0]) * expand_surf + \
                                             mid_point_surf[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_surf[1]) * expand_surf + \
                                             mid_point_surf[1]
            if i.index in self.subList:
                slab.positions[i.index][0] = (slab.get_positions()[i.index][0] - mid_point_sub[0]) * expand_lattice + \
                                             mid_point_sub[0]
                slab.positions[i.index][1] = (slab.get_positions()[i.index][1] - mid_point_sub[1]) * expand_lattice + \
                                             mid_point_sub[1]
        return slab
    
    def get_reward_trans(self, relative_energy):
        return -relative_energy / (self.H * self.k * self.temperature_K)

    def get_reward_tanh(self, relative_energy):
        reward = math.tanh(-relative_energy/(self.H * self.k * self.temperature_K))
        return reward
    
    def get_reward_sigmoid(self, relative_energy):
        return 2 * (0.5 - 1 / (1 + np.exp(-relative_energy/(self.H * self.k * self.temperature_K))))
    
    def to_drill(self, slab):
        action_done = True
        layer_C_atom_list = self.layer_ele_atom_list(slab, 'C')
        layer_O_atom_list = self.layer_ele_atom_list(slab, 'O')
        sub_C_atom_list = self.sub_ele_atom_list(slab, 'C')
        sub_O_atom_list = self.sub_ele_atom_list(slab, 'O')

        self.num_mole_drill_list = [len(layer_O_atom_list), len(sub_O_atom_list), len(layer_C_atom_list), len(sub_C_atom_list), 0]
        prob_list = self.get_prob_list(self.prob_drill_list, self.num_mole_drill_list)

        prob = np.random.rand()
        drill_index = 0
        if bool(sum(prob_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    drill_index = i
            to_drill_mole = self.drill_list[drill_index]  # select a target type of mole to desorb['CO', 'OO', 'OCO']
            ele_list = self.split_element(to_drill_mole)

            if drill_index == 0:
                selected_drill_ele_list = layer_O_atom_list
            elif drill_index == 1:
                selected_drill_ele_list = layer_C_atom_list
            elif drill_index == 2:
                selected_drill_ele_list = sub_O_atom_list
            elif drill_index == 3:
                selected_drill_ele_list = sub_C_atom_list

            if selected_drill_ele_list:
                selected_ele = selected_drill_ele_list[np.random.randint(len(selected_drill_ele_list))]

                # slab = self.to_expand_lattice(slab, 1.25, 1.25, 1.1)

                if selected_ele in layer_C_atom_list or selected_ele in layer_O_atom_list:
                    slab = self.to_drill_surf(slab, ele_list[0])
                elif selected_ele in sub_C_atom_list or selected_ele in sub_O_atom_list:
                    slab = self.to_drill_deep(slab, ele_list[0])

                # self.to_constraint(slab)
                # slab, _, _ = self.lasp_calc(slab)
                # slab = self.to_expand_lattice(slab, 0.8, 0.8, 10/11)
            else:
                action_done = False
        else:
            action_done = False

        print("Drill action_done:" , action_done)

        return slab, drill_index, action_done
    
    def to_drill_surf(self, slab, ele):
        layer_ele = []
        to_distance = []
        drillable_sites = []
        layer_ele_atom_list = []
        layer_eleObond_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])

        sub_sites = self.get_sub_sites(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == ele:
                layer_ele.append(i.index)
        
        for ads_sites in sub_sites:
            to_other_O_distance = []
            if layer_ele:
                for i in layer_ele:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_C:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)

        
        if layer_ele:
            ana = Analysis(slab)
            eleObonds = ana.get_bonds(ele,'O',unique = True)
            if eleObonds[0]:
                for i in eleObonds[0]:
                    if i[0] in layer_ele and i[1] in layer_ele:
                        layer_eleObond_list.append(i[0])
                        layer_eleObond_list.append(i[1])

            for j in layer_ele:
                if j not in layer_eleObond_list:
                    layer_ele_atom_list.append(j)

        if layer_ele_atom_list:
            selected_atom_index = layer_ele_atom_list[np.random.randint(len(layer_ele_atom_list))]
            position = slab.get_positions()[selected_atom_index]
            # del slab[[j for j in range(len(slab)) if j == selected_atom_index]]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))
        if to_distance:
            drill_site = drillable_sites[to_distance.index(min(to_distance))]
            # ele = Atom(ele, (drill_site[0], drill_site[1], drill_site[2] +1.3))
            # slab = slab + ele
            for i in range(len(slab)):
                if i == selected_atom_index:
                    slab.positions[i] = np.array([drill_site[0], drill_site[1], drill_site[2] +1.3])

            lifted_atoms_list = self.label_atoms(slab, [surf_z - fluct_d_metal/2, layer_z + fluct_d_layer])
            for j in lifted_atoms_list:
                slab.positions[j][2] += 1.0
        return slab
    
    def to_drill_deep(self, slab, ele):

        to_distance = []
        drillable_sites = []

        sub_ele_atom_list = self.sub_ele_atom_list(slab, ele)
        # layer_OObond_list = []
        sub_List = self.label_atoms(slab, [sub_z - 2.0, sub_z + 2.0])

        deep_sites = self.get_deep_sites(slab)

        '''for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)'''
        
        for ads_sites in deep_sites:
            to_other_C_distance = []
            if sub_ele_atom_list:
                for i in sub_ele_atom_list:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_C_distance.append(distance)
                if min(to_other_C_distance) > 2 * d_O_C:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)

        
        '''if sub_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_O and i[1] in sub_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in sub_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)'''

        if sub_ele_atom_list:
            i = sub_ele_atom_list[np.random.randint(len(sub_ele_atom_list))]
            position = slab.get_positions()[i]
            del slab[[j for j in range(len(slab)) if j == i]]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = drillable_sites[to_distance.index(min(to_distance))]
            ele = Atom(ele, (drill_site[0], drill_site[1], drill_site[2] +1.3))
            slab = slab + ele

            lifted_atoms_list = self.label_atoms(slab, [sub_z - 1.0, layer_z + fluct_d_layer])
            for j in lifted_atoms_list:
                slab.positions[j][2] += 1.0
        return slab

    def mol_dissociation(self, slab):
        # TODO: now modifying
        action_done = True
        ana = Analysis(slab)

        # mole in ['O-O', 'C-O', 'C-O']   后面的'CO'分别指CO的龟裂和异裂，即[C——>Pd, O——>Pd] or [C——>O, O——>Pd]
        # selected_mole_index in 0~2
        metal_ele_list_1 = []
        metal_ele_list_2 = []
        dissociate_mole_list = []

        layer_list = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        n_mol_list, _ = self.update_molecule_list(slab, self.dissociation_list, self.num_mole_dissociation_list, identify_single_atom= False)
        prob_list = self.get_prob_list(self.prob_dissociation_list, n_mol_list)

        prob = np.random.rand()
        dissociation_index = 0
        if bool(sum(n_mol_list)):    # 如果prob_list = [0.0, 0.0, 0.0, 0.0], 则直接停止操作，并且reward -= 5
            for i in range(len(prob_list) - 1):
                if prob_list[i - 1] < prob and prob < prob_list[i]:
                    dissociation_index = i
            to_dissociate_mole = self.dissociation_list[dissociation_index]  # select a target type of mole to diffuse['O', 'CO', 'PdCO']
            ele_list = self.split_element(to_dissociate_mole)

            ele_1 = ele_list[0]
            ele_2 = ele_list[1]

            metal_ele_Bonds_1 = ana.get_bonds(self.metal_ele, ele_1, unique = True) # self.metal_ele = Pd, Ni, ....
            metal_ele_Bonds_2 = ana.get_bonds(self.metal_ele, ele_2, unique = True)
            eleBonds_12 = ana.get_bonds(ele_1, ele_2, unique = True)

            if metal_ele_Bonds_1[0]:
                for i in metal_ele_Bonds_1[0]:
                    metal_ele_list_1.append(i[0])
                    metal_ele_list_1.append(i[1])
            if metal_ele_Bonds_2[0]:
                for i in metal_ele_Bonds_2[0]:
                    metal_ele_list_2.append(i[0])
                    metal_ele_list_2.append(i[1])
            
            if eleBonds_12[0]:
                for i in eleBonds_12[0]:
                    if (i[0] in metal_ele_list_1 or i[0] in metal_ele_list_2 or i[1] in metal_ele_list_1 or i[1] in metal_ele_list_2) and (i[0] in layer_list or i[1] in layer_list):
                        dissociate_mole_list.append([(i[0], i[1])])

            if dissociate_mole_list:
                selected_mol = dissociate_mole_list[np.random.randint(len(dissociate_mole_list))]
                # d = ana.get_values([OO])[0]
                zeta = self.get_angle_with_z(slab, selected_mol) * 180/ math.pi -5
                fi = 0
                slab = self.oxy_rotation(slab, selected_mol, zeta, fi)
                slab = self.to_dissociate(slab, selected_mol, dissociation_index)
            else:
                action_done = False
                # print("no dissociate_mole_list")
        else:
            action_done = False
            # print("no sum(n_mol_list)")

        print("Dissociation action_done:" , action_done)

        return slab, dissociation_index, action_done

    def label_atoms(self, atoms, zRange):
        myPos = atoms.get_positions()
        return [
            i for i in range(len(atoms)) \
            if min(zRange) < myPos[i][2] < max(zRange)
        ]

    def distance(self, x1, y1, z1, x2, y2, z2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
        return dis

    def to_generate_initial_slab(self):
        slab = fcc100(self.metal_ele, size=(6, 6, 4), vacuum=10.0)
        delList = [77, 83, 89, 95, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 119, 120, 125,
                   126, 131, 132, 137, 138, 139, 140, 141, 142, 143]
        del slab[[i for i in range(len(slab)) if i in delList]]
        return slab

    def get_layer_info(self, slab): # get layer Pd
        state = slab.copy()

        layerList = self.label_atoms(state, [surf_z, layer_z + fluct_d_layer])
        surfList = self.label_atoms(state, [surf_z - fluct_d_metal, surf_z + fluct_d_metal*2])
        '''for i in surfList:
            if state[i].symbol == 'O':
                surfList.remove(i)'''
        subList = self.label_atoms(state, [sub_z - fluct_d_metal, sub_z + fluct_d_metal])
        deepList = self.label_atoms(state, [deep_z - fluct_d_metal, deep_z + fluct_d_metal])
        bottomList = self.label_atoms(state, [bottom_z - fluct_d_metal, bottom_z + fluct_d_metal])

        layer = state.copy()
        del layer[[i for i in range(len(layer)) if i not in layerList]]
        layer_atom = layer.get_positions()

        surf = state.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]
        surf_atom = surf.get_positions()

        sub_layer = state.copy()
        del sub_layer[[i for i in range(len(sub_layer)) if i not in subList]]
        sub_atom = sub_layer.get_positions()

        deep_layer = state.copy()
        del deep_layer[[i for i in range(len(deep_layer)) if i not in deepList]]
        deep_atom = deep_layer.get_positions()

        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        fix = state.set_constraint(constraint)

        return layer_atom, surf_atom, sub_atom, deep_atom, constraint, layerList, surfList, subList, deepList, bottomList
    
    def RMSD(self, current_atoms, previous_atoms):
        similar = False
        top_sites, bridge_sites, hollow_sites, total_surf_sites, constraint_p = self.get_surf_sites(previous_atoms)
        
        free_atoms_p = []
        for i in range(len(previous_atoms)):
            if i not in constraint_p.index:
                free_atoms_p.append(i)
        len_atom_p = len(free_atoms_p)

        top_sites, bridge_sites, hollow_sites, total_surf_sites, constraint_c = self.get_surf_sites(current_atoms)
        free_atoms_c = []
        for i in range(len(current_atoms)):
            if i not in constraint_c.index:
                free_atoms_c.append(i)
        len_atom_c = len(free_atoms_c)

        RMSD = 0
        cell_x = current_atoms.cell[0][0]
        cell_y = current_atoms.cell[1][1]
        if len_atom_p == len_atom_c:
            for i in free_atoms_p:
                d = self.distance(previous_atoms.positions[i][0], previous_atoms.positions[i][1], previous_atoms.positions[i][2],
                                  current_atoms.positions[i][0], current_atoms.positions[i][1], current_atoms.positions[i][2])
                if d > max(cell_x, cell_y) / 2:
                    d = self._get_pbc_min_dis(previous_atoms, current_atoms, i)
                    
                RMSD += d * d
            RMSD = math.sqrt(RMSD / len_atom_p)
            if RMSD <= 0.5:
                similar = True

        return [similar, RMSD]

    def get_sub_sites(self, slab):
        layer_atom, surf_atom, sub_atom, deep_atom, fix, _, _, _, _, _ = self.get_layer_info(slab)
        # layer_atom, surf_atom, sub_atom, deep_atom, constraint, layerList, surfList, subList, deepList, bottomList

        atop = sub_atom
        pos_ext = sub_atom
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
        total_sub_sites = []

        for i in top_sites:
            sites_1.append(np.transpose(np.append(i, 1)))
        for i in bridge_sites:
            sites_1.append(np.transpose(np.append(i, 2)))
        for i in hollow_sites:
            sites_1.append(np.transpose(np.append(i, 3)))
        for i in sites_1:
            total_sub_sites.append(np.append(i, 0))

        total_sub_sites = np.array(total_sub_sites)

        return total_sub_sites
    
    def get_deep_sites(self, slab):
        layer_atom, surf_atom, sub_atom, deep_atom, fix, _, _, _, _, _ = self.get_layer_info(slab)

        atop = deep_atom
        pos_ext = deep_atom
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
        total_deep_sites = []

        for i in top_sites:
            sites_1.append(np.transpose(np.append(i, 1)))
        for i in bridge_sites:
            sites_1.append(np.transpose(np.append(i, 2)))
        for i in hollow_sites:
            sites_1.append(np.transpose(np.append(i, 3)))
        for i in sites_1:
            total_deep_sites.append(np.append(i, 0))

        total_deep_sites = np.array(total_deep_sites)

        return total_deep_sites

    def lasp_calc(self, atom):
        write_arc([atom])
        atom.calc = LASP(task='local-opt', pot=self.pot, potential='NN D3')
        energy = atom.get_potential_energy()
        force = atom.get_forces()
        atom = read_arc('allstr.arc', index = -1)
        return atom, energy, force
    
    def lasp_single_calc(self, atom):
        write_arc([atom])
        atom.calc = LASP(task='single-energy', pot=self.pot, potential='NN D3')
        energy = atom.get_potential_energy()
        force = atom.get_forces()
        atom = read_arc('allstr.arc', index = -1)
        return energy
    
    def to_constraint(self, slab):
        bottomList = self.label_atoms(slab, [bottom_z - fluct_d_metal, bottom_z + fluct_d_metal])
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        fix = slab.set_constraint(constraint)

    def exist_too_short_bonds(self,slab):
        exist = False
        ana = Analysis(slab)
        ele_list = self.split_element(self.target_system[0])
        for ele_1 in ele_list:
            for ele_2 in ele_list:
                d = Eleradii[Eledict[ele_1] - 1] + Eleradii[Eledict[ele_2] - 1]
                ele_12_Bonds = ana.get_bonds(ele_1, ele_2, unique = True)
                if ele_12_Bonds[0]:
                    ele_12_BondValues = ana.get_values(ele_12_Bonds)[0]
                    min_ele_12_Bonds = min(ele_12_BondValues)
                    if min_ele_12_Bonds < 0.3 * d:
                        exist = True
        return exist
    
    def atom_to_traj_distance(self, atom_A, atom_B, atom_C):
        d_AB = self.distance(atom_A[0], atom_A[1], atom_A[2], atom_B[0], atom_B[1], atom_B[2])
        d = abs((atom_C[0]-atom_A[0])*(atom_A[0]-atom_B[0])+
                (atom_C[1]-atom_A[1])*(atom_A[1]-atom_B[1])+
                (atom_C[2]-atom_A[2])*(atom_A[2]-atom_B[2])) / d_AB
        return d

    def to_get_bond_info(self, slab):
        exist = False
        ana = Analysis(slab)
        ele_list = self.split_element(self.target_system[0])
        for ele_1 in ele_list:
            for ele_2 in ele_list:
                d = Eleradii[Eledict[ele_1] - 1] + Eleradii[Eledict[ele_2] - 1]
                ele_12_Bonds = ana.get_bonds(ele_1, ele_2, unique = True)
                if ele_12_Bonds[0]:
                    ele_12_BondValues = ana.get_values(ele_12_Bonds)[0]
                    min_ele_12_Bonds = min(ele_12_BondValues)
                    max_ele_12_Bonds = max(ele_12_BondValues)
                    if min_ele_12_Bonds < 0.80 * d or max_ele_12_Bonds > 1.15 * d:
                        exist = True
        return exist
        
    
    def ball_func(self,pos1, pos2, zeta, fi):	# zeta < 36, fi < 3
        d = self.distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
        zeta = -math.pi * zeta / 180
        fi = -math.pi * fi / 180
        '''如果pos1[2] > pos2[2],atom_1旋转下来'''
        pos2_position = pos2
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
    
    def to_dissociate(self, slab, atoms, dissociation_idx):
        expanding_index = 2.0
        central_point = np.array([(slab.get_positions()[atoms[0][0]][0] + slab.get_positions()[atoms[0][1]][0])/2, 
                                  (slab.get_positions()[atoms[0][0]][1] + slab.get_positions()[atoms[0][1]][1])/2, (slab.get_positions()[atoms[0][0]][2] + slab.get_positions()[atoms[0][1]][2])/2])
        slab.positions[atoms[0][0]] += np.array([expanding_index*(slab.get_positions()[atoms[0][0]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][2]-central_point[2])])
        slab.positions[atoms[0][1]] += np.array([expanding_index*(slab.get_positions()[atoms[0][1]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][2]-central_point[2])])
        addable_sites_1, addable_sites_2 = [], []
        layer_ele_1, layer_ele_2 = [], []
        layerlist = self.label_atoms(slab,[layer_z - fluct_d_layer, layer_z + fluct_d_layer])

        # ele_list = self.split_element(atoms)
        # ele_1, ele_2 = ele_list[0], ele_list[1]
        ele_1, ele_2 = slab[atoms[0][0]].symbol, slab[atoms[0][1]].symbol

        # if dissociation_idx != 2:
        for ads_site in self.total_s:
            for i in layerlist:
                if slab[i].symbol == ele_1:
                    layer_ele_1.append(i)
                if slab[i].symbol == ele_2:
                    layer_ele_2.append(i)
            to_other_mol_distance = []
            if layer_ele_1:
                for i in layer_ele_1:
                    to_distance = self.distance(ads_site[0], ads_site[1], ads_site[2] + 1.5, slab.get_positions()[i][0],
                                            slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_mol_distance.append(to_distance)
                if min(to_other_mol_distance) > 1.5 * d_O_O:
                    ads_site[4] = 1
            else:
                ads_site[4] = 1
            if ads_site[4]:
                addable_sites_1.append(ads_site)
                addable_sites_2.append(ads_site)

            '''elif dissociation_idx == 2:
            for ads_site_1 in self.surf_O_s:
                for i in layerlist:
                    if slab[i].symbol == ele_1:
                        layer_ele_1.append(i)
                    to_other_mol_distance = []
                if layer_ele_1:
                    for i in layer_ele_1:
                        to_distance = self.distance(ads_site_1[0], ads_site_1[1], ads_site_1[2] + 1.5, slab.get_positions()[i][0],
                                            slab.get_positions()[i][1], slab.get_positions()[i][2])
                        to_other_mol_distance.append(to_distance)
                    if min(to_other_mol_distance) > 1.5 * d_O_O:
                        ads_site[4] = 1
                else:
                    ads_site[4] = 1
                if ads_site[4]:
                    addable_sites_1.append(ads_site)

            for ads_site_2 in self.surf_O_Pd:
                for i in layerlist:
                    if slab[i].symbol == ele_1:
                        layer_ele_1.append(i)
                    to_other_mol_distance = []
                if layer_ele_1:
                    for i in layer_ele_1:
                        to_distance = self.distance(ads_site_2[0], ads_site_2[1], ads_site_2[2] + 1.5, slab.get_positions()[i][0],
                                            slab.get_positions()[i][1], slab.get_positions()[i][2])
                        to_other_mol_distance.append(to_distance)
                    if min(to_other_mol_distance) > 1.5 * d_O_O:
                        ads_site[4] = 1
                else:
                    ads_site[4] = 1
                if ads_site[4]:
                    addable_sites_2.append(ads_site)'''

        if addable_sites_1 and addable_sites_2:

            ele1_distance = []
            for add_1_site in addable_sites_1:
                distance_1 = self.distance(add_1_site[0], add_1_site[1], add_1_site[2] + 1.3, slab.get_positions()[atoms[0][0]][0],
                                            slab.get_positions()[atoms[0][0]][1], slab.get_positions()[atoms[0][0]][2])
                ele1_distance.append(distance_1)
            
            ele1_site = addable_sites_1[ele1_distance.index(min(ele1_distance))]
            
            ad_2_sites = []
            for add_site in addable_sites_2:
                d = self.distance(add_site[0], add_site[1], add_site[2] + 1.3, ele1_site[0], ele1_site[1], ele1_site[2])
                if d > 2.0 * d_O_C:
                    ad_2_sites.append(add_site)

            ele2_distance = []
            for add_2_site in ad_2_sites:
                distance_2 = self.distance(add_2_site[0], add_2_site[1], add_2_site[2] + 1.3, slab.get_positions()[atoms[0][1]][0],
                                            slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][1]][2])
                ele2_distance.append(distance_2)
            
            ele2_site = ad_2_sites[ele2_distance.index(min(ele2_distance))]

            '''del slab[[i for i in range(len(slab)) if slab[i].index == atoms[0][0] or slab[i].index == atoms[0][1]]]

            if ele1_site[0] == ele2_site[0] and ele1_site[1] == ele2_site[1]:
                ele_1 = Atom(ele_1, (ele1_site[0], ele1_site[1], ele1_site[2] + 1.3))
                ele_2 = Atom(ele_2, (ele1_site[0], ele1_site[1], ele1_site[2] + 2.51))
            else:

                ele_1 = Atom(ele_1, (ele1_site[0], ele1_site[1], ele1_site[2] + 1.3))
                ele_2 = Atom(ele_2, (ele2_site[0], ele2_site[1], ele2_site[2] + 1.3))

            slab = slab + ele_1
            slab = slab + ele_2'''
            for atom in slab:
                if ele1_site[0] == ele2_site[0] and ele1_site[1] == ele2_site[1]:
                    O_1_position = np.array([ele1_site[0], ele1_site[1], ele1_site[2] + 1.3])
                    O_2_position = np.array([ele1_site[0], ele1_site[1], ele1_site[2] + 2.51])
                else:
                    O_1_position = np.array([ele1_site[0], ele1_site[1], ele1_site[2] + 1.3])
                    O_2_position = np.array([ele2_site[0], ele2_site[1], ele2_site[2] + 1.3])

                if atom.index == atoms[0][0]:
                    atom.position = O_1_position
                elif atom.index == atoms[0][1]:
                    atom.position = O_2_position
        return slab
    
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
        
    def RMSE(self, a:list) -> float:
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
        COBonds = ana.get_bonds('C', 'O', unique = True)
        metal_CBonds = ana.get_bonds(self.metal_ele, 'C', unique = True)
        metal_OBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)

        OCOangles = ana.get_angles('O', 'C', 'O',unique = True)
        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        metal_O_list = []
        metal_C_list = []
        ads_list = []
        if metal_OBonds[0]:
            for i in metal_OBonds[0]:
                metal_O_list.append(i[0])
                metal_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] not in metal_O_list and i[1] not in metal_O_list:
                    ads_list.append(i)

        if OOOangles[0]:
            for j in OOOangles[0]:
                if j[0] not in metal_O_list and j[1] not in metal_O_list and j[2] not in metal_O_list:
                    ads_list.append(j)

        if metal_CBonds[0]:
            for i in metal_CBonds[0]:
                metal_C_list.append(i[0])
                metal_C_list.append(i[1])
        
        if COBonds[0]:  # 定义环境中的氧气分子
            for i in COBonds[0]:
                if i[0] not in metal_C_list and i[1] not in metal_O_list:
                    ads_list.append(i)

        if OCOangles[0]:
            for j in OCOangles[0]:
                if j[0] not in metal_O_list and j[1] not in metal_C_list and j[2] not in metal_O_list:
                    ads_list.append(j)

        if ads_list:
            ads = ads_list[np.random.randint(len(ads_list))]
        return ads, ads_list
    
    def to_desorb_adsorbate(self, slab):
        desorb = ()
        ana = Analysis(slab)

        OOBonds = ana.get_bonds('O', 'O', unique = True)
        COBonds = ana.get_bonds('C', 'O', unique = True)
        metal_C_Bonds = ana.get_bonds(self.metal_ele, 'C', unique = True)
        metal_O_Bonds = ana.get_bonds(self.metal_ele, 'O', unique=True)

        OCOangles = ana.get_angles('O', 'C', 'O',unique = True)
        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        metal_O_list = []
        metal_C_list = []
        desorb_list = []
        if metal_O_Bonds[0]:
            for i in metal_O_Bonds[0]:
                metal_O_list.append(i[0])
                metal_O_list.append(i[1])
        
        if metal_C_Bonds[0]:
            for i in metal_C_Bonds[0]:
                metal_C_list.append(i[0])
                metal_C_list.append(i[1])
        
        if COBonds[0]:  # 定义环境中的CO分子
            for i in COBonds[0]:
                if i[0] in metal_O_list or i[1] in metal_O_list:
                    desorb_list.append(i)

        if OCOangles[0]:
            for j in OOOangles[0]:
                if j[0] in metal_O_list or j[1] in metal_O_list or j[2] in metal_O_list:
                    desorb_list.append(j)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
        return desorb, desorb_list
    
    def _2D_distance(self, x1,x2, y1,y2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        return dis
    
    def layer_ele_atom_list(self, slab, ele):
        layer_ele = []
        layer_ele_atom_list = []
        layer_eleObond_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])

        for i in slab:
            if i.index in layer_List and i.symbol == ele:
                layer_ele.append(i.index)
        
        if layer_ele:
            ana = Analysis(slab)
            OObonds = ana.get_bonds(ele,'O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_ele and i[1] in layer_ele:
                        layer_eleObond_list.append(i[0])
                        layer_eleObond_list.append(i[1])

            for j in layer_ele:
                if j not in layer_eleObond_list:
                    layer_ele_atom_list.append(j)
        return layer_ele_atom_list
    
    
    def sub_ele_atom_list(self, slab, ele):
        sub_ele = []
        sub_ele_atom_list = []
        sub_eleObond_list = []
        sub_List = self.label_atoms(slab, [sub_z - fluct_d_metal, surf_z])

        for i in slab:
            if i.index in sub_List and i.symbol == ele:
                sub_ele.append(i.index)
        
        if sub_ele:
            ana = Analysis(slab)
            OObonds = ana.get_bonds(ele,'O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_ele and i[1] in sub_ele:
                        sub_eleObond_list.append(i[0])
                        sub_eleObond_list.append(i[1])

            for j in sub_ele:
                if j not in sub_eleObond_list:
                    sub_ele_atom_list.append(j)
        return sub_ele_atom_list
    
    def generate_initial_slab_clean_metal_surface(self):
        slab = fcc111(self.metal_ele, size=(4, 4, 4), vacuum=10.0)
        '''delList = [108, 109, 110, 111, 112, 113, 114, 119, 120, 125,
                   126, 131, 132, 137, 138, 139, 140, 141, 142, 143]
        del slab[[i for i in range(len(slab)) if i in delList]]'''
        return slab
    
    def generate_initial_slab_oxide_metal_surface(self):
        a = 3.05
        metal = crystal(self.metal_ele,[(0,0,0)],spacegroup=225,cellpar=[a,a,a,90,90,90])
        O1 = Atom('O',(0.25*a,0.25*a,0.25*a))
        O2 = Atom('O',(0.25 * a, 0.75*a,0.75*a))
        O3 = Atom('O',(0.75 * a, 0.25*a,0.75*a))
        O4 = Atom('O',(0.75 * a, 0.75*a,0.25*a))
        metal_O = metal + O1
        metal_O = metal_O + O2
        metal_O = metal_O + O3
        metal_O = metal_O + O4
        metal_O_surface= surface(metal_O,(1,0,1),2)
        metal_O_surface = metal_O_surface*(2,4,1)

        # add_vacuum(PdOsurface, vacuum = 15.0)
        uc = metal_O_surface.get_cell()
        uc[2] = np.array([0, 0, 15.0])

        metal_O_surface.set_cell(uc)
        return metal_O_surface
    
    def get_sites(self, surf):
        atop = surf.get_positions()
        pos_ext = surf.get_positions()
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
        total_surf_sites = []

        for i in top_sites:
            sites_1.append(np.transpose(np.append(i, 1)))
        for i in bridge_sites:
            sites_1.append(np.transpose(np.append(i, 2)))
        for i in hollow_sites:
            sites_1.append(np.transpose(np.append(i, 3)))
        for i in sites_1:
            total_surf_sites.append(np.append(i, 0))

        total_surf_sites = np.array(total_surf_sites)
        return top_sites, bridge_sites, hollow_sites, total_surf_sites

    def split_element(self, molecule):
        mol = Chem.MolFromSmiles(molecule)
        ele_list = []
        for ele in mol.GetAtoms():
            ele_list.append(ele.GetSymbol())
        return ele_list
    
    # the func to find target atom or molecule in the slab and update the prob_list
    def update_molecule_list(self, slab, molecule_list, n_mol_list, identify_single_atom = None): 
        # for example:
        # molecule_list such as self.desorb_list = ['CO', 'O2', 'CO2']
        # n_mol_list such as self.num_mole_diffuse_list = [0.0, 0.0, 0.0, 0.0]

        ana = Analysis(slab)
        layer_list = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        mol_info_list = [[], [], []]    # a list for len_mole = 1、2、3
        # TODO: need to be modified
        for i in range(len(molecule_list)):
            mol = molecule_list[i]  # mol = 'O'
            ele_list = self.split_element(mol)  # ele_list = ['O']

            if len(ele_list) == 1:
                inorganic_bond_ele_list = []
                metal_inorganic_bond_ele_list = []
                ele = ele_list[0]   # ele = 'O'
                layer_ele = []  # layer层中，该element的总体数量
                num_ele = []    # layer层中，单独存在的该atom or molecule的总体数量
                for atom in slab:
                    if atom.symbol == ele and atom.index in layer_list:
                        layer_ele.append(atom.index)    # layer_O_list.index

                if layer_ele:   #确保layer层中存在ele
                    for j in self.ele_list: # self.ele_list = ['O', 'C', 'Pd']
                        ele_12_Bonds = ana.get_bonds(ele, j, unique = True)
                        if ele_12_Bonds[0]:
                            for bond in ele_12_Bonds[0]: 
                                if j != self.metal_ele:
                                    if bond[0] not in inorganic_bond_ele_list and bond[0] in layer_ele:
                                        inorganic_bond_ele_list.append(bond[0])
                                else:
                                    if bond[0] not in metal_inorganic_bond_ele_list and bond[0] in layer_ele:
                                        metal_inorganic_bond_ele_list.append(bond[0])

                if metal_inorganic_bond_ele_list:
                    for atom_index in metal_inorganic_bond_ele_list:
                        if atom_index not in inorganic_bond_ele_list:
                            num_ele.append(atom_index)

                if num_ele:
                    for k in num_ele:
                        if k not in mol_info_list[0]:
                            mol_info_list[0].append(k)            
                            
                if mol_info_list[0]:
                    n_mol_list[i] = len(num_ele)
                '''else:
                    raise ValueError('no single atom in the situation len(mole) = 1')'''

                    
            elif len(ele_list) == 2:        #['CO']
                ele_1 = ele_list[0] # for exp: ele_1 = 'C', ele_2 = 'O', if we want to get the single 'CO' molecule
                ele_2 = ele_list[1] # the bond C-O should be existed, and bond C-Pd or O-Pd should be existed either
                layer_ele_1 = []    # layer_c_indexs
                layer_ele_2 = []    # layer_O_indexs
                ele_12_bond_list = []   # COBonds
                num_molecule = []   # molecule_COBonds

                if ele_1 == ele_2:
                    for atom in slab:  # 寻找在layer层中的ele_1 和 ele_2
                        if atom.symbol == ele_1 and atom.index in layer_list:
                            layer_ele_1.append(atom.index)
                            layer_ele_2.append(atom.index)
                else:
                    for atom in slab:  # 寻找在layer层中的ele_1 和 ele_2
                        if atom.symbol == ele_1 and atom.index in layer_list:
                            layer_ele_1.append(atom.index)
                        elif atom.symbol == ele_2 and atom.index in layer_list:
                            layer_ele_2.append(atom.index)

                if layer_ele_1 and layer_ele_2: # 确保layer层中同时存在ele_1和ele_2
                    ele_12_Bonds = ana.get_bonds(ele_1, ele_2, unique = True)
                    if ele_12_Bonds[0]:
                        for bond in ele_12_Bonds[0]:
                            if bond[0] in layer_ele_1 and bond[1] in layer_ele_2:
                                ele_12_bond_list.append(bond)

                    inorganic_angle_ele_12_list = []
                    metal_inorganic_angle_ele_12_list = []

                    if ele_12_bond_list:    # 如果slab中ele_1与ele_2成键
                        for ele in self.ele_list:   # 'Pd', 'C', 'O'
                            angle_type_1 = ana.get_angles(ele_1, ele_2, ele, unique = True)
                            angle_type_2 = ana.get_angles(ele, ele_1, ele_2, unique = True)
                            if angle_type_1[0] or angle_type_2[0]:
                                
                                for angle_1 in angle_type_1[0]: # 获得体系+环境中ele_1与ele_2成键对的list
                                    if ele != self.metal_ele:   #观察所选择的ele_1、ele_2的成键是否是单独的
                                        if (angle_1[0], angle_1[1]) not in inorganic_angle_ele_12_list and (angle_1[0], angle_1[1]) in ele_12_bond_list:
                                            inorganic_angle_ele_12_list.append((angle_1[0], angle_1[1]))
                                    else:
                                        if (angle_1[0], angle_1[1]) not in metal_inorganic_angle_ele_12_list and (angle_1[0], angle_1[1]) in ele_12_bond_list:
                                            metal_inorganic_angle_ele_12_list.append((angle_1[0], angle_1[1]))

                                for angle_2 in angle_type_2[0]: # 获得体系+环境中ele_1与ele_2成键对的list
                                    if ele != self.metal_ele:   #观察所选择的ele_1、ele_2的成键是否是单独的
                                        if (angle_2[1], angle_2[2]) not in inorganic_angle_ele_12_list and (angle_2[1], angle_2[2]) in ele_12_bond_list:
                                            inorganic_angle_ele_12_list.append((angle_2[1], angle_2[2]))
                                    else:
                                        if (angle_2[1], angle_2[2]) not in metal_inorganic_angle_ele_12_list and (angle_2[1], angle_2[2]) in ele_12_bond_list:
                                            metal_inorganic_angle_ele_12_list.append((angle_2[1], angle_2[2]))
                            
                            if metal_inorganic_angle_ele_12_list:   # ele_1和ele_2需要首先和体系中的金属原子成键才能执行后续动作：dissociation、diffusion、desorb等
                                if inorganic_angle_ele_12_list:
                                    for bond in metal_inorganic_angle_ele_12_list:
                                        if bond not in inorganic_angle_ele_12_list:
                                            num_molecule.append(bond)
                                else:
                                    for bond in metal_inorganic_angle_ele_12_list:
                                        num_molecule.append(bond)   

                        if identify_single_atom:
                            if num_molecule:
                                for bond in num_molecule:
                                    if bond not in mol_info_list[1]:
                                        mol_info_list[1].append(bond)
                                        
                                if mol_info_list[1]:
                                    n_mol_list[i] = len(mol_info_list[1])
                        else:
                            for bond in ele_12_bond_list:
                                if bond not in mol_info_list[1]:
                                    mol_info_list[1].append(bond)

                            if mol_info_list[1]:
                                n_mol_list[i] = len(mol_info_list[1])

                '''if identify_single_atom:    # identify_single_atom = True for diffusion, desportion; identify_single_atom = False for dissociation
                    if num_molecule:
                        n_mol_list[i] = len(num_molecule)
                    else:
                        raise ValueError('no single atom in the situation len(mole) = 2')
                else:
                    if ele_12_bond_list:
                        n_mol_list[i] = len(ele_12_bond_list)
                    else:
                        raise ValueError('no single atom in the situation len(mole) = 2')'''


            elif len(ele_list) == 3:
                ele_1 = ele_list[0]     #PdCO or OCO
                ele_2 = ele_list[1]
                ele_3 = ele_list[2]
                layer_ele_1 = []
                layer_ele_2 = []
                layer_ele_3 = []
                ele_123_list = []
                for atom in slab:  # 寻找在layer层中的ele_1 和 ele_2
                    if atom.symbol == ele_1 and atom.index in layer_list:
                        layer_ele_1.append(atom.index)
                    elif atom.symbol == ele_2 and atom.index in layer_list:
                        layer_ele_2.append(atom.index)
                    elif atom.symbol == ele_3 and atom.index in layer_list:
                        layer_ele_3.append(atom.index)

                if layer_ele_1 and layer_ele_2 and layer_ele_3: # 确保layer层中同时存在ele_1和ele_2和ele_3
                    ele_123_Angles = ana.get_angles(ele_1, ele_2, ele_3, unique = True)
                    if ele_123_Angles[0]:
                        for mole in ele_123_Angles[0]:
                            ele_123_list.append(mole)
                        
                            if mole not in mol_info_list[2]:
                                mol_info_list[2].append(mole)
                if mol_info_list[2]:
                    n_mol_list[i] = len(mol_info_list[2])
                '''else:
                    raise ValueError('no single atom in the situation len(mole) = 3')'''
            else:
                raise ValueError('please enable the atom numbers in the molecule <= 3')
            
        # now we get a num_list for the target action
        return n_mol_list, mol_info_list
    
    # to update the prob list every time we select the target action
    def update_eprob_list(self, eprob_list, index, relative_energy = 0):
        # prob_list such as self.prob_diffuse_list = [0.0, 0.0, 0.0, 0.0]
        eprob_list[index] -= self.lr * relative_energy
        sum_prob = sum(eprob_list)
        for i in range(len(eprob_list)):
            eprob_list[i] = eprob_list[i] / sum_prob
        return eprob_list
    
    def get_prob_list(self, eprob_list, num_list):
        dot_list = []
        prob_list = []
        
        for i in range(len(eprob_list)):
            dot_list.append(eprob_list[i] * num_list[i])
        sum_prob = sum(dot_list)

        if sum_prob:
            for i in range(len(dot_list)):
                dot_list[i] = dot_list[i] / sum_prob
        
        for i in range(len(dot_list)):
            sum_j = 0
            for j in range(i + 1):
                sum_j += dot_list[j]
            prob_list.append(sum_j)
        
        prob_list.append(0.0)

        return prob_list
    

    def add_mole(self, atom, mole, d):
        new_state = atom.copy()
        energy_1 = self.lasp_single_calc(new_state)

        if len(mole) == 2:
            ele_1 = Atom(mole[0], (3.0, 3.0, 3.0))
            ele_2 = Atom(mole[1], (3.0, 3.0, 3.0 + d))
            new_state = new_state + ele_1
            new_state = new_state + ele_2

        elif len(mole) == 3:
            ele_1 = Atom(mole[0], (3.0, 3.0, 3.0 + d))
            ele_2 = Atom(mole[1], (3.0, 3.0, 3.0))
            ele_3 = Atom(mole[2], (3.0, 3.0, 3.0 - d))

            new_state = new_state + ele_1
            new_state = new_state + ele_2
            new_state = new_state + ele_3

        energy_2 = self.lasp_single_calc(new_state)
        energy = energy_2 - energy_1
        return energy

    def get_mole_energy(self, slab, mole):
        if mole == 'OO':
            energy = self.add_mole(slab, mole, 1.21)
        elif mole == 'CO':
            energy = self.add_mole(slab, mole, 1.13)
        return [energy, mole]

    def get_atoms_symbol_list(self, atoms):
        symbol_list = []
        for atom in atoms:
            symbol_list.append(atom.symbol)
        return symbol_list

    def to_pad_the_array(self, array:array, max_len:int, position = True):
        if position:
            array = np.append(array, [0.0, 0.0, 0.0] * (max_len - array.shape[0]))
            array = array.reshape(int(array.shape[0]/3), 3)
        else:
            array = np.append(array, [0.0] * (max_len - array.shape[0]))
        return array
    
    def _use_Painn_description(self, atoms):
        input_dict = painn.atoms_to_graph_dict(atoms, self.cutoff)
        atom_model = painn.PainnDensityModel(
            num_interactions = self.num_interactions,
            hidden_state_size = self.hidden_state_size,
            cutoff = self.cutoff,
            atoms = atoms,
            embedding_size = self.embedding_size,
        )
        atom_representation_scalar, atom_representation_vector = atom_model(input_dict)

        atom_representation_scalar = np.array(self.pd(torch.tensor(np.array(atom_representation_scalar[0].tolist()))))

        # print(atom_representation_vector[0].shape)
        atom_representation_vector = rearrange(atom_representation_vector[0], "a b c -> b a c")
        # print(atom_representation_vector.shape)

        atom_representation_vector = np.array(self.pd(torch.tensor(np.array(atom_representation_vector.tolist()))))
        # print(atom_representation_vector.shape)
        atom_representation_vector = rearrange(atom_representation_vector, "b a c -> a b c")
        # print(atom_representation_vector.shape)

        return [atom_representation_scalar, atom_representation_vector]


bottom_z = 10.0
deep_z = 12.2
sub_z = 14.4
surf_z = 16.6
layer_z = 19.6

fluct_d_metal = 1.1
fluct_d_layer = 3.0


Eleradii = [0.32, 0.46, 1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67, 1.55, 1.39,
            1.26, 1.16, 1.11, 1.03, 0.99, 0.96, 1.96, 1.71, 1.48, 1.36, 1.34, 1.22,
            1.19, 1.16, 1.11, 1.10, 1.12, 1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17,
            2.10, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.20, 1.28, 1.36,
            1.42, 1.40, 1.40, 1.36, 1.33, 1.31, 2.32, 1.96, 1.80, 1.63, 1.76, 1.74,
            1.73, 1.72, 1.68, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.70, 1.62, 1.52,
            1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24, 1.33, 1.44, 1.44, 1.51, 1.45,
            1.47, 1.42, 2.23, 2.01, 1.86, 1.75, 1.69, 1.70, 1.71, 1.72, 1.66, 1.66,
            1.68, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34,
            1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57, ]

Eledict  = { 'H':1,     'He':2,   'Li':3,    'Be':4,   'B':5,     'C':6,     'N':7,     'O':8,
            'F':9,     'Ne':10,  'Na':11,   'Mg':12,  'Al':13,   'Si':14,   'P':15,    'S':16,
		'Cl':17,   'Ar':18,  'K':19,    'Ca':20,  'Sc':21,   'Ti':22,   'V':23,    'Cr':24,
		'Mn':25,   'Fe':26,  'Co':27,   'Ni':28,  'Cu':29,   'Zn':30,   'Ga':31,   'Ge':32,
		'As':33,   'Se':34,  'Br':35,   'Kr':36,  'Rb':37,   'Sr':38,   'Y':39,    'Zr':40,
		'Nb':41,   'Mo':42,  'Tc':43,   'Ru':44,  'Rh':45,   'Pd':46,   'Ag':47,   'Cd':48,
		'In':49,   'Sn':50,  'Sb':51,   'Te':52,  'I':53,    'Xe':54,   'Cs':55,   'Ba':56,
		'La':57,   'Ce':58,  'Pr':59,   'Nd':60,  'Pm':61,   'Sm':62,   'Eu':63,   'Gd':64, 
		'Tb':65,   'Dy':66,  'Ho':67,   'Er':68,  'Tm':69,   'Yb':70,   'Lu':71,   'Hf':72, 
		'Ta':73,   'W':74,   'Re':75,   'Os':76,  'Ir':77,   'Pt':78,   'Au':79,   'Hg':80, 
		'Tl':81,   'Pb':82,  'Bi':83,   'Po':84,  'At':85,   'Rn':86,   'Fr':87,   'Ra':88, 
		'Ac':89,   'Th':90,  'Pa':91,   'U':92,   'Np':93,   'Pu':94,   'Am':95,   'Cm':96, 
		'Bk':97,   'Cf':98,  'Es':99,   'Fm':100, 'Md':101,  'No':102,  'Lr':103,  'Rf':104, 
		'Db':105,  'Sg':106, 'Bh':107,  'Hs':108, 'Mt':109,  'Ds':110,  'Rg':111,  'Cn':112, 
		'Nh':113, 'Fl':114, 'Mc':115, 'Lv':116, 'Ts':117, 'Og':118} 

r_C = Eleradii[Eledict['C'] - 1]
r_O = Eleradii[Eledict['O'] - 1]
r_Pd = Eleradii[Eledict['Pd'] - 1]

d_O_Pd = r_O + r_Pd
d_O_C = r_C + r_O
d_C_Pd = r_C + r_Pd
d_O_O = 2 * r_O
d_Pd_Pd = 2 * r_Pd
d_C_C = 2 * r_C

