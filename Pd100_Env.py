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
from ase.build import fcc100, add_adsorbate
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.calculators.lasp_PdO import LASP
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
# from slab import images

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
slab = fcc100('Pd', size=(6, 6, 4), vacuum=20.0)
'''delList = [108, 109, 110, 111, 112, 113, 114, 119, 120, 125, 126,
           131, 132, 137, 138, 139, 140, 141, 142, 143]
del slab[[i for i in range(len(slab)) if i in delList]]'''

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
# ACTION_SPACES = ['ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']
ACTION_SPACES = ['ADS', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']
# TODO:
'''
    3.add "LAT_TRANS" part into step(considered in the cluster situation)
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
                 use_DESW = None):
        
        self.initial_state = slab.copy()  # 设定初始结构
        self.to_constraint(self.initial_state)
        self.initial_state, self.energy, self.force = self.lasp_calc(self.initial_state)  # 由于使用lasp计算，设定初始能量

        self.episode = 0  # 初始化episode为0
        self.max_episodes = max_episodes

        self.save_every = save_every
        self.save_every_min = save_every_min
        self.plot_every = plot_every
        self.use_DESW = use_DESW

        self.step_size = step_size  # 设定单个原子一次能移动的距离
        self.timesteps = timesteps  # 定义episode的时间步上限

#       self.episode_reward = 0  # 初始化一轮epsiode所获得的奖励
        self.timestep = 0  # 初始化时间步数为0
        
        # self.H = 112690 * 32/ 96485   # 没有加入熵校正, 单位eV
        self.reaction_H = reaction_H
        self.reaction_n = reaction_n
        self.delta_s = delta_s
        self.H = self.reaction_H * self.reaction_n

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
        self.free_atoms = list(set(range(len(self.initial_state))) - set(bottomList))
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
        self.observation_space = self.get_observation_space()

        # 一轮过后重新回到初始状态
        self.reset()

        return

    def step(self, action):
        pro = 1  # 定义该step完成该动作的概率，初始化为1
        barrier = 0
        self.steps = 50  # 定义优化的最大步长
        reward = 0  # 定义初始奖励为0

        diffusable = 1
        self.action_idx = action
        RMSD_similar = False
        kickout = False
        RMSE = 10

        self.done = False  # 开关，决定episode是否结束
        done_similar = False
        episode_over = False  # 与done作用类似

        self.atoms, previous_structure, previous_energy = self.state

        # 定义表层、次表层、深层以及环境层的平动范围
        self.lamada_d = 0.2
        self.lamada_s = 0.4
        self.lamada_layer = 0.6
        self.lamada_env = 0


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

        self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint, self.layer_atom, self.surf_atom, self.sub_atom, self.deep_atom, self.envList = self.get_surf_sites(
            self.atoms)

        # env_list = self.label_atoms(self.atoms, [2.0- fluct_d_layer, 2.0 + fluct_d_layer])  # 判断整个slab中是否还存在氧气分子，若不存在且动作依旧执行吸附，则强制停止
        _,  ads_exist = self.to_ads_adsorbate(self.atoms)
        if not ads_exist and action == 0:
            # self.done = True
            self.action_idx = 1

        layerList = self.label_atoms(self.atoms, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        layer_O = []
        for i in layerList:
            if self.atoms[i].symbol == 'O':
                layer_O.append(i)

        subList = self.label_atoms(self.atoms, [sub_z - fluct_d_Pd , surf_z])
        sub_O = []
        for i in subList:
            if self.atoms[i].symbol == 'O':
                sub_O.append(i)
        
        if not bool(layer_O) and self.action_idx == 6:
            self.action_idx = 1
                        
        '''——————————————————————————————————————————以下是动作选择————————————————————————————————————————————————————————'''
        if self.action_idx == 0:
            self.atoms = self.choose_ads_site(self.atoms, self.total_s)
            # return new_state,new_state_energy

            '''elif self.action_idx == 1:

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


        elif self.action_idx == 2:
            initial_state = self.atoms.copy()
            zeta = math.pi * 9 / 180
            central_point = np.array([initial_state.cell[0][0] / 2, initial_state.cell[1][1] / 2, 0])
            matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
            matrix = np.array(matrix)

            for atom in initial_state.positions:
                if 24.5 < atom[2] < 44.0 :
                    atom += np.array(
                        (np.dot(matrix, (np.array(atom.tolist()) - central_point).T).T + central_point).tolist()) - atom
            self.atoms.positions = initial_state.get_positions()



        elif self.action_idx == 3:
            initial_state = self.atoms.copy()
            zeta = -math.pi * 9 / 180
            central_point = np.array([initial_state.cell[0][0] / 2, initial_state.cell[1][1] / 2, 0])
            matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
            matrix = np.array(matrix)

            for atom in initial_state.positions:

                if 24.5 < atom[2] < 44.0:
                    atom += np.array(
                        (np.dot(matrix, (np.array(atom.tolist()) - central_point).T).T + central_point).tolist()) - atom
            self.atoms.positions = initial_state.get_positions()'''


        elif self.action_idx == 1:
            self.atoms.set_constraint(constraint)
            self.atoms.calc = EMT()
            dyn = Langevin(self.atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory=save_path_md,
                           logfile='MD.log')
            dyn.run(self.steps)

            '''------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------'''

        elif self.action_idx == 2:  # 表面上氧原子的扩散，单原子行为
            self.atoms, diffusable = self.to_diffuse_oxygen(self.atoms, self.total_s)

        elif self.action_idx == 3:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
            selected_drill_O_list = []
            layer_O_atom_list = self.layer_O_atom_list(self.atoms)
            sub_O_atom_list = self.sub_O_atom_list(self.atoms)
            if layer_O_atom_list:
                for i in layer_O_atom_list:
                    selected_drill_O_list.append(i)
            if sub_O_atom_list:
                for j in sub_O_atom_list:
                    selected_drill_O_list.append(j)

            if selected_drill_O_list:
                selected_O = selected_drill_O_list[np.random.randint(len(selected_drill_O_list))]

                self.atoms = self.to_expand_lattice(self.atoms, 1.25, 1.25, 1.1)

                if selected_O in layer_O_atom_list:
                    self.atoms = self.to_drill_surf(self.atoms)
                elif selected_O in sub_O_atom_list:
                    self.atoms = self.to_drill_deep(self.atoms)

                self.to_constraint(self.atoms)
                self.atoms, _, _ = self.lasp_calc(self.atoms)
                self.atoms = self.to_expand_lattice(self.atoms, 0.8, 0.8, 10/11)
            else:
                self.atoms = self.choose_ads_site(self.atoms, self.total_s)
                reward -= 1

        elif self.action_idx == 4:  # 氧气解离
            self.atoms = self.O_dissociation(self.atoms)

        elif self.action_idx == 5:
            _,  desorblist = self.to_desorb_adsorbate(self.atoms)
            if desorblist:
                self.atoms = self.choose_ads_to_desorb(self.atoms)
            else:
                reward -= 1
            
        else:
            print('No such action')

        self.timestep += 1
        
        self.to_constraint(self.atoms)
   
        # 优化该state的末态结构以及next_state的初态结构
        self.atoms, current_energy, current_force = self.lasp_calc(self.atoms)
        self.top_s, self.bridge_s, self.hollow_s, self.total_s, constraint, self.layer_atom, self.surf_atom, self.sub_atom, self.deep_atom, self.envList = self.get_surf_sites(
            self.atoms)
        free_atoms = []
        for i in range(len(self.atoms)):
            if i not in constraint.index:
                free_atoms.append(i)
        len_atom = len(free_atoms)
        previous_atom = self.trajectories[-1]

        # kickout the structure if too similar
        if self.RMSD(self.atoms, previous_atom)[0] and (current_energy - previous_energy) > 0:
            self.atoms = previous_atom
            current_energy = previous_energy
            RMSD_similar = True
            
        if self.timestep > 3:
            if self.RMSD(self.atoms, self.trajectories[-2])[0] and (current_energy - self.history['energies'][-2]) > 0:
                self.atoms = previous_atom
                current_energy = previous_energy
                RMSD_similar = True
                reward -= 1

        if self.timestep > 21:
            if self.RMSD(self.atoms, self.trajectories[-20])[0] and (current_energy - self.history['energies'][-20]) > 0: 
                self.atoms = previous_atom
                current_energy = previous_energy
                RMSD_similar = True
                reward -= 1
        
        if RMSD_similar:
            kickout = True

        if self.to_get_bond_info(self.atoms):   # 如果结构过差，将结构kickout
            self.atoms = previous_atom
            current_energy = previous_energy
            kickout = True
            # current_force = self.history['forces'][-1]
            reward += -5

        if self.action_idx == 0:
            current_energy = current_energy - self.delta_s
            self.adsorb_history['traj'] = self.adsorb_history['traj'] + [self.atoms.copy()]
            self.adsorb_history['structure'] = self.adsorb_history['structure'] + [self.atoms.get_positions()]
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
            
        self.RMSD_list.append(self.RMSD(self.atoms, previous_atom)[1])

        if self.timestep > 6:
            current_action_list = self.history['actions'][-5:]
            result = all(x == current_action_list[0] for x in current_action_list)
            if result and self.action_idx == current_action_list[0] and (RMSD_similar and relative_energy >= 0):
                self.repeat_action += 1
                reward -= self.repeat_action * 1
            elif result and self.action_idx != current_action_list[0]:
                self.repeat_action = 0
      
        if self.action_idx in [2,3,4]: 
            barrier = self.check_TS(previous_atom, self.atoms, previous_energy, current_energy, self.action_idx)    # according to Boltzmann probablity distribution
            if barrier > 5:
                reward += self.get_reward_trans(5.0)
                barrier = 5.0
            else:
                # reward += math.tanh(-relative_energy /(self.H * 8.314 * self.temperature_K)) * (math.pow(10.0, 5))
                reward += self.get_reward_trans(relative_energy)

        current_structure = self.atoms.get_positions()

        self.energy = current_energy
        self.force = current_force

        observation = self.get_obs()  # 能观察到该state的结构与能量信息

        self.state = self.atoms, current_structure, current_energy

        # Update the history for the rendering

        self.history, self.trajectories = self.update_history(self.action_idx, kickout)

        '''sub_Pd_list = []
        sub_list = self.label_atoms(self.atoms, [sub_z - fluct_d_Pd, sub_z + fluct_d_Pd])
        for i in self.atoms:
            if i.index in sub_list and i.symbol == 'Pd':
                sub_Pd_list.append(i.index)
        if len(sub_Pd_list) > 25:
            reward -= (len(sub_Pd_list) - 25) * 5
        else:
            reward += 5'''

        env_Pd_list = []
        env_list = self.label_atoms(self.atoms, [43.33, 45.83])
        for i in self.atoms:    #查找是否Pd原子游离在环境中
            if i.index in env_list and i.symbol == 'Pd':
                env_Pd_list.append(i.index)
        
        exist_too_short_bonds = self.exist_too_short_bonds(self.atoms)

        if exist_too_short_bonds or env_Pd_list or self.energy - self.initial_energy > self.len_atom * self.max_energy_profile  or relative_energy > self.max_RE:
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
            

        '''Pd_z = []
        O_z = []
        for i in range(len(self.atoms)):
            if self.atoms[i].symbol == 'Pd':
                Pd_z.append(self.atoms.positions[i][2])
            if self.atoms[i].symbol == 'O':
                O_z.append(self.atoms.positions[i][2])

        highest_Pd_z = max(Pd_z)
        highest_O_z = max(O_z)
        lowest_O_z = min(O_z)
        ana = Analysis(self.atoms)
        OObonds = ana.get_bonds('O', 'O', unique = True)

        if not OObonds[0]:  # 若表层以及次表层的氧都以氧原子的形式存在，加分
            if highest_O_z < highest_Pd_z and lowest_O_z > 12.0:
                reward += 50'''

#        reward -= 0.5 # 每经历一步timesteps，扣一分

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
            if self.episode % self.save_every == 0:
                self.save_episode()
                self.plot_episode()

        
        return observation, reward, episode_over, [done_similar]


    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d.npz' % self.episode)
        np.savez_compressed(
            save_path,
            traj = self.trajectories,
            initial_energy=self.initial_energy,
            energies=self.history['energies'],
            actions=self.history['actions'],
            structures=self.history['structures'],
            timesteps=self.history['timesteps'],
            forces = self.history['forces'],
            reward = self.history['reward'],

            adsorb_traj=self.adsorb_history['traj'],
            adsorb_structure=self.adsorb_history['structure'],
            adsorb_energy=self.adsorb_history['energy'],
            adsorb_timesteps = self.adsorb_history['timesteps'],

            ts_energy = self.TS['energies'],
            ts_timesteps = self.TS['timesteps'],

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

        self.atoms = slab.copy()

        self.to_constraint(self.atoms)
        self.atoms, self.initial_energy, self.initial_force= self.lasp_calc(self.atoms)

        self.action_idx = 0
        self.episode_reward = 0.5 * self.timesteps
        self.timestep = 0

        self.total_steps = self.timesteps
        self.max_RE = 3
        self.min_RE_d = self.convergence * self.len_atom
        self.repeat_action = 0

        self.atoms = self.choose_ads_site(self.atoms, surf_sites)

        self.trajectories = []
        self.RMSD_list = []
        self.trajectories.append(self.atoms.copy())

        self.TS = {}
        # self.TS['structures'] = [slab.get_scaled_positions()[self.free_atoms, :]]
        self.TS['energies'] = [0.0]
        self.TS['timesteps'] = [0]

        self.adsorb_history = {}
        self.adsorb_history['traj'] = [slab]
        self.adsorb_history['structure'] = [slab.get_scaled_positions()[self.free_atoms, :].flatten()]
        self.adsorb_history['energy'] = [0.0]
        self.adsorb_history['timesteps'] = [0]

        results = ['energies', 'actions', 'structures', 'timesteps', 'forces', 'scaled_structures', 'real_energies', 'reward']
        for item in results:
            self.history[item] = []
        self.history['energies'] = [0.0]
        self.history['real_energies'] = [0.0]
        self.history['actions'] = [0]
        self.history['forces'] = [self.initial_force]
        self.history['structures'] = [slab.get_positions().flatten()]
        self.history['scaled_structures'] = [slab.get_scaled_positions()[self.free_atoms, :].flatten()]
        self.history['timesteps'] = [0]
        self.history['reward'] = []

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
        observation_space = spaces.Dict({'structures':
            spaces.Box(
                low=-1,
                high=2,
                shape=(self.len_atom * 3, ),
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
                shape=(self.len_atom * 3, ),
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
        observation['structure'] = self.atoms.get_scaled_positions()[self.free_atoms, :].flatten()
        observation['energy'] = np.array([self.energy - self.initial_energy]).reshape(1, )
        # observation['force'] = self.force[self.free_atoms, :].flatten()
        return observation['structure']

    def update_history(self, action_idx, kickout):
        self.trajectories.append(self.atoms.copy())
        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.energy - self.initial_energy]
        self.history['forces'] = self.history['forces'] + [self.force]
        self.history['actions'] = self.history['actions'] + [action_idx]
        self.history['structures'] = self.history['structures'] + [self.atoms.get_positions().flatten()]
        self.history['scaled_structures'] = self.history['scaled_structures'] + [self.atoms.get_scaled_positions()[self.free_atoms, :].flatten()]
        if not kickout:
            self.history['real_energies'] = self.history['real_energies'] + [self.energy - self.initial_energy]

        return self.history, self.trajectories

    def transition_state_search(self, previous_atom, current_atom, previous_energy, current_energy, action):
        layerlist = self.label_atoms(previous_atom, [26.0, 31.0])
        layer_O = []
        for i in layerlist:
            if previous_atom[i].symbol == 'O':
                layer_O.append(i)

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
            '''if action == 1:
                if current_energy - previous_energy < -1.0:
                    barrier = 0
                elif current_energy - previous_energy >= -1.0 and current_energy - previous_energy <= 1.0:
                    barrier = np.random.normal(2, 2/3)
                else:
                    barrier = 4.0

            if action == 2 or action == 3:
                barrier = math.log(1 + pow(math.e, current_energy-previous_energy), 10)'''
            if action == 2:
                barrier = math.log(0.5 + 1.5 * pow(math.e, 2 *(current_energy - previous_energy)), 10)
            elif action == 3:
                barrier = 0.93 * pow(math.e, 0.615 * (current_energy - previous_energy)) - 0.16
            elif action == 4:
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
        new_state = state.copy()

        add_total_sites = []
        layer_O = []

        for ads_sites in surf_sites:
            for i in layerList:
                if state[i].symbol == 'O':
                    layer_O.append(i)
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    distance = self.distance(ads_sites[0], ads_sites[1], ads_sites[2] + 1.3, state.get_positions()[i][0],
                                           state.get_positions()[i][1], state.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                add_total_sites.append(ads_sites)
        
        if add_total_sites:
            ads_site = add_total_sites[np.random.randint(len(add_total_sites))]
            new_state = state.copy()
            ads,  _ = self.to_ads_adsorbate(new_state)
            # env_List = self.label_atoms(new_state, [env_z - 2.0, env_z+2.0])
            '''env_2 = self.label_atoms(new_state, [0.0, 4.0])
            for i in env_2:
                env_List.append(i)
            env_s = state.copy()

            del env_s[[i for i in range(len(env_s)) if i not in env_List]]
            env_index = min(env_List)'''
            if len(ads):
                if len(ads) == 2:
                    delstatelist = [ads[0], ads[1]]
                elif len(ads) == 3:
                    delstatelist = [ads[0], ads[1], ads[2]]
            del new_state[[i for i in range(len(new_state)) if i in delstatelist]]

            # delenvlist = [0, 1]
            # del env_s[[i for i in range(len(env_s)) if i in delenvlist]]
            if len(ads):
                if len(ads) == 2:
                    O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + 1.3))
                    O2 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + 2.51))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    # ads = O1 + O2
                elif len(ads) == 3:
                    O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + 1.3))
                    O2 = Atom('O', (ads_site[0], ads_site[1] + 1.09, ads_site[2] + 1.97))
                    O3 = Atom('O', (ads_site[0], ads_site[1] - 1.09, ads_site[2] + 1.97))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    new_state = new_state + O3
        return new_state
    
    def choose_ads_to_desorb(self, state):
        new_state = state.copy()

        # add_total_sites = []
        layer_O = []
        O_position = []
        desorblist = []

        layerList = self.label_atoms(state, [16.0, 18.0])
        for i in layerList:
            if state[i].symbol == 'O':
                layer_O.append(i)
        if layer_O: 
            desorb,  _ = self.to_desorb_adsorbate(new_state)
            if len(desorb):
                if len(desorb) == 2:
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                elif len(desorb) == 3:
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                    desorblist.append(desorb[2])
            
            for i in desorblist:
                O_position.append(state.get_positions()[i][0])
                O_position.append(state.get_positions()[i][1])
                O_position.append(state.get_positions()[i][2])

            del new_state[[i for i in range(len(new_state)) if i in desorblist]]

            if len(desorb):
                if len(desorb) == 2:
                    O1 = Atom('O', (O_position[0], O_position[1], O_position[2] + 6.0))
                    O2 = Atom('O', (O_position[3], O_position[4], O_position[5] + 6.0))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    # ads = O1 + O2
                elif len(desorb) == 3:
                    O1 = Atom('O', (O_position[0], O_position[1], O_position[2] + 6.0))
                    O2 = Atom('O', (O_position[3], O_position[4], O_position[5] + 6.0))
                    O3 = Atom('O', (O_position[6], O_position[7], O_position[8] + 6.0))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    new_state = new_state + O3
        return new_state

    def get_surf_sites(self, slab):
        state = slab.copy()

        layerList = self.label_atoms(state, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        surfList = self.label_atoms(state, [surf_z - fluct_d_Pd, surf_z + fluct_d_Pd * 2])
        for i in surfList:
            if state[i].symbol == 'O':
                surfList.remove(i)

        subList = self.label_atoms(state, [sub_z - fluct_d_Pd, sub_z + fluct_d_Pd])
        deepList = self.label_atoms(state, [deep_z - fluct_d_Pd, deep_z + fluct_d_Pd])
        bottomList = self.label_atoms(state, [bottom_z - fluct_d_Pd, bottom_z + fluct_d_Pd])
        envList = self.label_atoms(state, [env_z - 2.0, env_z + 2.0])
        env_2 = self.label_atoms(state, [0.0,  19.0])
        for i in env_2:
            envList.append(i)

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

        env = state.copy()
        del env[[i for i in range(len(env)) if i not in envList]]

        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        constraint_1 = FixAtoms(mask=[a.symbol != 'Pd' and a.index in envList for a in slab])
        # constraint_deepatomlist = sample(deepList, int(len(deepList)/2))
        # constraint_2 = FixAtoms(mask=[a.symbol != 'O' and a.index in constraint_deepatomlist for a in slab])
        # constraint_1.index = np.append(constraint_1.index, constraint_2.index)
        constraint.index = np.append(constraint.index, constraint_1.index)
        fix = state.set_constraint(constraint)

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

        return top_sites, bridge_sites, hollow_sites, total_surf_sites, constraint, layer_atom, surf_atom, sub_atom, deep_atom, envList

    def to_diffuse_oxygen(self, slab, surf_sites):
        layer_O = []
        to_diffuse_O_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        diffusable_sites = []
        interference_O_distance = []
        diffusable = True

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        for ads_sites in surf_sites:    # 寻找可以diffuse的位点
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
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

        if layer_O: # 防止氧原子被trap住无法diffuse
            for i in layer_O:
                to_other_O_distance = []
                for j in layer_O:
                    if j != i:
                        distance = self.distance(slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2],slab.get_positions()[j][0],
                                           slab.get_positions()[j][1], slab.get_positions()[j][2])
                        to_other_O_distance.append(distance)
                        
                if self.to_get_min_distances(to_other_O_distance,4):
                    d_min_4 = self.to_get_min_distances(to_other_O_distance, 4)
                    if d_min_4 > 2.0:
                        to_diffuse_O_list.append(i)
                else:
                    to_diffuse_O_list.append(i)

        if to_diffuse_O_list:
            selected_O_index = layer_O[np.random.randint(len(to_diffuse_O_list))]
            if diffusable_sites:
                diffuse_site = diffusable_sites[np.random.randint(len(diffusable_sites))]
            else:
                diffuse_site = slab.get_positions()[selected_O_index]
            interference_O_list = [i for i in layer_O if i != selected_O_index]
            for j in interference_O_list:
                d = self.atom_to_traj_distance(slab.positions[selected_O_index], diffuse_site, slab.positions[j])
                interference_O_distance.append(d)
            if interference_O_distance:
                if min(interference_O_distance) < 0.3 * d_O_O:
                    diffusable = False
        
            if diffusable:
                del slab[[j for j in range(len(slab)) if j == selected_O_index]]
                O = Atom('O', (diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.5))
                slab = slab + O
            
        return slab, diffusable

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
            mid_point_layer = [mid_point_x, mid_point_y, mid_point_z(surfList)]
        mid_point_surf = [mid_point_x, mid_point_y, mid_point_z(surfList)]
        mid_point_sub = [mid_point_x, mid_point_y, mid_point_z(subList)]

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
    
    def get_reward_trans(self, relative_energy):
        return -relative_energy / (self.H * self.k * self.temperature_K)

    def get_reward_tanh(self, relative_energy):
        reward = math.tanh(-relative_energy/(self.H * self.k * self.temperature_K))
        return reward
    
    def get_reward_sigmoid(self, relative_energy):
        return 2 * (0.5 - 1 / (1 + np.exp(-relative_energy/(self.H * self.k * self.temperature_K))))
    
    def to_drill_surf(self, slab):
        layer_O = []
        to_distance = []
        drillable_sites = []
        layer_O_atom_list = []
        layer_OObond_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])

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

        
        if layer_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_O and i[1] in layer_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in layer_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)

        if layer_O_atom_list:
            i = layer_O[np.random.randint(len(layer_O_atom_list))]
            position = slab.get_positions()[i]
            del slab[[j for j in range(len(slab)) if j == i]]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = drillable_sites[to_distance.index(min(to_distance))]
            O = Atom('O', (drill_site[0], drill_site[1], drill_site[2] +1.3))
            slab = slab + O

            lifted_atoms_list = []
            current_surfList = self.label_atoms(slab, [surf_z - fluct_d_Pd/2, surf_z + fluct_d_Pd])
            c_layerList = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
            current_layer_O = []
            for i in slab:
                if i.index in c_layerList and i.symbol == 'O':
                    current_layer_O.append(i.index)
            if current_layer_O:
                for i in current_layer_O:
                    current_surfList.append(i)
            for i in current_surfList:
                lifted_atoms_list.append(i)
            for j in lifted_atoms_list:
                slab.positions[j][2] += 1.0
        return slab
    
    def to_drill_deep(self, slab):
        # sub_O = []
        to_distance = []
        drillable_sites = []
        sub_O_atom_list = self.sub_O_atom_list(slab)
        # layer_OObond_list = []
        sub_List = self.label_atoms(slab, [sub_z - 2.0, sub_z + 2.0])

        deep_sites = self.get_deep_sites(slab)

        '''for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)'''
        
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

        if sub_O_atom_list:
            i = sub_O_atom_list[np.random.randint(len(sub_O_atom_list))]
            position = slab.get_positions()[i]
            del slab[[j for j in range(len(slab)) if j == i]]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = drillable_sites[to_distance.index(min(to_distance))]
            O = Atom('O', (drill_site[0], drill_site[1], drill_site[2] +1.3))
            slab = slab + O

            lifted_atoms_list = self.label_atoms(slab, [sub_z - 1.0, layer_z + fluct_d_layer])
            # current_surfList = self.label_atoms(slab, [surf_z - fluct_d_Pd/2, surf_z + fluct_d_Pd])
            '''current_sub_surfList = self.label_atoms(slab, [sub_z - fluct_d_Pd, surf_z + 2.0])
            c_layerList = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
            current_layer_O = []
            for i in slab:
                if i.index in c_layerList and i.symbol == 'O':
                    current_layer_O.append(i.index)
            if current_layer_O:
                for i in current_layer_O:
                    current_sub_surfList.append(i)
            for i in current_sub_surfList:
                lifted_atoms_list.append(i)'''
            for j in lifted_atoms_list:
                slab.positions[j][2] += 1.0
        return slab

    def O_dissociation(self, slab):
        ana = Analysis(slab)

        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds('Pd', 'O', unique=True)

        Pd_O_list = []
        dissociate_O2_list = []
        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] in Pd_O_list or i[1] in Pd_O_list:
                    dissociate_O2_list.append([(i[0], i[1])])

        if dissociate_O2_list:
            OO = dissociate_O2_list[np.random.randint(len(dissociate_O2_list))]
            # d = ana.get_values([OO])[0]
            zeta = self.get_angle_with_z(slab, OO) * 180/ math.pi -5
            fi = 30
            slab = self.oxy_rotation(slab, OO, zeta, fi)
            slab = self.to_dissociate(slab, OO)
        return slab
    
    def adsorbate_desportion(self, slab):
        ana = Analysis(slab)
        PdOOAngles = ana.get_angles()

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
        slab = fcc100('Pd', size=(6, 6, 4), vacuum=10.0)
        delList = [77, 83, 89, 95, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 119, 120, 125,
                   126, 131, 132, 137, 138, 139, 140, 141, 142, 143]
        del slab[[i for i in range(len(slab)) if i in delList]]
        return slab

    def get_layer_info(self, slab):
        state = slab.copy()

        layerList = self.label_atoms(state, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
        surfList = self.label_atoms(state, [surf_z - fluct_d_Pd, surf_z + fluct_d_Pd*2])
        for i in surfList:
            if state[i].symbol == 'O':
                surfList.remove(i)
        subList = self.label_atoms(state, [sub_z - fluct_d_Pd, sub_z + fluct_d_Pd])
        deepList = self.label_atoms(state, [deep_z - fluct_d_Pd, deep_z + fluct_d_Pd])
        bottomList = self.label_atoms(state, [bottom_z - fluct_d_Pd, bottom_z + fluct_d_Pd])
        envList = self.label_atoms(state, [env_z - 2.0, env_z + 2.0])
        env_2 = self.label_atoms(state, [0.0, 19.0])
        for i in env_2:
            envList.append(i)

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

        env = state.copy()
        del env[[i for i in range(len(env)) if i not in envList]]
        env_atom = env.get_positions()

        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        constraint_1 = FixAtoms(mask=[a.symbol != 'Pd' and a.index in envList for a in slab])
        # constraint_deepatomlist = sample(deepList, int(len(deepList)/2))
        # constraint_2 = FixAtoms(mask=[a.symbol != 'O' and a.index in constraint_deepatomlist for a in slab])
        # constraint_1.index = np.append(constraint_1.index, constraint_2.index)
        constraint.index = np.append(constraint.index, constraint_1.index)
        fix = state.set_constraint(constraint)

        return layer_atom, surf_atom, sub_atom, deep_atom, env_atom, constraint
    
    def RMSD(self, current_atoms, previous_atoms):
        similar = False
        _, _, _, _, constraint_p, _, _, _, _, _ = self.get_surf_sites(previous_atoms)
        free_atoms_p = []
        for i in range(len(previous_atoms)):
            if i not in constraint_p.index:
                free_atoms_p.append(i)
        len_atom_p = len(free_atoms_p)

        _, _, _, _, constraint_c, _, _, _, _, _ = self.get_surf_sites(current_atoms)
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
        layer_atom, surf_atom, sub_atom, deep_atom, env_atom, fix = self.get_layer_info(slab)

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
        layer_atom, surf_atom, sub_atom, deep_atom, env_atom, fix = self.get_layer_info(slab)

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
        atom.calc = LASP(task='local-opt', pot='PdO', potential='NN D3')
        energy = atom.get_potential_energy()
        force = atom.get_forces()
        atom = read_arc('allstr.arc', index = -1)
        return atom, energy, force
    
    def to_constraint(self, slab):
        bottomList = self.label_atoms(slab, [bottom_z - fluct_d_Pd, bottom_z + fluct_d_Pd])
        envList_1 = self.label_atoms(slab, [env_z - fluct_d_layer, env_z + fluct_d_layer])
        envList_2 = self.label_atoms(slab, [0.0,  19.0])
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
        constraint_1 = FixAtoms(mask=[a.symbol != 'Pd' and a.index in envList_1 for a in slab])
        constraint_2 = FixAtoms(mask=[a.symbol != 'Pd' and a.index in envList_2 for a in slab])
        # constraint_deepatomlist = sample(deepList, int(len(deepList)/2))
        # constraint_3 = FixAtoms(mask=[a.symbol != 'O' and a.index in constraint_deepatomlist for a in slab])
        # constraint_2.index = np.append(constraint_3.index, constraint_2.index)
        constraint_1.index = np.append(constraint_1.index, constraint_2.index)
        constraint.index = np.append(constraint.index, constraint_1.index)
        fix = slab.set_constraint(constraint)

    def exist_too_short_bonds(self,slab):
        exist = False
        ana = Analysis(slab)
        PdPdBonds = ana.get_bonds('Pd','Pd',unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds('Pd', 'O', unique=True)
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
        PdPdBonds = ana.get_bonds('Pd','Pd',unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds('Pd', 'O', unique=True)
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
        d = distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
        zeta = -math.pi * zeta / 180
        fi = -math.pi * fi / 180
        '''如果pos1[2] > pos2[2],atom_1旋转下来'''
        pos2_position = pos2
        pos1_position = [pos2[0]+ d*sin(zeta)*cos(fi), pos2[1] + d*sin(zeta)*sin(fi),pos2[2]+d*cos(zeta)]
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
        central_point = np.array([(slab.get_positions()[atoms[0][0]][0] + slab.get_positions()[atoms[0][1]][0])/2, 
                                  (slab.get_positions()[atoms[0][0]][1] + slab.get_positions()[atoms[0][1]][1])/2, (slab.get_positions()[atoms[0][0]][2] + slab.get_positions()[atoms[0][1]][2])/2])
        slab.positions[atoms[0][0]] += np.array([1.15*(slab.get_positions()[atoms[0][0]][0]-central_point[0]), 
                                                 1.15*(slab.get_positions()[atoms[0][0]][1]-central_point[1]), 1.15*(slab.get_positions()[atoms[0][0]][2]-central_point[2])])
        slab.positions[atoms[0][1]] += np.array([1.15*(slab.get_positions()[atoms[0][1]][0]-central_point[0]), 
                                                 1.15*(slab.get_positions()[atoms[0][1]][1]-central_point[1]), 1.15*(slab.get_positions()[atoms[0][1]][2]-central_point[2])])
        addable_sites = []
        layer_O = []
        layerlist = self.label_atoms(slab,[26.5,31.5])

        for ads_site in self.total_s:
            for i in layerlist:
                if slab[i].symbol == 'O':
                    layer_O.append(i)
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    to_distance = self.distance(ads_site[0], ads_site[1], ads_site[2] + 1.5, slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2])
                    to_other_O_distance.append(to_distance)
                if min(to_other_O_distance) > 1.5 * d_O_O:
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

        O1_site = addable_sites[O1_distance.index(min(O1_distance))]
        
        ad_2_sites = []
        for add_site in addable_sites:
            d = self.distance(add_site[0], add_site[1], add_site[2] + 1.3, O1_site[0], O1_site[1], O1_site[2])
            if d > 1.0:
                ad_2_sites.append(add_site)

        O2_distance = []
        for add_2_site in ad_2_sites:
            distance_2 = self.distance(add_2_site[0], add_2_site[1], add_2_site[2] + 1.3, slab.get_positions()[atoms[0][1]][0],
                                        slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][1]][2])
            O2_distance.append(distance_2)
        
        O2_site = ad_2_sites[O2_distance.index(min(O2_distance))]

        del slab[[i for i in range(len(slab)) if slab[i].index == atoms[0][0] or slab[i].index == atoms[0][1]]]

        if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
            O_1 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 1.3))
            O_2 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 2.51))
        else:

            O_1 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 1.3))
            O_2 = Atom('O', (O2_site[0], O2_site[1], O2_site[2] + 1.3))

        slab = slab + O_1
        slab = slab + O_2
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

    '''def wrap_out(self,current_atoms, previous_atoms):
        cell_x = current_atoms.cell[0][0]
        cell_y = current_atoms.cell[1][1]

        d_list = []
        x_list = []
        y_list = []

        x_p = previous_atoms.'''

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
        PdOBonds = ana.get_bonds('Pd', 'O', unique=True)

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
                    ads_list.append(j)

        if ads_list:
            ads = ads_list[np.random.randint(len(ads_list))]
        return ads, ads_list
    
    def to_desorb_adsorbate(self, slab):
        desorb = ()
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds('Pd', 'O', unique=True)

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

    def layer_O_atom_list(self, slab):
        layer_O = []
        layer_O_atom_list = []
        layer_OObond_list = []
        layer_List = self.label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        if layer_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_O and i[1] in layer_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in layer_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)
        return layer_O_atom_list
    
    def sub_O_atom_list(self, slab):
        sub_O = []
        sub_O_atom_list = []
        sub_OObond_list = []
        sub_List = self.label_atoms(slab, [sub_z - fluct_d_Pd, surf_z])

        for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)
        
        if sub_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_O and i[1] in sub_O:
                        sub_OObond_list.append(i[0])
                        sub_OObond_list.append(i[1])

            for j in sub_O:
                if j not in sub_OObond_list:
                    sub_O_atom_list.append(j)
        return sub_O_atom_list

def label_atoms(atoms, zRange):
    myPos = atoms.get_positions()
    return [
        i for i in range(len(atoms)) \
        if min(zRange) < myPos[i][2] < max(zRange)
    ]


fluct_d_Pd = 1.0
fluct_d_layer = 3.0
env_z = 46.335
layer_z = 29.0
layerList = label_atoms(slab, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
surf_z = 26.0
surfList = label_atoms(slab, [surf_z - fluct_d_Pd, surf_z + fluct_d_Pd * 2])
for i in surfList:
	if slab[i].symbol == 'O':
		surfList.remove(i)
sub_z = 24.0
subList = label_atoms(slab, [sub_z - fluct_d_Pd, sub_z + fluct_d_Pd])
deep_z = 22.0
deepList = label_atoms(slab, [deep_z - fluct_d_Pd, deep_z + fluct_d_Pd])
bottom_z = 20.0
bottomList = label_atoms(slab, [bottom_z - fluct_d_Pd, bottom_z + fluct_d_Pd])

layer = slab.copy()
del layer[[i for i in range(len(layer)) if i not in layerList]]
layer_atom = layer.get_positions()

surf = slab.copy()
del surf[[i for i in range(len(surf)) if i not in surfList]]
surf_atom = surf.get_positions()
atop = surf.get_positions()

sub_layer = slab.copy()
del sub_layer[[i for i in range(len(sub_layer)) if i not in subList]]
sub_atom = sub_layer.get_positions()

deep_layer = slab.copy()
del deep_layer[[i for i in range(len(deep_layer)) if i not in deepList]]
deep_atom = deep_layer.get_positions()

# 设定环境中的氧氛围
molecule = Atoms("OO", positions=[[0, 0, 0], [0, 0, 1.21]], pbc=True)

for i in atop:
    add_adsorbate(slab, molecule, 22.395 - 45.835, position=(i[0] + 0.5, i[1] + 0.5))
    add_adsorbate(slab, molecule, 27.895 - 45.835, position=(i[0] + 0.5, i[1] + 0.5))
    add_adsorbate(slab, molecule, 32.395 - 45.835, position=(i[0] + 0.5, i[1] + 0.5))
envList = label_atoms(slab, [2.0- fluct_d_layer, 6.0 + fluct_d_layer])
env = slab.copy()
del env[[i for i in range(len(env)) if i not in envList]]

# 固定住最下层原子以及环境中的氧气分子
constraint = FixAtoms(mask=[a.symbol != 'O' and a.index in bottomList for a in slab])
constraint_1 = FixAtoms(mask=[a.symbol != 'Pd' and a.index in envList for a in slab])
# constraint_deepatomlist = sample(deepList, int(len(deepList)/2))
# constraint_2 = FixAtoms(mask=[a.symbol != 'O' and a.index in constraint_deepatomlist for a in slab])
# constraint_1.index = np.append(constraint_1.index, constraint_2.index)
constraint.index = np.append(constraint.index, constraint_1.index)
fix = slab.set_constraint(constraint)


def distance(x1, y1, z1, x2, y2, z2):
    dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
    return dis


pos_ext = surf.get_positions()
tri = Delaunay(pos_ext[:, :2])
pos_nodes = pos_ext[tri.simplices]

bridge = []
hollow = []
"""Pd(100)表面吸附位点"""
for i in pos_nodes:
    if (distance(i[0][0], i[0][1], i[0][2], i[1][0], i[1][1], i[1][2])) < 3.0:
        bridge.append((i[0] + i[1]) / 2)
    else:
        hollow.append((i[0] + i[1]) / 2)
    if (distance(i[2][0], i[2][1], i[2][2], i[1][0], i[1][1], i[1][2])) < 3.0:
        bridge.append((i[2] + i[1]) / 2)
    else:
        hollow.append((i[2] + i[1]) / 2)
    if (distance(i[0][0], i[0][1], i[0][2], i[2][0], i[2][1], i[2][2])) < 3.0:
        bridge.append((i[0] + i[2]) / 2)
    else:
        hollow.append((i[0] + i[2]) / 2)

ontop = np.array(atop)
hollow = np.array(hollow)
bridge = np.array(bridge)

sites_1 = []
surf_sites = []
site_dict = {0: 'vacancy', 1: 'ontop', 2: 'bridge', 3: 'hollow', 4: 'layer_top_O', 5: 'layer_top_Pd'}

for i in ontop:
    sites_1.append(np.transpose(np.append(i, 1)))
for i in bridge:
    sites_1.append(np.transpose(np.append(i, 2)))
for i in hollow:
    sites_1.append(np.transpose(np.append(i, 3)))
for i in sites_1:
    surf_sites.append(np.append(i, 0))

surf_sites = np.array(surf_sites)

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

r_O = Eleradii[7]
r_Pd = Eleradii[45]

d_O_Pd = r_O + r_Pd
d_O_O = 2 * r_O
d_Pd_Pd = 2 * r_Pd

