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
from ase.cluster.wulff import wulff_construction
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
surfaces = [(1, 0, 0),(1, 1, 0), (1, 1, 1)]
esurf = [1.0, 1.0, 1.0]   # Surface energies.
lc = 3.89
size = 38 # Number of atoms
atoms = wulff_construction('Pd', surfaces, esurf,
                           size, 'fcc',
                           rounding='closest', latticeconstant=lc)

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
                 max_energy_profile = 0.5,
                 convergence = 0.005,
                 save_every=None,
                 save_every_min=None,
                 plot_every=None,
                 reaction_H = None,         #/(KJ/mol)
                 reaction_n = None,
                 delta_s = None,            #/eV
                 use_DESW = None):
        
        self.initial_state = atoms.copy()  # 设定初始结构
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
        # self.free_atoms = list(set(range(len(self.initial_state))) - set(bottomList))
        # self.len_atom = len(self.free_atoms)
        self.convergence = convergence

        # 设定环境温度为473 K，定义热力学能
        # T = 473.15
        self.temperature_K = temperature
        self.k = k  # eV/K
        self.thermal_energy = k * temperature * self.len_atom

        self.action_space = spaces.Dict({'action_type': spaces.Discrete(len(ACTION_SPACES)),
                                        'facet_selection': spaces.Discrete(len(atoms.get_surfaces()))})

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
        self.action_idx = action['action_type']
        
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

        surfList = []
        for facet in atoms.get_surfaces():
            atoms= self.cluster_rotation(atoms, facet)
            list = self.get_surf_atoms(atoms)
            for i in list:
                surfList.append(i)
            atoms = self.recover_rotation(atoms, facet)

        surfList = [i for n, i in enumerate(surfList) if i not in surfList[:n]]
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index not in surfList for a in atoms])
        assert self.action_space.contains(self.action_idx), "%r (%s) invalid" % (
            self.action_idx,
            type(self.action_idx),
        )
        
        if action in [0, 2, 3, 5, 6]:
            self.facet_selection = action['facet_selection']
            self.cluster_rotation(self.atoms, self.facet_selection)

        self.muti_movement = np.array([np.random.normal(0.25,0.25), np.random.normal(0.25,0.25), np.random.normal(0.25,0.25)])
        # 定义层之间的平动弛豫

        #   定义保存ts，min和md动作时的文件路径
        save_path_ts = None
        save_path_ads = None
        save_path_md = None

        # env_list = self.label_atoms(self.atoms, [2.0- fluct_d_layer, 2.0 + fluct_d_layer])  # 判断整个slab中是否还存在氧气分子，若不存在且动作依旧执行吸附，则强制停止
        _,  ads_exist = self.to_ads_adsorbate(self.atoms)
        if not ads_exist and action == 0:
            # self.done = True
            self.action_idx = 1

        layerList = self.get_layer_atoms(self.atoms)
        layer_O = []
        for i in layerList:
            if self.atoms[i].symbol == 'O':
                layer_O.append(i)

        surfList = []
        for facet in atoms.get_surfaces():
            atoms= self.cluster_rotation(atoms, facet)
            list = self.get_surf_atoms(atoms)
            for i in list:
                surfList.append(i)
            atoms = self.recover_rotation(atoms, facet)

        surfList = [i for n, i in enumerate(surfList) if i not in surfList[:n]]
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index not in surfList for a in atoms])
        
        subList = self.get_sub_atoms(self.atoms)
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

        elif self.action_idx == 1:

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
            self._to_rotation(self.atoms, 9)

        elif self.action_idx == 3:
            self._to_rotation(self.atoms, -9)


        elif self.action_idx == 4:
            self.atoms.set_constraint(constraint)
            self.atoms.calc = EMT()
            dyn = Langevin(self.atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory=save_path_md,
                           logfile='MD.log')
            dyn.run(self.steps)

            '''------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------'''

        elif self.action_idx == 5:  # 表面上氧原子的扩散，单原子行为
            self.atoms, diffusable = self.to_diffuse_oxygen(self.atoms, self.total_s)

        elif self.action_idx == 6:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
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
        
                if selected_O in layer_O:
                    self.atoms = self.to_drill_surf(self.atoms)
                elif selected_O in sub_O:
                    self.atoms = self.to_drill_deep(self.atoms)

                self.to_constraint(self.atoms)
                self.atoms, _, _ = self.lasp_calc(self.atoms)
                self.atoms = self.to_expand_lattice(self.atoms, 0.8, 0.8, 10/11)
            else:
                reward -= 1

        elif self.action_idx == 7:  # 氧气解离
            self.atoms = self.O_dissociation(self.atoms)

        elif self.action_idx == 8:
            _,  desorblist = self.to_desorb_adsorbate(self.atoms)
            if desorblist:
                self.atoms = self.choose_ads_to_desorb(self.atoms)
            else:
                reward -= 1
            
        else:
            print('No such action')

        self.timestep += 1
        
        if action in [0, 2, 3, 5, 6]:
            self.recover_rotation(self.atoms, self.facet_selection)

        self.to_constraint(self.atoms)
   
        # 优化该state的末态结构以及next_state的初态结构
        self.atoms, current_energy, current_force = self.lasp_calc(self.atoms)

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

        if self.action_idx == 8:
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
      
        if self.action_idx in [1,2,3,5,6,7]: 
            barrier = self.check_TS(previous_atom, self.atoms, previous_energy, current_energy, self.action_idx)    # according to Boltzmann probablity distribution
            if barrier > 5:
                reward += -5.0 / (self.reaction_n * self.k * self.temperature_K)
                barrier = 5.0
            else:
                # reward += math.tanh(-relative_energy /(self.H * 8.314 * self.temperature_K)) * (math.pow(10.0, 5))
                reward += -relative_energy / (self.reaction_n * self.k * self.temperature_K)

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
        env_list = self.label_atoms(self.atoms, [23.33, 25.83])
        for i in self.atoms:    #查找是否Pd原子游离在环境中
            if i.index in env_list and i.symbol == 'Pd':
                env_Pd_list.append(i.index)
        
        exist_too_short_bonds = self.exist_too_short_bonds(self.atoms)

        if exist_too_short_bonds or env_Pd_list or self.energy - self.initial_energy > 4 or relative_energy > self.max_RE:
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
        if len(self.history['real_energies']) > 31:
            RMSE = self.RMSE(self.history['real_energies'][-30:])
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
        plt.xticks(fontsize=12, fontfamily='Arial', fontweight='bold')
        plt.yticks(fontsize=12, fontfamily='Arial', fontweight='bold')
        # plt.title('Epi_{}'.format(620 + episode), fontsize=28, fontweight='bold', fontfamily='Arial')
        plt.xlabel('steps(fs)', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')
        plt.ylabel('Energies(eV)', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')
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

        self.atoms = atoms.copy()

        self.to_constraint(self.atoms)
        self.atoms, self.initial_energy, self.initial_force= self.lasp_calc(self.atoms)

        self.action_idx = 0
        self.episode_reward = 0.5 * self.timesteps
        self.timestep = 0

        self.total_steps = self.timesteps
        self.max_RE = 3
        self.min_RE_d = self.convergence * self.len_atom
        self.repeat_action = 0

        self.n_O2 = 2000
        self.n_O3 = 0

        self.ads_list = []
        for _ in range(self.n_O2):
            self.ads_list.append(2)

        self.atoms = self.choose_ads_site(self.atoms)

        self.trajectories = []
        self.RMSD_list = []
        self.trajectories.append(self.atoms.copy())


        self.TS = {}
        # self.TS['structures'] = [slab.get_scaled_positions()[self.free_atoms, :]]
        self.TS['energies'] = [0.0]
        self.TS['timesteps'] = [0]

        self.adsorb_history = {}
        self.adsorb_history['traj'] = [atoms]
        self.adsorb_history['structure'] = [atoms.get_scaled_positions()[self.free_atoms, :].flatten()]
        self.adsorb_history['energy'] = [0.0]
        self.adsorb_history['timesteps'] = [0]

        results = ['energies', 'actions', 'structures', 'timesteps', 'forces', 'scaled_structures', 'real_energies', 'reward']
        for item in results:
            self.history[item] = []
        self.history['energies'] = [0.0]
        self.history['real_energies'] = [0.0]
        self.history['actions'] = [0]
        self.history['forces'] = [self.initial_force]
        self.history['structures'] = [atoms.get_positions().flatten()]
        self.history['scaled_structures'] = [atoms.get_scaled_positions()[self.free_atoms, :].flatten()]
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
        observation['force'] = self.force[self.free_atoms, :].flatten()
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
        layerlist = self.label_atoms(previous_atom, [16.0, 21.0])
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

    def choose_ads_site(self, state):
        new_state = state.copy()
        surf_sites = self.get_surf_sites(state)
        layerList = self.get_layer_atoms(new_state)
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
            # ads,  _ = self.to_ads_adsorbate(new_state)
            choosed_adsorbate = np.random.randint(len(self.ads_list))
            ads = self.ads_list[choosed_adsorbate]
            
            del self.ads_list[choosed_adsorbate]

            # delenvlist = [0, 1]
            # del env_s[[i for i in range(len(env_s)) if i in delenvlist]]
            if len(ads):
                if len(ads) == 2:
                    self.n_O2 -= 1
                    O1 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + 1.3))
                    O2 = Atom('O', (ads_site[0], ads_site[1], ads_site[2] + 2.51))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    # ads = O1 + O2
                elif len(ads) == 3:
                    self.n_O3 -= 1
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
                    self.ads_list.append(2)
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                elif len(desorb) == 3:
                    self.ads_list.append(3)
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
                    self.n_O2 += 1
                    O1 = Atom('O', (O_position[0], O_position[1], O_position[2] + 4.0))
                    O2 = Atom('O', (O_position[3], O_position[4], O_position[5] + 4.0))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    # ads = O1 + O2
                elif len(desorb) == 3:
                    self.n_O3 += 1
                    O1 = Atom('O', (O_position[0], O_position[1], O_position[2] + 4.0))
                    O2 = Atom('O', (O_position[3], O_position[4], O_position[5] + 4.0))
                    O3 = Atom('O', (O_position[6], O_position[7], O_position[8] + 4.0))
                    new_state = new_state + O1
                    new_state = new_state + O2
                    new_state = new_state + O3
        return new_state
    
    def _to_rotation(self, atoms, zeta):
        initial_state = atoms.copy()
        zeta = math.pi * zeta / 180
        matrix = [[cos(zeta), -sin(zeta), 0],
                      [sin(zeta), cos(zeta), 0],
                      [0, 0, 1]]
        matrix = np.array(matrix)

        rotation_list = []
        surf_list = self.get_surf_atoms(atoms)
        layer_list = self.get_layer_atoms(atoms)

        for i in surf_list:
            rotation_list.append(i)
        for j in layer_list:
            rotation_list.append(j)

        rotation_list = [i for n, i in enumerate(rotation_list) if i not in rotation_list[:n]]

        central_point = self.mid_point(atoms, surf_list)

        for atom in initial_state.positions:

            if atom.index in rotation_list:
                atom += np.array(
                        (np.dot(matrix, (np.array(atom.tolist()) - central_point).T).T + central_point).tolist()) - atom
        atoms.positions = initial_state.get_positions()
    
    def to_diffuse_oxygen(self, slab, facet):
        layer_O = []
        to_diffuse_O_list = []
        diffuse_sites = []
        layer_List = self.get_layer_atoms(slab)
        neigh_facet = self.neighbour_facet(slab, facet)
        for i in neigh_facet:
            new_state = self.cluster_rotation(new_state, i)
            list = self.get_surf_atoms(new_state)
            sites = self.get_surf_sites(atoms,list)
            for site in sites:
                diffuse_sites.append(site.tolist())
            new_state = self.recover_rotation(new_state, i)
        diffuse_sites = np.array(diffuse_sites)

        diffusable_sites = []
        interference_O_distance = []
        diffusable = True

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        for ads_sites in diffuse_sites:    # 寻找可以diffuse的位点
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
            diffuse_site = diffusable_sites[np.random.randint(len(diffusable_sites))]
            interference_O_list = [i for i in layer_O if i != selected_O_index]
            for j in interference_O_list:
                d = self.atom_to_traj_distance(slab.positions[selected_O_index], diffuse_site, slab.positions[j])
                interference_O_distance.append(d)
            if interference_O_distance:
                if min(interference_O_distance) < 0.3 * d_O_O:
                    diffusable = False
        
            if diffusable:
                for atom in slab:
                    if atom.index == selected_O_index:
                        atom.position = np.array([diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.5])
                # del slab[[j for j in range(len(slab)) if j == selected_O_index]]
                # O = Atom('O', (diffuse_site[0], diffuse_site[1], diffuse_site[2] + 1.5))
                # slab = slab + O
            
        return slab, diffusable

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
    
    def to_drill_surf(self, slab):
        layer_O = []
        to_distance = []
        drillable_sites = []
        layer_O_atom_list = []
        layer_OObond_list = []
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
            i = layer_O_atom_list[np.random.randint(len(layer_O_atom_list))]
            position = slab.get_positions()[i]
            del slab[[j for j in range(len(slab)) if j == i]]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_site = sub_sites[to_distance.index(min(to_distance))]
            # O = Atom('O', (drill_site[0], drill_site[1], drill_site[2] +1.3))
            # slab = slab + O
            for atom in slab:
                if atom.index == i:
                    atom.position = np.array([drill_site[0], drill_site[1], drill_site[2] +1.3])

            lifted_atoms_list = []
            current_surfList = self.get_surf_atoms(slab)
            c_layerList = self.get_layer_atoms(slab)
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
        sub_List = self.get_sub_atoms(slab)

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
            drill_site = deep_sites[to_distance.index(min(to_distance))]
            # O = Atom('O', (drill_site[0], drill_site[1], drill_site[2] +1.3))
            # slab = slab + O
            for atom in slab:
                if atom.index == i:
                    atom.position = np.array([drill_site[0], drill_site[1], drill_site[2] +1.3])


            lifted_atoms_list = []
            # current_surfList = self.label_atoms(slab, [surf_z - fluct_d_Pd/2, surf_z + fluct_d_Pd])
            '''current_sub_surfList = self.get_sub_atoms(slab)
            c_layerList = self.get_layer_atoms(slab)
            
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
        layerList = self.label_atoms(slab, [16.5,21.5])
        layer_O2_list = []
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O','O',unique = True)
        for i in OOBonds[0]:
            if i[0] in layerList and i[1] in layerList:
                layer_O2_list.append([(i[0],i[1])])
        if layer_O2_list:
            OO = layer_O2_list[np.random.randint(len(layer_O2_list))]
            d = ana.get_values([OO])[0]
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

    
    def RMSD(self, current_atoms, previous_atoms):
        similar = False
        constraint_p = self.get_constraint(previous_atoms)
        free_atoms_p = []
        for i in range(len(previous_atoms)):
            if i not in constraint_p.index:
                free_atoms_p.append(i)
        len_atom_p = len(free_atoms_p)

        constraint_c = self.get_constraint(current_atoms)
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
    
    def mid_point(self, slab, List):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for i in slab:
            if i.index in List:
                sum_x += slab.get_positions()[i.index][0]
                sum_x += slab.get_positions()[i.index][1]
                sum_z += slab.get_positions()[i.index][2]
        mid_point = [sum_x/len(List), sum_y/len(List), sum_z/len(List)]
        return mid_point
        
    
    def get_surf_sites(self, atoms):
        surfList = self.get_surf_atoms(atoms)

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]

        total_surf_sites = self.get_sites(surf)

        return total_surf_sites
    
    def get_sub_sites(self, atoms):
        subList = self.get_sub_atoms(atoms)

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if i not in subList]]

        total_sub_sites = self.get_sites(sub)
        return total_sub_sites
    
    def get_deep_sites(self, atoms):
        deepList = self.get_deep_atoms(atoms)

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if i not in deepList]]

        total_deep_sites = self.get_sites(deep)

        return total_deep_sites
    
    def get_sites(self, atoms):
        atop = atoms
        pos_ext = atoms
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

    def lasp_calc(self, atom):
        write_arc([atom])
        atom.calc = LASP(task='local-opt', pot='PdO', potential='NN D3')
        energy = atom.get_potential_energy()
        force = atom.get_forces()
        atom = read_arc('allstr.arc', index = -1)
        return atom, energy, force
    
    def to_constraint(self, atoms): # depending on such type of atoms
        surfList = []
        for facet in atoms.get_surfaces():
            atoms= self.cluster_rotation(atoms, facet)
            list = self.get_surf_atoms(atoms)
            for i in list:
                surfList.append(i)
            atoms = self.recover_rotation(atoms, facet)

        surfList = [i for n, i in enumerate(surfList) if i not in surfList[:n]]
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index not in surfList for a in atoms])
        fix = atoms.set_constraint(constraint)

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
        d = self.distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
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
        layerlist = self.label_atoms(slab,[16.5,21.5])

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

        # del slab[[i for i in range(len(slab)) if slab[i].index == atoms[0][0] or slab[i].index == atoms[0][1]]]

        # if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
            # O_1 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 1.3))
            # O_2 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 2.51))
        # else:

            # O_1 = Atom('O', (O1_site[0], O1_site[1], O1_site[2] + 1.3))
            # O_2 = Atom('O', (O2_site[0], O2_site[1], O2_site[2] + 1.3))

        # slab = slab + O_1
        # slab = slab + O_2
        for atom in slab:
            if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
                O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + 1.3])
                O_2_position = np.array([O1_site[0], O1_site[1], O1_site[2] + 2.51])
            else:
                O_1_position = np.array([O1_site[0], O1_site[1], O1_site[2] + 1.3])
                O_2_position = np.array([O2_site[0], O2_site[1], O2_site[2] + 1.3])

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
                    ads_list.append(i)

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
                    desorb_list.append(i)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
        return desorb, desorb_list
    
    def _2D_distance(self, x1,x2, y1,y2):
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        return dis
    
    def matrix_x(self, zeta):
        return np.array([[1, 0, 0],
                     [0, cos(zeta), sin(zeta)],
                      [0, -sin(zeta), cos(zeta)]
                      ])
    def matrix_y(self, zeta):
        return np.array([[cos(zeta), 0, -sin(zeta)],
                        [0, 1, 0],
                        [sin(zeta), 0, cos(zeta)]])
    def matrix_z(self, zeta):
        return np.array([[cos(zeta), -sin(zeta), 0],
                        [sin(zeta), cos(zeta), 0],
                        [0, 0, 1]])
    
    def cluster_rotation(self, atoms, facet):
        if facet[0] == 0 and facet[1] == 0 and facet[2] == 1:
            return atoms
        elif facet[0] == 0 and facet[1] == 0 and facet[2] == -1:
            zeta = math.acos(facet[2]/math.sqrt(facet[0] * facet[0] + facet[1] * facet[1]
                                                + facet[2] * facet[2]))
            facet = np.dot(self.matrix_y(zeta), facet)
            state = atoms.copy()
            for atom in state.positions:
                atom += np.array((np.dot(self.matrix_y(zeta),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state.get_positions()
            
            return atoms
        else:
            zeta_1 = math.acos(facet[0]/math.sqrt(facet[0] * facet[0] +
                                                        facet[1] * facet[1]))
            state_1 = atoms.copy()
            if facet[1] > 0:
                facet_1 = np.dot(self.matrix_z(-zeta_1), facet)
                
                for atom in state_1.positions:
                    atom += np.array((np.dot(self.matrix_z(-zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
                
            else:
                facet_1 = np.dot(self.matrix_z(zeta_1), facet)
                for atom in state_1.positions:
                    atom += np.array((np.dot(self.matrix_z(zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            
            atoms.positions = state_1.get_positions()
            zeta_2 = math.acos(facet_1[2]/math.sqrt(facet_1[0] * facet_1[0] +
                                                    facet_1[2] * facet_1[2]))
            facet_2 = np.dot(self.matrix_y(zeta_2), facet_1)
            state_2 = atoms.copy()
            for atom in state_2.positions:
                atom += np.array((np.dot(self.matrix_y(zeta_2),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_2.get_positions()
            return atoms

    def recover_rotation(self, atoms, facet):
        if facet[0] == 0 and facet[1] == 0 and facet[2] == 1:
            return atoms
        elif facet[0] == 0 and facet[1] == 0 and facet[2] == -1:
            zeta = math.acos(facet[2]/math.sqrt(facet[0] * facet[0] + facet[1] * facet[1]
                                                + facet[2] * facet[2]))
            facet = np.dot(self.matrix_y(-zeta), facet)
            state = atoms.copy()
            for atom in state.positions:
                atom += np.array((np.dot(self.matrix_y(-zeta),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state.get_positions()
                
            return atoms
        else:
            zeta_1 = math.acos(facet[0]/math.sqrt(facet[0] * facet[0] +
                                                    facet[1] * facet[1]))

            if facet[1] > 0:
                facet_1 = np.dot(self.matrix_z(-zeta_1), facet)
            else:
                facet_1 = np.dot(self.matrix_z(zeta_1), facet)
                
            zeta_2 = math.acos(facet_1[2]/math.sqrt(facet_1[0] * facet_1[0] +
                                                    facet_1[2] * facet_1[2]))
            facet_2 = np.dot(self.matrix_y(zeta_2), facet_1)

            state_1 = atoms.copy()
            for atom in state_1.positions:
                atom += np.array((np.dot(self.matrix_y(-zeta_2),
                                        np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_1.get_positions()

            state_2 = atoms.copy()
            if facet[1] > 0:
                for atom in state_2.positions:
                    atom += np.array((np.dot(self.matrix_z(zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            else:
                for atom in state_2.positions:
                    atom += np.array((np.dot(self.matrix_z(-zeta_1),
                                            np.array(atom.tolist()).T).T).tolist()) - atom
            atoms.positions = state_2.get_positions()

            return atoms

    def get_layer_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == 'Pd':
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        list = self.label_atoms(atoms, [z_max, z_max + 6.0])
        layerlist = []
        for i in list:
            if atoms[i].symbol == 'Pd':
                layerlist.append(i)

        return layerlist
    
    def get_surf_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == 'Pd':
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        list = self.label_atoms(atoms, [z_max - 1.0, z_max + 1.0])
        surflist = []
        for i in list:
            if atoms[i].symbol == 'Pd':
                surflist.append(i)

        return surflist
    
    def get_sub_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == 'Pd':
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        list = self.label_atoms(atoms, [z_max - 3.0, z_max - 1.0])
        sublist = []
        for i in list:
            if atoms[i].symbol == 'Pd':
                sublist.append(i)

        return sublist
    
    def get_deep_atoms(self, atoms):
        z_list = []
        for i in range(len(atoms)):
            if atoms[i].symbol == 'Pd':
                z_list.append(atoms.get_positions()[i][2])
        z_max = max(z_list)

        list = self.label_atoms(atoms, [z_max - 5.0, z_max - 3.0])
        deeplist = []
        for i in list:
            if atoms[i].symbol == 'Pd':
                deeplist.append(i)

        return deeplist

    def neighbour_facet(self, atoms, facet):
        atoms= self.cluster_rotation(atoms, facet)
        surface_list = self.get_surf_atoms(atoms)
        atoms = self.recover_rotation(atoms, facet)
        neighbour_facet = []
        neighbour_facet.append(facet)
        for selected_facet in atoms.get_surfaces():
            if selected_facet.tolist() != facet:
                atoms = self.cluster_rotation(atoms, selected_facet)
                selected_surface_list = self.get_surf_atoms(atoms)
                atoms = self.recover_rotation(atoms, selected_facet)
                repeat_atoms = [i for i in selected_surface_list if i in surface_list]
                if len(repeat_atoms) >= 2:
                    neighbour_facet.append(selected_facet.tolist())
        return neighbour_facet

    def get_constraint(self, atoms):
        surfList = []
        for facet in atoms.get_surfaces():
            atoms= self.cluster_rotation(atoms, facet)
            list = self.get_surf_atoms(atoms)
            for i in list:
                surfList.append(i)
            atoms = self.recover_rotation(atoms, facet)

        surfList = [i for n, i in enumerate(surfList) if i not in surfList[:n]]
        constraint = FixAtoms(mask=[a.symbol != 'O' and a.index not in surfList for a in atoms])
        return constraint
        
    def layer_O_atom_list(self, slab):
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
        sub_List = self.get_sub_atoms(slab)

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


