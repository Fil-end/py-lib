# from GDPy.computation import lasp
import os
from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from pathlib import Path
import random


'''train_dir = os.getcwd() + "/Trainstr.txt"
# print(train_dir.parent)
lasp.read_laspset(train_dir)'''
def read_laspset(train_structures, nequip_read_method = None, split = None, shrinked = True):
    if nequip_read_method not in ['random', 'sequential']:
        raise ValueError('please use the read method in [random, sequential]<----[str, str]')
    """Read LASP TrainStr.txt and TrainFor.txt files."""
    train_structures = Path(train_structures)
    frames = []

    all_energies, all_forces, all_stresses = [], [], []
    num_list = []
    with open(train_structures, "r") as r:
        lines = r.readlines()
        with open(train_structures,"r+") as w:
            for l in lines:
                if "weight" not in l:
                    w.write(l)

    # - TrainStr.txt
    # TODO: use yield
    with open(train_structures, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - energy
                line = fopen.readline()
                energy = float(line.strip().split()[-2])
                all_energies.append(energy)
                # - natoms
                line = fopen.readline()
                natoms = int(line.strip().split()[-1])
                # skip 5 lines, symbol info and training weights
                skipped_lines = [fopen.readline() for _ in range(4)]
                # - cell
                cell = np.array([fopen.readline().strip().split()[1:] for i in range(3)], dtype=float)
                # - symbols, positions, and charges
                anumbers, positions, charges = [], [], []
                for i in range(natoms):
                    data = fopen.readline().strip().split()[1:]
                    anumbers.append(int(data[0]))
                    positions.append([float(x) for x in data[1:4]])
                    charges.append(float(data[-1]))
                atoms = Atoms(numbers=anumbers, positions=positions, cell=cell, pbc=True)
                assert fopen.readline().strip().startswith("End one structure")
                frames.append(atoms)
                #break
            if not line:
                break
    
    # - TrainFor.txt
    train_forces = train_structures.parent / "TrainFor.txt"
    with open(train_forces, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - stress, voigt order
                stress = np.array(fopen.readline().strip().split()[1:], dtype=float)
                # - symbols, forces
                anumbers, forces = [], []
                line = fopen.readline()
                while True:
                    if line.strip().startswith("force"):
                        data = line.strip().split()[1:]
                        anumbers.append(int(data[0]))
                        forces.append([float(x) for x in data[1:4]])
                    else:
                        all_forces.append(forces)
                        assert line.strip().startswith("End one structure")
                        break
                    line = fopen.readline()
                #break
            if not line:
                break

    if nequip_read_method == 'sequential':
        train_frames, validation_frames, shrinked_frames = [], [], []
        train_energies, train_forces, train_stresses = [], [], []
        validation_energies, validation_forces, validation_stresses = [], [], []
        shrinked_energies, shrinked_forces, shrinked_stresses = [], [], []
        
        # 得到所有的shrink和不shrink的结构、能量、力信息
        for i in range(len(all_energies)):
            if all_energies[i] >= 0:
                train_frames.append(frames[i])
                train_energies.append(all_energies[i])
                train_forces.append(all_forces[i])
                shrinked_frames.append(frames[i])
                shrinked_energies.append(all_energies[i])
                shrinked_forces.append(all_forces[i])

            else:
                if np.random.random() > 0.95:
                    validation_frames.append(frames[i])
                    validation_energies.append(all_energies[i])
                    validation_forces.append(all_forces[i])
                else:
                    train_frames.append(frames[i])
                    train_energies.append(all_energies[i])
                    train_forces.append(all_forces[i])
                    
        validation_shrinked_frames = random.sample(shrinked_frames, int(len(shrinked_frames)/10))

        for frame in validation_shrinked_frames:
            validation_frames.append(frame)
            validation_energies.append(shrinked_energies[validation_shrinked_frames.index(frame)])
            validation_forces.append(shrinked_forces[validation_shrinked_frames.index(frame)])

        frames = []
        all_energies, all_forces = [], []
        for i in range(len(train_frames)):
            frames.append(train_frames[i])
            all_energies.append(train_energies[i])
            all_forces.append(train_forces[i])

        for i in range(len(validation_frames)):
            frames.append(validation_frames[i])
            all_energies.append(validation_energies[i])
            all_forces.append(validation_forces[i])

        num_list = [len(train_frames),len(train_energies), len(train_forces),
                    len(validation_frames),len(validation_energies), len(validation_forces),
                    len(frames), len(all_energies), len(all_forces)]

    elif nequip_read_method == 'random':
        num_list = [len(frames)]


    if split and nequip_read_method == 'sequential':
        for i, atoms in enumerate(train_frames):
            calc = SinglePointCalculator(
                atoms, energy=train_energies[i], forces=train_forces[i]
            )
            atoms.calc = calc
        write(train_structures.parent / "trainset.xyz", train_frames)

        for i, atoms in enumerate(validation_frames):
            calc = SinglePointCalculator(
                atoms, energy=validation_energies[i], forces=validation_forces[i]
            )
            atoms.calc = calc
        write(train_structures.parent / "validationset.xyz", validation_frames)
    else:
        for i, atoms in enumerate(frames):
            calc = SinglePointCalculator(
                atoms, energy=all_energies[i], forces=all_forces[i]
            )
            atoms.calc = calc
        write(train_structures.parent / "testset.xyz", frames)

    return frames, all_energies, all_forces, all_stresses, num_list

train_dir = os.getcwd() + "/non_shrink" + "/TrainStr.txt"
frames, all_energies, all_forces, all_stresses, num_list = read_laspset(train_dir,
                                                                        'sequential', split = False, shrinked = False)
