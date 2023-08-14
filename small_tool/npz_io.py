import numpy as np
import time
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

def pad_array2list(array, max_len, position = True):
    if position:
        array = np.append(array, [0.0, 0.0, 0.0] * (max_len - array.shape[0]))
        array = array.reshape(int(array.shape[0]/3), 3)
    else:
        array = np.append(array, ['H'] * (max_len - array.shape[0]))
    return array.tolist()

def add_text_head(filename):
    with open(filename, 'r+') as f:
            context = f.read()
            f.seek(0)
            f.write(f"!BIOSYM archive 2\nPBC=ON\nASE Generated CAR File MS\n")
            f.write(context)



def npz2arc(file_num, structures):
    with open('allstr_{}.arc'.format(file_num), 'w') as fd:
        a, b, c, alpha, beta, gamma = 11.05, 11.05, 25.83500000, 90.0, 90.0, 60.0
        for atoms in structures:
            fd.write(f"!DATE {time.asctime()}\n")
            fd.write("%s" % "PBC" + 
                    "%14.8f" % (a) + 
                    "%14.8f" % (b) + 
                    "%14.8f" % (c) +
                    "%14.8f" % (alpha) + 
                    "%14.8f" % (beta) +
                    "%14.8f" % (gamma) +'\n')
        
            atoms = np.array(atoms)
            i = 0
            for atom in atoms.reshape(int(len(structures[0])/3),3):
                i = i + 1
                
                if i < 125:
                    symbol = 'Pd'
                else:
                    symbol = 'O'

                if atom[2] < 9.0:
                    atom[0] = 0
                    atom[1] = 0
                    atom[2] = 0
                fd.write("%-4s" % symbol +
                                "%15.8f" % (atom[0]) +
                                    "%15.8f" % (atom[1]) +
                                    "%15.8f" % (atom[2]) + 
                                    "%5s" % "CORE" +
                                    "%5d" % (i) +
                                    "%4s" % (symbol) + 
                                    "%4s" % (symbol) +  
                                    "%10.4f" % 0.0000 +   
                                    "%5d" % (i) + 
                                    '\n')
            fd.write('end\nend\n')

def npz2xyz(file_num, structures, energies, element_list, force = False):
    cell = np.array([[11.05, 0, 0],
                    [0, 11.05, 0],
                    [0, 0, 25.835]])
    
    frames = []

    num_list = []
    asymbols_list = []
    positions_list = []

    for i in range(len(structures)):
        natoms = structures[i]
        elements = element_list[i]

        natoms = natoms.reshape(int(len(natoms)/3), 3)
        asymbols, positions = [], []

        for atom in natoms:
            if atom[0] != 0.0 or atom[1] != 0.0 or atom[2] != 0.0:
                positions.append(atom.tolist())

        for ele in elements:
            if ele != '0.0':
                asymbols.append(ele)

        asymbols_list.append(asymbols)
        positions_list.append(positions)
        num_list.append(len(positions))

    max_len = max(num_list)

    for j in range(len(positions_list)):
        positions = np.array(positions_list[j])
        asymbols = np.array(asymbols_list[j])

        positions = pad_array2list(positions, max_len, position = True)
        asymbols = pad_array2list(asymbols, max_len, position = False)

        atoms = Atoms(symbols = asymbols, positions=positions, cell=cell, pbc=[True, True, False])
        frames.append(atoms)

    print(len(positions_list), max_len)

    for i, atoms in enumerate(frames):
        calc = SinglePointCalculator(
            atoms, energy=energies[i]
        )
        atoms.calc = calc

    write("./allstr_{}.xyz".format(file_num), frames)


if __name__ == '__main__':
    file_num = 82
    data = np.load('./{}.npz'.format(file_num), allow_pickle = True)
    print(data.files)

    structures = data['structures']
    energies = data['energies']
    element_list = data['element_list']

    '''positions = []
    for atom in structures[0].reshape(int(len(structures[0])/3), 3):
        if atom[0] != 0.0 or atom[1] != 0.0 and atom[3] != 0.0:
            positions.append(atom.tolist())

    print(positions)'''

    # npz2arc(file_num, structures)
    npz2xyz(file_num, structures, energies, element_list)

    # add_text_head(filename='allstr_{}.arc'.format(file_num))