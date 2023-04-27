import os
import time
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.io import write
import numpy as np


def read_arc(fileobj, index='a', pbc=True):
    ''' fileobj is the *.arc filename such as input.arc, uncm.arc, allstr.arc
    when filetype=0, the fileobj must be single structure file
    when your structure file is a multi-structures file ,index= 'a' means return a Atoms List, and index=int means return the atoms[index]
    '''

    atoms = []
    Cell = []
    with open(fileobj, 'r') as fd:
        lines = fd.readlines()
    end_order = []
    PBC_order = []
    for order, line in enumerate(lines):
        Line = line.split()
        if 'end' in Line:
            end_order.append(order)
        if 'PBC' in Line:
            PBC_order.append(order)
        if len(Line) == 7:
            if float(Line[-1]) > 90.0:
                Line[-1] = str(180.0 - float(Line[-1]))
            Cell.append(Line[1:])
        

    end_order = end_order[::2]
    for i in range(len(end_order)):
        start = PBC_order[i] + 1
        end = end_order[i] - 1
        positions = []
        symbols = []
        for order in lines[start:end+1]:
            line = order.split()
            positions.append(line[1:4])
            symbols.append(line[0])
        atoms.append(Atoms(positions=positions, symbols=symbols, pbc=pbc, cell=Cell[i]))
    if index == 'a':
        return atoms
    else:
        return atoms[index]



def write_arc(structures, **params):
    
    
    if len(structures) == 0:
        print("There is no atoms!")

    elif len(structures) == 1:
        atoms = structures[0]
        write_single(atoms, write_type='w')
        add_text_head(filename="input.arc")

    else:
        write_single(structures[0], write_type='w')
        for i in range(1, len(structures)):
            write_single(structures[i], write_type='a')

        add_text_head(filename="input.arc")
        if len(structures) == 2:
            os.rename('input.arc', 'uncm.arc')

        else:
            os.rename('input.arc', 'allstr.arc')
        
def write_single(atoms, write_type='w'):

    with open('input.arc', write_type) as fd:
        if atoms.pbc.any() == np.array([False, False, False]).any():
            a, b, c, alpha, beta, gamma = 15.00, 15.00, 15.00, 90.00, 90.00, 90.00   #The lattice constant is preset for the aperiodic isolated system
        else:
            Cell_params = atoms.cell
            a = eval("{:.8f}".format(Cell_params.lengths()[0]))
            b = eval("{:.8f}".format(Cell_params.lengths()[1]))
            c = eval("{:.8f}".format(Cell_params.lengths()[2]))
            alpha = eval("{:.8f}".format(Cell_params.angles()[0]))
            beta = eval("{:.8f}".format(Cell_params.angles()[1]))
            gamma = eval("{:.8f}".format(Cell_params.angles()[2]))

        fd.write("%s" % "PBC" + 
                 "%14.8f" % (a) + 
                 "%14.8f" % (b) + 
                 "%14.8f" % (c) +
                 "%14.8f" % (alpha) + 
                 "%14.8f" % (beta) +
                 "%14.8f" % (gamma) +'\n')
        atom_type = []
        for atom in atoms:
            count = 0
            atom_type.append(atom.symbol)
            for i in atom_type:
                if i == atom.symbol:
                    count += 1
            symbol = atom.symbol
            fd.write("%-4s" % symbol +
                        "%15.8f" % (atom.position[0]) +
                        "%15.8f" % (atom.position[1]) +
                        "%15.8f" % (atom.position[2]) + 
                        "%5s" % "CORE" +
                        "%5d" % (atom.index + 1) +
                        "%4s" % (atom.symbol) + 
                        "%4s" % (atom.symbol) +  
                        "%10.4f" % 0.0000 +   
                        "%5d" % (atom.index + 1) + 
                        '\n')
        fd.write('end\nend\n')

def add_text_head(filename):
    with open(filename, 'r+') as f:
            context = f.read()
            f.seek(0)
            f.write(f"!BIOSYM archive 2\nPBC=ON\nASE Generated CAR File MS\n!DATE {time.asctime()}\n")
            f.write(context)
