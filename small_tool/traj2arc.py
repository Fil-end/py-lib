import numpy as np
import time
from ase.io import Trajectory

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


def traj2arc():
    traj = Trajectory('lbfgs_opt.traj')

    def add_text_head(filename):
        with open(filename, 'r+') as f:
                context = f.read()
                f.seek(0)
                f.write(f"!BIOSYM archive 2\nPBC=ON\nASE Generated CAR File MS\n")
                f.write(context)

    with open('allstr.arc', 'w') as fd:
        
        # a, b, c, alpha, beta, gamma = 16.50387227, 16.50387227, 25.83500000, 90.0, 90.0, 90.0
        for atoms in traj:
            if atoms.cell.cellpar()[0]:
                a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            else:
                a, b, c, alpha, beta, gamma = 16.50387227, 16.50387227, 25.83500000, 90.0, 90.0, 90.0 
            fd.write(f"!DATE {time.asctime()}\n")
            fd.write("%s" % "PBC" + 
                    "%14.8f" % (a) + 
                    "%14.8f" % (b) + 
                    "%14.8f" % (c) +
                    "%14.8f" % (alpha) + 
                    "%14.8f" % (beta) +
                    "%14.8f" % (gamma) +'\n')
            atoms = np.array(atoms)

            for atom in atoms:
                i = atom.index
                symbol = [k for k, v in Eledict.items() if v == atom.number][0]
                position = atom.position
                fd.write("%-4s" % symbol +
                                "%15.8f" % (position[0]) +
                                    "%15.8f" % (position[1]) +
                                    "%15.8f" % (position[2]) + 
                                    "%5s" % "CORE" +
                                    "%5d" % (i) +
                                    "%4s" % (symbol) + 
                                    "%4s" % (symbol) +  
                                    "%10.4f" % 0.0000 +   
                                    "%5d" % (i) + 
                                    '\n')
            fd.write('end\nend\n')

    add_text_head(filename='allstr.arc')


if __name__ == '__main__':
    traj2arc()