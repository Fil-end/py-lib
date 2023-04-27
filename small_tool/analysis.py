import numpy as np
import time
data = np.load('./217.npz', allow_pickle = True)
print(data.files)
# print(data['structures'][1].reshape(145,3))
structures = data['structures']

def add_text_head(filename):
    with open(filename, 'r+') as f:
            context = f.read()
            f.seek(0)
            f.write(f"!BIOSYM archive 2\nPBC=ON\nASE Generated CAR File MS\n")
            f.write(context)

with open('allstr_217.arc', 'w') as fd:
    a, b, c, alpha, beta, gamma = 16.50387227, 16.50387227, 25.83500000, 90.0, 90.0, 90.0
    for atoms in structures:
        fd.write(f"!DATE {time.asctime()}\n")
        fd.write("%s" % "PBC" + 
                 "%14.8f" % (a) + 
                 "%14.8f" % (b) + 
                 "%14.8f" % (c) +
                 "%14.8f" % (alpha) + 
                 "%14.8f" % (beta) +
                 "%14.8f" % (gamma) +'\n')
        '''if j ==0:
            for atom in atoms:
                i = i + 1
                if i < 114:
                    symbol = 'Pd'
                else:
                    symbol = 'O'
                fd.write("%-4s" % symbol +
                        "%15.8f" % (atom[0]) +
                            "%15.8f" % (atom[1]) +
                            "%15.8f" % (atom[2]) + 
                            "%5s" % "CORE" +
                            "%5d" % (i) +
                            "%4s" % (symbol) + 
                            "%4s" % (symbol) +  
                            "%10.4f" % 0.0000 +   
                            "%5d" % (i + 1) + 
                            '\n')
            fd.write('end\nend\n')
        else:'''
        atoms = np.array(atoms)
        i = 0
        for atom in atoms.reshape(156,3):
            i = i + 1
            
            if i < 125:
                symbol = 'Pd'
            else:
                symbol = 'O'

            if atom[2] < 5.0:
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

add_text_head(filename='allstr_217.arc')