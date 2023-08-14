import numpy as np
import os

energies = []
positions = []
forces = []
atom_numbers = []

Trainstr = './Trainstr.txt'
TrainFor = './TrainFor.txt'

with open(Trainstr,'r') as fr:
	line = fr.readline()
	position = []
	atom_number = []
	pattern_1 = 'Energy'
	pattern_2 = 'ele'
	pattern_3 = 'End'
	while line:
		if line.split():
			if line.split()[0]==pattern_1:
				energies.append([line.split()[2]])
			elif line.split()[0]==pattern_2:
				atom_number.append(line.split()[1])
				position.append([line.split()[2],line.split()[3],line.split()[4]])
			elif line.split()[0]==pattern_3:
				positions.append(position)
				atom_numbers.append(atom_number)
				position = []
				atom_number = []
		line = fr.readline()
		
with open(TrainFor,'r') as fr:
	line = fr.readline()
	force=[]
	pattern_1 = 'force'
	pattern_2 = 'End'
	while line:
		if line.split():
			if line.split()[0]==pattern_1:
				force.append([line.split()[2],line.split()[3],line.split()[4]])
			if line.split()[0]==pattern_2:
				forces.append(force)
				force = []
		line = fr.readline()
		
energies = np.array(energies)
forces = np.array(forces)
positions = np.array(positions)
atom_numbers = np.array(atom_numbers)

z = np.zeros([len(atom_numbers),len(max(atom_numbers,key=lambda x:len(x)))])
for i,j in enumerate(atom_numbers):
	z[i][0:len(j)]=j

R = np.zeros([len(positions),len(max(positions,key=lambda x:len(x))),3])
for i,j in enumerate(positions):
	R[i][0:len(j)]=j

F = np.zeros([len(forces),len(max(forces,key=lambda x:len(x))),3])
for i,j in enumerate(forces):
	F[i][0:len(j)]=j

energies = energies.astype(float)
'''z = z.astype(float)
R = R.astype(float)
F = F.astype(float)
'''

np.savez('PdO.npz', E=energies, F=F, R=R, z=z)
