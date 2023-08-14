import numpy as np
import time
import os

'''def file_name_walk(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return root, dirs, files

_, _, files = file_name_walk('./')
'''
def getFileName1(path,suffix):
    # 获取指定目录下的所有指定后缀的文件名 
    input_template_All=[]
    f_list = os.listdir(path)#返回文件名
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] ==suffix:
            input_template_All.append(i)
            #print(i)
    return input_template_All

files = getFileName1('./', '.npz')

lowest_energy_file = None
lowest_energy = 0
lowest_energy_epi_re = 0

highest_epi_re = 0
highest_epi_re_energy_file = None
highest_epi_re_energy = 0

for file in files:
    data = np.load(file, allow_pickle = True)
    # energy_list.append[(file, str(data['energies'][-1]))]
    
    f=open("energy.txt","a")
    f.write(str(file) + str(data['energies'][-1]) + ' ' + str(data['episode_reward']) + '\n')
    f.close()

    if data['energies'][-1] < lowest_energy:
        lowest_energy = data['energies'][-1]
        lowest_energy_file = file
        lowest_energy_epi_re = data['episode_reward']
    
    if data['episode_reward'] > highest_epi_re:
        highest_epi_re = data['episode_reward']
        highest_epi_re_energy = data['energies'][-1]
        highest_epi_re_energy_file = file

f=open("energy.txt","a")
f.write('\n' + 
        'Lowest energy file and lowest energy:' + 
        str(lowest_energy_file) + str(lowest_energy) + ' ' + str(lowest_energy_epi_re) + '\n' + 
        'Highest reward file and its corresponding energy:' + 
        str(highest_epi_re_energy_file) + str(highest_epi_re_energy) + ' ' + str(highest_epi_re) + '\n')
f.close()
