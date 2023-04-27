import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dpi = 150
fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)

def update(num):
    x = num
    line.set_xdata((x, x))
    return line,
    
episode = 217
data = np.load('./%d.npz' % episode, allow_pickle = True)

ACTION_SPACES = ['ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation']

save_path = os.path.join('./%d.png' % episode)

energies = np.array(data['energies'])
actions = np.array(data['actions'])


'''fig = plt.figure(figsize=(100, 100))
ax = fig.subplots(sharex=True, sharey=True)'''

line = ax.axvline(x=0, color='r')
plt.xticks(fontsize=12, fontfamily='Arial', fontweight='bold')
plt.yticks(fontsize=12, fontfamily='Arial', fontweight='bold')
plt.title('Epi_{}'.format(620 + episode), fontsize=28, fontweight='bold', fontfamily='Arial')
plt.xlabel('steps(fs)', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')
plt.ylabel('Energies(eV)', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')
ax.plot(energies, color='blue')

'''for action_index in range(len(ACTION_SPACES)):
    action_time = np.where(actions == action_index)[0]
    ax.plot(action_time, energies[action_time], 'o',
                label=ACTION_SPACES[action_index])
    plt.xticks(fontsize=15, fontfamily='Arial', fontweight='bold')
    plt.yticks(fontsize=15, fontfamily='Arial', fontweight='bold')'''

ax.scatter(data['ts_timesteps'], data['ts_energy'], label='TS', marker='x', color='g', s=80)
# ax.scatter(data['adsorb_timesteps'], data['adsorb_energy'], label='ADS', marker='p', color='black', s=80)
ax.legend(loc='upper right', fontsize = 24)

ani = FuncAnimation(fig, update, frames=np.arange(0, len(data['actions']), 1), interval=188)
ani.save('animation.gif', writer='imagemagick')

# plt.savefig(save_path, bbox_inches='tight')