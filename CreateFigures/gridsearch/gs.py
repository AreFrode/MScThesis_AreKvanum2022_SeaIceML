import numpy as np
import seaborn as sns

from cmcrameri import cm
from matplotlib import pyplot as plt


data = np.array([[0.39633, 0.41150, 0.39658],
                 [0.35737, 0.35720, 0.36842],
                 [0.37369, 0.42806, 0.43243]])

sns.set_theme(context = 'paper')
ax = sns.heatmap(data = data, 
            cmap=cm.lajolla,
            annot = True, 
            fmt = ".5f", 
            cbar = False, 
            square = True,
            xticklabels = ['256', '512', '1024'],
            yticklabels = ['0.01', '0.001', '0.0001'])

ax.set_title('U-Net Grid Search')
ax.set_xlabel('U-Net depth [# Channels]')
ax.set_ylabel('Learning rate')

plt.savefig('grid_search.png')
