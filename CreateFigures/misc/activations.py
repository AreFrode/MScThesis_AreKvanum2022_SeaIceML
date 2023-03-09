import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)


x = np.linspace(-6, 6, int(1e5))

sns.set_theme(context='talk', style='white')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('bottom')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,sigmoid(x))
plt.savefig('sigmoid.png')