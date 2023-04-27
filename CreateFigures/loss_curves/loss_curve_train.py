import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

def running_min(df):
    minimums = np.zeros(len(df.index))
    changes = np.zeros_like(minimums)
    changes[0] = 1

    for index, row in df.iterrows():
        minimums[index] = df['val_loss'].iloc[0:index+1].min()
        
    for i in range(1, len(minimums)):
        change = (minimums[i] != minimums[i-1])
        changes[i] = 1 if change else 0

    return minimums, changes




# PATH_FILE2 = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/histories/weights_15022058.log"

PATH_FILE = "/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/outputs/histories/weights_09021654.log"

file = pd.read_table(PATH_FILE, sep = ',', index_col=0)

file_mins, lr_changes = running_min(file)

# file2 = pd.read_table(PATH_FILE2, sep = ',', index_col=0)

# file2_mins, lr2_changes = running_min(file2)

sns.set_theme()

ax = file['loss'].plot(label='Training loss', zorder = 1)
file['val_loss'].plot(ax = ax, label='Validation loss', zorder = 3)
ax.plot(file_mins, label="Current minimum Validation loss", zorder = 2)
ax.set_title('Model train/val loss')
ax.set_ylabel('Loss')

ax.legend()

print(lr_changes[10:].sum())

# plt.show()
# plt.savefig('loss_big_data_t2m.png')
plt.savefig('/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/loss_curves/loss_gs_middle_left.png')
