import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt


PATH_FILE = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/iiee_val/train.csv"
# sns.set_theme(context='paper')
sns.set_theme()
file = pd.read_csv(PATH_FILE,index_col=0)


fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(data = file, x = file.index, y = 'val_loss', label='Validation loss', zorder = 1)
ax2 = ax.twinx()
sns.lineplot(data = file, x = file.index, y = 'val_norm_iiee', ax = ax2, label='Validation ice edge displacement', zorder = 2, c='orange')

ax.set_title('Model train/val loss')
ax.set_ylabel('Loss')

ax.get_legend().remove()
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2)


# plt.show()
plt.savefig('val_loss_iiee.pdf')

print(file['val_loss'].corr(file['val_norm_iiee']))