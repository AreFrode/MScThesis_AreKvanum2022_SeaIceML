import numpy as np

from matplotlib import colors as mcolors

class cm:
    @staticmethod
    def sea_ice_chart():
        cmap = np.zeros((7,4))
        cmap[:, -1] = 1.
        
        cmap[0, :-1] = np.array([255., 255., 255.])/255.
        cmap[1, :-1] = np.array([150., 200., 255.])/255.
        cmap[2, :-1] = np.array([140., 255., 160.])/255.
        cmap[3, :-1] = np.array([255., 255., 0.])/255.
        cmap[4, :-1] = np.array([255., 125., 7.])/255.
        cmap[5, :-1] = np.array([255., 0., 0.])/255.
        cmap[6, :-1] = np.array([150., 150., 150.])/255.
    
        return mcolors.ListedColormap(cmap)

    @staticmethod
    def land():
        cmap = np.zeros((2,4))
        cmap[:, -1] = 1.

        cmap[0, :-1] = np.array([196., 196., 196.])/255.

        return mcolors.ListedColormap(cmap)
