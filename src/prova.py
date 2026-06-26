import interferometer as inf
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

U = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])
Interf = inf.square_decomposition(U)
Interf.draw()