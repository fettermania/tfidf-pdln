from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_result_surface(z_label, result_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x = np.arange(0, THRESHOLD_MAX, THRESHOLD_MAX / THRESHOLD_POINTS)
  y = np.arange(0, SLOPE_MAX, SLOPE_MAX / SLOPE_POINTS)
  X, Y = np.meshgrid(x, y)
  zs = np.array([result_matrix[int(x * THRESHOLD_POINTS / THRESHOLD_MAX)][int(y * SLOPE_POINTS / SLOPE_MAX)] for x,y in zip(np.ravel(X), np.ravel(Y))])
  Z = zs.reshape(X.shape)
  surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.set_xlabel('Threshold')
  ax.set_ylabel('Slope')
  ax.set_zlabel(z_label)
  ax.set_title('PDLN ' + z_label + ' vs. hyperparams')
  ax.set_zlim(np.min(result_matrix), np.max(result_matrix))
  fig.colorbar(surface, shrink=0.5, aspect=5)
  plt.show()
