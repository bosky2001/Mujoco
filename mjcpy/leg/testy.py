import matplotlib.pyplot as plt
import numpy as np



y= np.arange(0,10,1)

x = np.ones((10,10))

plt.scatter(y,x[1,:])
plt.scatter(y,x[2,:])
plt.show()