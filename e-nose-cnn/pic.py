import matplotlib.pyplot as plt
import numpy as np

X=np.linspace(-10,10,1000,endpoint=True)

f = np.select([X>=0, X<0], [X, 0])
plt.plot(X,f)

f1 = np.select([X+3>=0, X+3<0], [X*(X+3)/6, 0])
plt.plot(X,f1)

f2 = np.select([X+3>=0, X+3<0], [X*(X+3)/4, 0])
plt.plot(X,f2)


plt.show()