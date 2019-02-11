import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt

x0 = [1,0,0,1,0,0]
x1 = [1,0,0,0,0,0]

def absolute_loss(x0, x1):
	x0 = np.array(x0, dtype=np.float32)
	x1 = np.array(x1, dtype=np.float32)
	loss = F.absolute_error(x0, x1)
	plt.plot(loss.data)
	plt.show()
	print loss

def bernoulli_nll(x, y):
	x = np.array(x, dtype=np.float32)
	y = np.array(y, dtype=np.float32)
	loss = F.bernoulli_nll(x,y,reduce='no')
	plt.plot(loss.data)
	plt.show()
	print loss


#absolute_loss(x0, x1)
#bernoulli_nll(x0, x1)

