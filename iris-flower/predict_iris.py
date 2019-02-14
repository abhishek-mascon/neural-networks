import pandas as pd
import numpy as np
import six
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import chainer
import chainer.links as L
import chainer.functions as F


def prepare_data():
	np.random.seed(3)
	dataframe = pd.read_csv('IRIS.csv',header=None)
	dataset = dataframe.values
	X = dataset[1:,0:4].astype(float)
	Y = dataset[1:,4]
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	X, encoded_Y = shuffle(X, encoded_Y)
	X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y,test_size=0.30, random_state=7)
	return X_train, X_test, y_train, y_test
	#one hot encoding
	#dummy_Y = pd.get_dummies(encoded_Y).values.astype(float)
	

class Iris_Network(chainer.Chain):
	def __init__(self, hidden_neuron, out_neuron):
		super(Iris_Network, self).__init__()
		with self.init_scope():
			self.L1 = L.Linear(None, hidden_neuron)
			self.L2 = L.Linear(None, hidden_neuron)
			self.L3 = L.Linear(None, out_neuron)
	def __call__(self,x):
		h1 = F.softmax(self.L1(x))
		h2 = F.softmax(self.L2(h1))
		h3 = F.sigmoid(self.L3(h2))
		return h3

def train_model(X_train, X_test, y_train, y_test):
	hidden_neuron = 1000
	out_neuron = 3
	epochs = 10000
	batch_size = 20
	N = len(X_train)
	M = len(X_test)
	#test_N = len(X_test)

	model = Iris_Network(hidden_neuron,out_neuron)	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	for epoch in range(1, epochs+1):
		batch = 0
		epoch_loss = 0
		epoch_accuracy = 0
		test_batch = 0
		test_epoch_loss = 0
		test_epoch_accuracy = 0
		for i in six.moves.range(0, N, batch_size):
			batch+=1
			x=chainer.Variable(np.array(X_train[i:i+batch_size], dtype=np.float32))
			t=chainer.Variable(np.array(y_train[i:i+batch_size], dtype=np.int32))
			x1=model(x)
			#import ipdb; ipdb.set_trace()
			loss=F.softmax_cross_entropy(x1,t)
			epoch_loss = epoch_loss + loss
			accuracy = F.accuracy(x1,t)
			epoch_accuracy = epoch_accuracy + accuracy
			model.cleargrads()
			loss.backward()
			optimizer.update()
		for i in six.moves.range(0, M, batch_size):
			test_batch+=1
			x_t=chainer.Variable(np.array(X_test[i:i+batch_size], dtype=np.float32))
			t_t=chainer.Variable(np.array(y_test[i:i+batch_size], dtype=np.int32))
			x_t1=model(x_t)
			#import ipdb; ipdb.set_trace()
			loss=F.softmax_cross_entropy(x_t1,t_t)
			test_epoch_loss = test_epoch_loss + loss
			test_accuracy = F.accuracy(x_t1,t_t)
			test_epoch_accuracy = test_epoch_accuracy + test_accuracy
		epoch_loss = epoch_loss / (N/batch_size)
		epoch_accuracy = epoch_accuracy / (N/batch_size)
		test_epoch_loss = test_epoch_loss / (M/batch_size)
		test_epoch_accuracy = test_epoch_accuracy / (M/batch_size)
		print "Epoch: ",str(epoch)
		print "Train loss: "+str(float(epoch_loss.data)) + " Train accuracy: "+str(float(epoch_accuracy.data))
		print "Test loss: "+str(float(test_epoch_loss.data)) + " Test accuracy: "+str(float(test_epoch_accuracy.data))

X_train, X_test, y_train, y_test = prepare_data()
train_model(X_train, X_test, y_train, y_test)

