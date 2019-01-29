import numpy as np
import chainer
import six
from pima_mlp import Build_Network
from sklearn.model_selection import train_test_split
from chainer import serializers
import chainer.functions as F

def prepare_data():
        """prepare_data function loads data from csv to numpy
        and it splits data in training set and test set."""
	dataset = np.loadtxt("pima-diabetes.csv", delimiter=",")
	input_data_x = dataset[:,0:8]
	input_label_y = dataset[:,8]
	X_train, X_test, y_train, y_test = train_test_split(input_data_x, \
	input_label_y, test_size=0.33, random_state=42)
	return X_train, X_test, y_train, y_test


def train_network(X_train, X_test, y_train, y_test):
	"""train_network function initializes the model and trains the model"""
	"""Initializing variables """
	neuron_units = 100
	neuron_units_out = 2
	n_epoch = 1000
	batch_size = 50
	N = len(X_train)
	test_N = len(X_test)
	train_batch_count = N/batch_size
	test_batch_count = test_N/batch_size
	
	"""Instanciating the Build_network class and setting up the optimizer"""
	model = Build_Network(neuron_units, neuron_units_out)
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	"""Epoch loop"""
	for epoch in range(1, n_epoch + 1):
		batch_count = 0
		epoch_loss = 0
		epoch_accuracy = 0
		test_batch_count = 0
                test_epoch_loss = 0
                test_epoch_accuracy = 0
		perm = np.random.permutation(N)
		test_perm = np.random.permutation(test_N)
		"""Batch loop"""
		for i in six.moves.range(0, N, batch_size):
			batch_count += 1
			"""converting training data to chainer variables"""
			x = chainer.Variable(np.array(X_train[perm[i:i + batch_size]], dtype=np.float32))
			t = chainer.Variable(np.array(y_train[perm[i:i + batch_size]], dtype=np.int32))
			"""Passing the training data x to model object"""
			x1 = model(x)
			"""Loss computation"""
			loss = F.softmax_cross_entropy(x1,t)
			epoch_loss = epoch_loss + loss
			accuracy = F.accuracy(x1,t)
			epoch_accuracy = epoch_accuracy + accuracy
			"""Backpropogation and optimizing"""
			model.cleargrads()
			loss.backward()
			optimizer.update()
		epoch_loss = epoch_loss / train_batch_count
		epoch_accuracy =epoch_accuracy / train_batch_count
		print "Epoch: "+str(epoch)
		print "Train Loss: "+ str(float(epoch_loss.data)) + " Training Accuracy: "+ str(float(epoch_accuracy.data))
		
		"""Test batch loop"""
                for i in six.moves.range(0, test_N, batch_size):
                        test_batch_count += 1
			"""preparing and passing Test Data through network"""
                        x_test = chainer.Variable(np.array(X_test[i:i + batch_size], dtype=np.float32))
                        t_test = chainer.Variable(np.array(y_test[i:i + batch_size], dtype=np.int32))
                        x1_test = model(x_test)
			"""Computing Test Loss"""
                        loss = F.softmax_cross_entropy(x1_test, t_test)
                        test_epoch_loss = test_epoch_loss + loss
                        accuracy = F.accuracy(x1_test, t_test)
                        test_epoch_accuracy = test_epoch_accuracy + accuracy
                test_epoch_loss = test_epoch_loss / test_batch_count
                test_epoch_accuracy = test_epoch_accuracy / test_batch_count
                print "Test Loss: "+ str(float(test_epoch_loss.data)) + " Test Accuracy: "+ str(float(test_epoch_accuracy.data))+ '\n'

	serializers.save_npz('pima_mlp.model', model)
	serializers.save_npz('pima_mlp.state', optimizer)
	print ("Completed Training !!")

X_train, X_test, y_train, y_test = prepare_data()
train_network(X_train, X_test, y_train, y_test)

