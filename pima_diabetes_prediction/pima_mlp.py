import chainer
import chainer.links as L
import chainer.functions as F


class Build_Network(chainer.Chain):
    """Neural Network definition, Multi Layer Perceptron"""
    def __init__(self, neuron_units, neuron_units_out):
        super(Build_Network, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred when `None`
            self.l1 = L.Linear(None, neuron_units)      # n_in -> n_units
            self.l2 = L.Linear(None, neuron_units)      # n_units -> n_units
            self.l3 = L.Linear(None, neuron_units_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x) )  #, ratio=0.4
        h2 = F.relu(self.l2(h1))
	h3 = self.l3(h2)
        #h3 = F.sigmoid(self.l3(h2))
        #y = self.l3(h3)
        return h3

class SoftmaxClassifier_Loss(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SoftmaxClassifier_Loss, self).__init__()
        with self.init_scope():
            self.predictor = predictor


    def __call__(self, x, t):

        y = self.predictor(x)
	#import ipdb; ipdb.set_trace()
        self.loss = F.softmax_cross_entropy(y, t)   # this line gives error
        self.accuracy = F.accuracy(y, t)
        #print "Loss : ", self.loss
        #print "Accuracy : ", self.accuracy
        return self.loss

"""
def loss_function(x, t):
	return F.softmax_cross_entropy(x, t, enable_double_backprop=True)
"""
