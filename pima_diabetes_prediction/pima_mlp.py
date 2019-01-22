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
        return h3

