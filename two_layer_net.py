import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################
        self.elems ={}
        self.elems['weight1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.elems['bias1'] = np.zeros(hidden_dim)
        self.elems['weight2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.elems['bias2'] = np.zeros(num_classes)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        params = None
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return self.elems

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################
        l1, l1_cache = fc_forward(X, self.elems['weight1'], self.elems['bias1'])
        relu1, relu1_cache =  relu_forward(l1)    
        # Compute the second layer scores
        l2, l2_cache = fc_forward(relu1, self.elems['weight2'], self.elems['bias2'])
    
        # Store the cache for the backward pass
        cache = (l1_cache, relu1_cache ,l2_cache)
        scores = l2
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################
        grads = {}
        l1_cache, relu1_cache ,l2_cache = cache 
        # Backpropagate through the second layer
        d_l2, grads['weight2'], grads['bias2'] = fc_backward(grad_scores, l2_cache)
        # Backpropagate through the ReLU layer
        d_relu1 = relu_backward(d_l2, relu1_cache)
        # Backpropagate through the first layer
        _, grads['weight1'], grads['bias1'] = fc_backward(d_relu1, l1_cache)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads
