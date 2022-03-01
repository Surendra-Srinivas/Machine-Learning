from audioop import bias
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self,inputs):
        sum = np.dot(self.weights,inputs) + self.bias
        return sigmoid(sum)

weights = np.array([0,1])
b = 4
n = Neuron(weights,b)

x = np.array([2,3])
print(n.feedforward(x))
