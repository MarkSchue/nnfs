import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        # output ist prediction des Modells, incl. Softmax
        sample_losses = self.forward(output, y)
        # Hier wird einfach der Durchschnitt der Verluste berechnet
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples im Output
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            #range(samples) gibt ein Array mit den Zahlen von 0 bis samples - 1
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

class Accuracy:
    def calculate(self, predictions, y):
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)

activationReLu = Activation_ReLU()
activationSoftMax = Activation_Softmax()
dense1.forward(X)

activationReLu.forward(dense1.output)
dense2.forward(activationReLu.output)
activationSoftMax.forward(dense2.output)

lossFunction = Loss_CategoricalCrossentropy()
loss = lossFunction.calculate(activationSoftMax.output, y)

accuracy = Accuracy()
acc = accuracy.calculate(activationSoftMax.output, y)

print(activationSoftMax.output[:10])
print("Loss: ", loss)
print("Accuracy: ", acc)
