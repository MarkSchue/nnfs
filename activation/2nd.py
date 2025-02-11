import numpy as np

softmax_outputs = np.array([[0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.02, 0.9, 0.08]])

class_target = [2, 1, 1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_target])

average_loss = np.mean(neg_log)
print(average_loss)
print(neg_log)

