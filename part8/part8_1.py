import os
os.system("cls")

import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

softmax_outputs_clipped = np.clip(softmax_outputs, 1e-7, 1-1e-7)

class_targets = [0, 1, 1]

neg_log = -np.log(softmax_outputs_clipped[range(len(softmax_outputs_clipped)), class_targets])

average_loss = np.mean(neg_log)
print(average_loss)