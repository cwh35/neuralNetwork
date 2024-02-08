# A neural network trained w/ backpropagation
# Input predicts output

import numpy as np

# 2 Layer Neural Network


# Sigmoid function
# nonlinearity
# maps any value to a value between 0 and 1
def nonlin(x, derivative = False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Input dataset
X = np.array([ [0, 0, 1],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 1]
            ])

# Output dataset
# Transpose to line up dimensions with input dataset
y = np.array([[0, 0, 1, 1]]).T

# Seed random numbers --> make calculation deterministic
np.random.seed(1)

# Initialize weights randomly with mean 0 
# syn0 = First layer of weights (Synapse0) --> this connects l0 to l1
syn0 = 2 * np.random.random((3,1)) - 1

for iter in range(10000):

    # Forward Propagation
    # l0 = First Layer of Network
    # l1 = Second Layer of Network (Hidden Layer)
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # Error = what we missed
    l1_error = y - l1

    # Multiply error by slope of sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # Update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)