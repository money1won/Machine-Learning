# DON'T TOUCH THIS! Think of it as a template
# This is used to show a basic example of how a program computes neural netowrk processes


# ----------------------------------------------------------------------------------------

import numpy as np  # numpy is not local and must be installed

def sigmoid(x):  # Used to normalize the prediction curve
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)  # derivative being used. Not sure why


total_iterations = 200000  # As you increase this number, the results will be better

                        #    x1x2x3
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# Think of the inputs (columns) as 3 different (t/f) options each example (or row) can choose from


                            # y1y2y3y4
training_outputs = np.array([[0,1,1,0]]).T
# Transposes the array to make it [0,  y1
                                #  1,  y2
                                #  1,  y3
                                #  0]  y4


np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) -1
# Creates the initial w1w2w3 (the weights)
# These will change during the training process in the for loop

print("Random starting synaptic weights: ")
print(synaptic_weights)

# ex1: x1w1 + x2w2 + x3w3 = y1
# ex2: x1w1 + x2w2 + x3w3 = y2
# ex3: x1w1 + x2w2 + x3w3 = y3
# ex4: x1w1 + x2w2 + x3w3 = y4

for iteration in range(total_iterations):

    input_layer = training_inputs
    # copies the inputs to a new var for mutation

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # calculates what the program THINKS the output should be based on current weights

    error = training_outputs - outputs
    # notices the error between what was calculated and what is desired

    adjustments = error * sigmoid_derivative(outputs)
    # Takes the error found and weights it by the sigmoid derivative with the desired outputs
    # to give a more accurate new synaptic weight

    synaptic_weights += np.dot(input_layer.T, adjustments)
    # multiplies the current inputs bu adjustments and adds it to be the new weight

print("Synaptic Weights after training")
print(synaptic_weights)  # These are what the weights will be to most closely follow the
# Outputs listed on line 17

print("Outputs after training: ")
print(outputs)  # These outputs are attempting to be the same as on line 17
# The accuracy will increase the more total_iterations you have