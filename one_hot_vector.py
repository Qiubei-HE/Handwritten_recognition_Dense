############### Note 📒 ###############
## 08/11/2024

# 1. one_hot_vector(): Input a number and return the corresponding one-hot vector.
# 2. mean_loss(): Calculating MSE, used as a loss function.

# The professor explained the logic of model training: 
# the equation for calculating error (MSE), and the process of minimizing this error by adjusting the model parameters ('min L(x)')

# He also introduced the one-hot vector, 
# which can convert categorical variables into a form that algorithms can easily utilize (binary vectors).
#######################################

import numpy as np

def one_hot_vector(y, n = 10):
# In here, I make the n defult = 10, because the number only have 0-9, 10 numbers.

    if isinstance(y, int):
        # If input is a single number: return a one-hot vector
        vector = [0] * n
        if 0 <= y < n:
            vector[y] = 1
        return np.array(vector)
    
    elif isinstance(y, (list, tuple, set)):
        # If input is a Sequence: return a n(10) x m matrix where m is the length of y
        matrix = []
        for index in y:
            row = [0] * n
            if 0 <= index < n:
                row[index] = 1
            matrix.append(row)
        return np.array(matrix)

def mean_loss(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Calculate individual losses (errors)
    errors = (y_true - y_pred) ** 2
    # Calculate the mean of all errors
    return np.mean(errors)



## TEST EXAMPLE

# Test one_hot_vector
print("Single number test:", "\n", one_hot_vector(7))  # Should be print 1 at the 7th position in a single vector.
print("Multiple numbers test:", "\n", one_hot_vector([1, 3, 4]))  # Should be return a matrix with n=10 columns and m=3 rows.

# Test Loss function
y_true = [3, 1, 2, 7] # assume it is real data
y_pred = [3, 1, 2, 8] # assume it is prediction
print("Mean Loss (MSE):", mean_loss(y_true, y_pred))
