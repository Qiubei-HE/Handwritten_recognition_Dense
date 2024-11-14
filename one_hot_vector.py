############### Note 📒 ###############
## 08/11/2024

# 1. one_hot_vector(): Input a number and return the corresponding one-hot vector.
# 2. mean_loss(): Calculating MSE, used as a loss function.

# The professor explained the logic of model training: 
# the equation for calculating error (MSE), and the process of minimizing this error by adjusting the model parameters ('min L(x)')

# He also introduced the one-hot vector, 
# which can convert categorical variables into a form that algorithms can easily utilize (binary vectors).
#######################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

def one_hot_vector(y, n=10):
    encoder = OneHotEncoder(categories=[range(n)], sparse_output=False)
    y = np.array(y).reshape(-1, 1)  # Convert the input to a column
    return encoder.fit_transform(y)

def mean_loss(y_true, y_pred):
    # Use mean_squared_error to get the MSE
    return mean_squared_error(y_true, y_pred)


## TEST EXAMPLE

# Test one_hot_vector
print("Single number test:", "\n", one_hot_vector(7))  # Should be print 1 at the 7th position in a single vector.
print("Multiple numbers test:", "\n", one_hot_vector([1, 3, 4]))  # Should be return a matrix with n=10 columns and m=3 rows.

# Test Loss function
y_true = [3, 1, 2, 7] # assume it is real data
y_pred = [3, 1, 2, 8] # assume it is prediction
print("Mean Loss (MSE):", mean_loss(y_true, y_pred))

