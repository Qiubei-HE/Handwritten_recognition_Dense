#### Parameters
epochs = 10
batch_size = 32
learning_rate = 0.001

#### Dataset
load_digits()

70% Train set, 15% Eval set, 15% test set

#### One-hot-vector

In the code, to make it easier for the machine to process. I applied one-hot encoding to the labels of handwritten digits, converting each single-digit label into a one-hot vector format.

- The label befor One-hot encode
  
 [2 3 3 ... 4 9 5]
- The label after One-hot encode
  
 [[0. 0. 1. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 ...
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 1.]
 
 [0. 0. 0. ... 0. 0. 0.]]

#### Model

The input layer flattens the input data into a 1D array of 64, as the dataset consists of 8x8 images with 64 pixels of information. Next is a fully connected layer with 128 neurons, followed by another fully connected layer with 64 neurons. Both of these intermediate layers (128, 64) use the ReLU activation function. Finally, there is an output layer with 10 neurons, as we have a total of 10 classes (0-9), with the softmax activation function.

#### Loss

Since this is a multi-class classification problem, we used 'categorical_crossentropy' as our loss function.

#### Result plot
![training_evaluation_metrics](https://github.com/user-attachments/assets/fb2bea6e-4487-4284-8cdd-71fc868177ce)


#### Evaluation Results Based on Independent Test Set
- Test Loss: 0.1007
- Test Accuracy: 0.9593
- Test Precision: 0.9699
- Test Recall: 0.9556
- Test F1 Score: 0.9627
