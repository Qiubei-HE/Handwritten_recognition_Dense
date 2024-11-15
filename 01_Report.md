#### Parameter

epochs = 10
batch_size = 32
learning_rate = 0.001

#### Datensatz

load_digits()

70% Trainingssatz, 15% Evaluationssatz, 15% Testsatz

#### One-hot-Vektor

Im Code habe ich zur einfacheren Verarbeitung durch die Maschine eine One-hot-Kodierung auf die Labels der handschriftlichen Ziffern angewendet, wobei jedes einzelne Ziffernlabel in ein One-hot-Vektor-Format umgewandelt wurde.

Das Label vor der One-hot-Kodierung

[2 3 3 ... 4 9 5]

Das Label nach der One-hot-Kodierung

[[0. 0. 1. ... 0. 0. 0.]

[0. 0. 0. ... 0. 0. 0.]

[0. 0. 0. ... 0. 0. 0.]

...

[0. 0. 0. ... 0. 0. 0.]

[0. 0. 0. ... 0. 0. 1.]

[0. 0. 0. ... 0. 0. 0.]]

#### Modell

Die Eingabeschicht flacht die Eingabedaten in ein 1D-Array der Größe 64 ab, da der Datensatz aus 8x8-Bildern mit 64 Pixeln Information besteht. Als Nächstes folgt eine vollständig verbundene Schicht mit 128 Neuronen, gefolgt von einer weiteren vollständig verbundenen Schicht mit 64 Neuronen. Beide Zwischenschichten (128, 64) verwenden die ReLU-Aktivierungsfunktion. Schließlich gibt es eine Ausgabeschicht mit 10 Neuronen, da wir insgesamt 10 Klassen (0-9) haben, mit der Softmax-Aktivierungsfunktion.

#### Verlustfunktion

Da es sich um ein Mehrklassenklassifizierungsproblem handelt, verwenden wir 'categorical_crossentropy' als Verlustfunktion.

#### Ergebnisgrafik

![training_evaluation_metrics](https://github.com/user-attachments/assets/fb2bea6e-4487-4284-8cdd-71fc868177ce)

#### Bewertungsergebnisse basierend auf einem unabhängigen Testsatz

Testverlust: 0.1007

Testgenauigkeit: 0.9593

Testpräzision: 0.9699

Test-Recall: 0.9556

Test-F1-Wert: 0.9627




---------**ENGLISH BELOW**------------




#### Parameters
epochs = 10
batch_size = 32
learning_rate = 0.001

#### Dataset
load_digits()

70% Train set, 15% Eval set, 15% test set

#### **One-hot-vector**

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
