import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train
array([2, 1, 4, ..., 9, 7, 8])
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0
    def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
Iteration:  0
[9 7 9 ... 2 7 9] [2 1 4 ... 9 7 8]
0.10373170731707317
Iteration:  10
[1 0 9 ... 2 7 1] [2 1 4 ... 9 7 8]
0.16109756097560976
Iteration:  20
[3 3 4 ... 4 7 3] [2 1 4 ... 9 7 8]
0.2694146341463415
Iteration:  30
[3 3 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.3693658536585366
Iteration:  40
[3 3 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.44697560975609757
Iteration:  50
[3 3 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.5068048780487805
Iteration:  60
[3 3 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.5510487804878049
Iteration:  70
[3 3 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.5859512195121951
Iteration:  80
[3 1 4 ... 4 9 3] [2 1 4 ... 9 7 8]
0.6153414634146341
Iteration:  90
[3 1 4 ... 9 9 3] [2 1 4 ... 9 7 8]
0.6401219512195122
Iteration:  100
[3 1 4 ... 9 9 3] [2 1 4 ... 9 7 8]
0.6623170731707317
Iteration:  110
[3 1 4 ... 9 9 3] [2 1 4 ... 9 7 8]
0.6808048780487805
Iteration:  120
[3 1 4 ... 9 9 3] [2 1 4 ... 9 7 8]
0.6977317073170731
Iteration:  130
[1 1 4 ... 9 9 3] [2 1 4 ... 9 7 8]
0.7113170731707317
Iteration:  140
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7225609756097561
Iteration:  150
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7330243902439024
Iteration:  160
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7426829268292683
Iteration:  170
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7519756097560976
Iteration:  180
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7601951219512195
Iteration:  190
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.767609756097561
Iteration:  200
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7745609756097561
Iteration:  210
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7805365853658537
Iteration:  220
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7865365853658537
Iteration:  230
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.791609756097561
Iteration:  240
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.7959024390243903
Iteration:  250
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.800439024390244
Iteration:  260
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8046829268292683
Iteration:  270
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8083170731707318
Iteration:  280
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.811609756097561
Iteration:  290
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8155121951219512
Iteration:  300
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.819390243902439
Iteration:  310
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8219512195121951
Iteration:  320
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8242439024390243
Iteration:  330
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8264634146341463
Iteration:  340
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8285609756097561
Iteration:  350
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.831170731707317
Iteration:  360
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.833170731707317
Iteration:  370
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.835170731707317
Iteration:  380
[1 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8368048780487805
Iteration:  390
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8386097560975609
Iteration:  400
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8399756097560975
Iteration:  410
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8415609756097561
Iteration:  420
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8428780487804878
Iteration:  430
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8442439024390244
Iteration:  440
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8455853658536585
Iteration:  450
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8466829268292683
Iteration:  460
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8480975609756097
Iteration:  470
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8491463414634146
Iteration:  480
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8502439024390244
Iteration:  490
[3 1 4 ... 9 9 8] [2 1 4 ... 9 7 8]
0.8517073170731707
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
