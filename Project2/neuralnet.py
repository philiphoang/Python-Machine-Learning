from keras.models import Sequential
from keras.layers import Activation, Dense, Lambda, Dropout
import time
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as k
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

def neural_network_training(X_train, X_test, y_train, y_test):
    tf.set_random_seed(19)
    rn.seed(19)
    np.random.seed(19)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    k.set_session(sess)

    neural(X_train, X_test, y_train, y_test)

def create_model(input_dim):
    model = Sequential() # initialize
    model.add(Dense(input_dim, input_dim=input_dim, activation='tanh')) 
    model.add(Dense(512, activation='relu')) 
    model.add(Dense(500, activation='relu')) 
    model.add(Dense(250, activation='relu')) 
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    return model 


def neural(X_train, X_test, y_train, y_test):
    features = X_train.shape[1]
    keras_model = KerasClassifier(build_fn=create_model, input_dim=features, epochs=15, batch_size=1000)
    
    start_time = time.time()
    keras_model.fit(X_train, y_train)
    print("Time Neural Network: %0.10f seconds" % (time.time() - start_time))
    
    y_pred = keras_model.predict(X_test)
    print("Accuracy: %0.4f " % accuracy_score(y_test, y_pred))

    conf_matrix("Neural Network", y_test, y_pred)

def conf_matrix(name, y_test, y_pred):
    target_names = [str(x) for x in range(0, 10)]
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels= target_names,
            yticklabels= target_names)

    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.title(name)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()   