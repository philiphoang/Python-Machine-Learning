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
from sklearn.model_selection import ParameterGrid, GridSearchCV


# Mainly used to initialize the algorithm to run on one thread 
def neural_network_training(X_train):
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    k.set_session(sess)

    return neural(X_train)

# Create a neural network model
def create_model(input_dim):
    model = Sequential() # initialize
    model.add(Dense(input_dim, input_dim=input_dim, activation='tanh')) 
    model.add(Dense(700, activation='relu')) 
    model.add(Dense(700, activation='relu')) 
    model.add(Dense(150, activation='relu')) 
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    return model 

# Use Keras Classifier from sklearn to be able to run create confusion matrix and classifier report 
def neural(X_train):
    features = X_train.shape[1]
    keras_model = KerasClassifier(build_fn=create_model, input_dim=features, epochs=20, batch_size=1500)
    
    return keras_model