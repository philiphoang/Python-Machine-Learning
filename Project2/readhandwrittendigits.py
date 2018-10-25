import csv 
import numpy as np 
import os.path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def readhandwrittendigits():
    image_path = os.path.join(os.path.dirname(__file__), "data/handwritten_digits_images.csv")
    labels_path = os.path.join(os.path.dirname(__file__) ,"data/handwritten_digits_labels.csv")

    with open(image_path) as f:
        reader = csv.reader(f)
        x_data = np.array(list(reader))
    
    # Reading the label file 
    with open(labels_path) as f:
        reader = csv.reader(f)
        y = np.array(list(reader))

    X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.4, random_state = 42)

    return X_train, X_test, y_train, y_test
