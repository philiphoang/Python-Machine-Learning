# Create Neural network classifier
from keras.models import Sequential

iris = datasets.load_iris()

X_iris = iris["data"]
y_iris = (iris["target"])

X_train, y_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.33, random_state=42)

model = Sequential() # initialize
model.add(Dense(12, input_dim=8, activation='relu')) # 2 layers, input 8 and 12 hidden layer
model.add(Dense(8, activation='relu')) # input 8, activation relu
model.add(Dense(1, activation='sigmoid')) # input 1, activation sigmoid
#Then you need to compile model (for speed):
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model (X is features, Y is output, this is for training data)
model.fit(X_train, Y_train, epochs=150, batch_size=10)
# epochs are the number of iterations wanted.
# batch size the number of iterations before update
# (150 epoches, 10 batch size = 15 #updates).
# score that should be at least 95%
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
