from anubia import *

X = np.random.rand(100, 5)
Y = np.random.rand(100, 2)

model = DeepLearning(X, Y, learning_rate=0.01, hidden_layers=[5, 5], activation='sigmoid')
model.train(epochs=1000000, verbose=False)

predictions = model.predict(X.tolist())

print(mean_squared_error(Y, predictions))