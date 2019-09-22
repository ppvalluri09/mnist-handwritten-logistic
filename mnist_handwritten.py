import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def prepareData(X, type):
	if type == 'f' or type == 'features':
	    padding = np.ones((len(X), 1)).T[0]
	    X_ = np.array(X)
	    X_train = np.array([padding, X_])
	    return X_train.T
	elif type == 'l' or type == 'labels':
		y_ = np.array([X]).T
		return y_

def categorize(y):
    y_train = np.zeros((len(y), 10))

    for i in range(len(y)):
        y_train[i][y[i]] = 1

    return y_train

def fit_to_categories(y_pred):
    result = []
    for i in range(len(y_pred)):
        row = list(y_pred[i])
        ans = row.index(max(y_pred[i]))
        result.append(ans)

    return result

def calculate_error(y, y_pred):
    train_set = list(y)
    error_count = 0
    for i in range(len(y)):
        if train_set[i] != y_pred[i]:
            error_count = error_count + 1

    return error_count / len(y) * 100.0
        


def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def sort(a, b):
	for i in range(len(a)):
		for j in range(len(b)):
			if i != j and a[i] < a[j]:
				a[i], a[j] = a[j], a[i]
				b[i], b[j] = b[j], b[i]

	return a, b

def best_fit(X, y, y_pred):
	plt.scatter(X, y, s=10, c='b')
	plt.xlabel('Features')
	plt.ylabel('Labels')
	plt.title('Logit')
	X, y_pred = sort(X, y_pred)
	plt.plot(X, y_pred, c='r')
	# plt.show()

class Model:

	def __init__(self, dim=1):
		self.R2 = 0
		self.w = np.random.randn(dim, 1)
		self.prediction = 0
		self.X = 0
		self.y = 0
		self.mean = []
		self.var = []
		self.cost = []
		self.cycles = 1
		self.y_pred = 0

	def scale(self, data):
		return data / 255.0

	def fit(self, X):
            self.y_pred = sigmoid(np.dot(X, self.w))
            return self.y_pred

	def transform(self, X, y, cycles=5000):

		self.X = X
		self.y = y
		self.cost = []
		self.cycles = cycles
		self.w = np.random.randn(len(X[0]), len(y[0]))	
		learning_rate = 0.07
		m = len(X)

		for iterations in range(cycles):
			y_pred = self.fit(X)
			cost = self.calculate_cost(y_pred)
			self.cost.append(cost)
			error = (1/m) * np.dot(X.T, (y_pred - y))
			self.w = self.w - learning_rate * error

	def calculate_cost(self, y_pred):
		m = len(y_pred)
		return -1.0 * (1 / m) * np.sum((self.y * (np.log(y_pred))) + ((1 - self.y) * (np.log(1 - y_pred))))

	def plot_cost(self):
                x = [i for i in range(self.cycles)]
                plt.xlabel('Iterations')
                plt.ylabel('Cost')
                plt.title('Cost vs Iterations')
                plt.plot(x, self.cost, c='g')
                plt.show()

df = pd.read_csv(r'datasets/mnist_train.csv')
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
y = y[:1000]
X = X[:1000]
print('Data Loaded...')

X_train = np.zeros((len(X), len(X[0]) + 1))
X_train[:, 1:] = X
X_train[:, 0] = np.ones((len(X), 1)).T[0]
y_train = categorize(y)

logit = Model()
X_train = logit.scale(X_train)
print('Training...')
logit.transform(X_train, y_train)
print('Training Done...')
y_pred = logit.fit(X_train)
print('Predicted Output:-')

result = fit_to_categories(y_pred)
print(result)

print('Error:- {}'.format(calculate_error(y, result)))

logit.plot_cost()


