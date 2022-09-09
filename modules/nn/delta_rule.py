# Source code: PyImageSearch.com and some modifications made by myself
# import the necessary packages
import numpy as np

class Perceptron1:
	def __init__(self, N, alpha=0.1):
		# initialize the weight matrix and store the learning rate
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		# apply the step function
		return 1 if x > 0 else 0

	def sumaColumnas(self, m):
		#sum  matrix columns 
		result = []
		for row in range(len(m[0])):
			t = 0
			for col in range(len(m)):
				t += m[col][row]
			float(t)			
			result.append(t)		
		result = result[::-1]
		np.array(result)
		return(result)	 

	def fit(self, X, y, epochs=10):
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
		X = np.c_[X, np.ones((X.shape[0]))]

		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point
			EX = []
			for (x, target) in zip(X, y):
				# take the dot product between the input features
				# and the weight matrix, then pass this value
				# through the step function to obtain the prediction
				p = np.dot(x, self.W)

				# only perform a weight update if our prediction
				# does not match the target
				#if self.step(p) != target:
				# determine the error
				error = target - p
				errorx = error * x
				EX.append(errorx)			
			
			ErrorX = self.sumaColumnas(EX)
			# update the weight matrix
			self.W += np.multiply(ErrorX, self.alpha * (1/X.shape[0]))

	def predict(self, X, addBias=True):
		# ensure our input is a matrix
		X = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			X = np.c_[X, np.ones((X.shape[0]))]

		# take the dot product between the input features and the
		# weight matrix, then pass the value through the step
		# function
		return self.step(np.dot(X, self.W))

	def weights(self):
		#return weights vector
		return (self.W)
