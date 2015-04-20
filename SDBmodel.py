import os
import sys
from sklearn import linear_model
import numpy

class SDBmodel:
	def __init__(self):
		self.clf = linear_model.Lasso(alpha=0.1 , max_iter=1000)
		self.input_data = []
		self.state_data = []
		self.Amatrix = []
		self.bmatrix = []

	def add_data(self, input_vector, state_vector):
		self.input_data.append( input_vector )
		self.state_data.append( state_vector )

	def assemble_matrices(self):
		numrows = len(self.state_data)-1

		for i in range(0,numrows):
			arow = self.state_data[i][:]
			arow.extend(self.input_data[i][:])
			self.Amatrix.append(arow)
			self.bmatrix.extend(self.state_data[i+1][:])

	def identify_model(self):
		self.clf.fit(self.Amatrix,self.bmatrix)

	def predict(self , state_vector, input_vector):
		inputs = state_vector[:]
		inputs.extend(input_vector[:])
		prediction = self.clf.predict( inputs )
		return [prediction]

	def get_model(self):
		return self.clf.coef_
