import os
import sys
from sklearn import linear_model
import numpy

class MPC:
	def __init__(self):
		self.clf = linear_model.Lasso(alpha=0.1 , max_iter=1000)
		self.input_data = []
		self.state_data = []

	def add_data(self, input_vector, state_vector):
		self.input_data.append( input_vector )
		self.state_data.append( state_vector )

	def predict(self , input_vector):
		self.clf.fit( self.input_data , self.state_data )
		prediction = self.clf.predict( input_vector )
		return prediction

	def get_model(self):
		return self.clf.coef_
