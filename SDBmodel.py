import os
import sys
from sklearn import linear_model
import numpy
import collections

class SDBmodel:
	def __init__(self):
		self.clf = linear_model.Lasso(alpha=0.1 , max_iter=1000)
		self.input_data = []
		self.state_data = []

	def add_data(self, input_vector, state_vector):
		self.input_data.append( input_vector )
		self.state_data.append( state_vector )

	def assemble_fit_matrices(self,n,m):
		numrows = len(self.state_data)-1

		for t in range(0,numrows):
			arow = numpy.array(self.state_data[t][0]*numpy.identity(n));
			for i in range(1,n):
				arow = numpy.hstack([arow,self.state_data[t][i]*numpy.identity(n)]);
			for i in range(0,m):
				arow = numpy.hstack([arow,self.input_data[t][i]*numpy.identity(n)]);
			if t == 0:
				Amatrix = arow;
				bmatrix = numpy.vstack([self.state_data[t+1][:]]).transpose();
			else:
				Amatrix = numpy.vstack([Amatrix,arow]);
				bmatrix = numpy.vstack([bmatrix,numpy.vstack([self.state_data[t+1][:]]).transpose()]);

		matrices = collections.namedtuple('matrices',['A','b'])
		Abmatrices = matrices(Amatrix,bmatrix)
		return Abmatrices


	def identify_model(self):
		self.clf.fit(self.Amatrix,self.bmatrix)

	def identify_model_JuMP(self):
		# Tony write this
		print "To be written"

	def predict(self , state_vector, input_vector):
		inputs = state_vector[:]
		inputs.extend(input_vector[:])
		prediction = self.clf.predict( inputs )
		return [prediction]

	def assemble_linmodel_matrices(self,n,m):
		A = [];
		B = [];
		return A,B;

	def get_model(self):
		return self.clf.coef_
