import os
import sys
from sklearn import linear_model
import numpy
import collections

class SDBmodel:
	def __init__(self):
		self.clf = linear_model.Lasso(alpha=0.001 , max_iter=10000)
		self.input_data = []
		self.state_data = []

	def add_data(self, input_vector, state_vector):
		self.input_data.append( input_vector )
		self.state_data.append( state_vector )

	def assemble_ARMAfit_matrices(self,n,m):
		numrows = len(self.state_data)-1

		for t in range(0,numrows):
			arow = numpy.array(self.state_data[t][0]);
			for i in range(1,n):
				arow = numpy.hstack([arow,self.state_data[t][i]]);
			for i in range(0,m):
				arow = numpy.hstack([arow,self.input_data[t][i]]);
			if t == 0:
				Amatrix = arow;
				bmatrix = numpy.vstack([self.state_data[t+1][0]]).transpose();
			else:
				Amatrix = numpy.vstack([Amatrix,arow]);
				bmatrix = numpy.vstack([bmatrix,numpy.vstack([self.state_data[t+1][0]]).transpose()]);

		matrices = collections.namedtuple('matrices',['A','b'])
		Abmatrices = matrices(Amatrix,bmatrix)
		return Abmatrices


	def identify_model(self,A,b):
		self.clf.fit(A,b)

	def identify_model_JuMP(self):
		# Tony write this
		print "To be written"

	def predict(self , state_vector, input_vector,n,m):
		inputs = []
		for i in range(0,n):
			for j in range(0,n):
				inputs.append(state_vector[i])
		for i in range(0,m):
			for j in range(0,n):
				inputs.append(input_vector[i])
		print inputs
		prediction = self.clf.predict( inputs )
		return prediction

	def assemble_linmodel_matrices(self,params,n,m):
		A = numpy.zeros((n,n));
		B = numpy.zeros((n,m));
		A[0][:] = params[0:n];
		B[0][:] = params[n:];
		for i in range(1,n):
			for j in range(0,n):
				if i-j ==1:
					A[i][j] = 1
				else:
					A[i][j] = 0
		linmatrices = collections.namedtuple('matrices',['A','B'])
		ABmatrices = linmatrices(A,B)
		return ABmatrices;

	def get_model(self):
		return self.clf.coef_
