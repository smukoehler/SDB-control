from SDBmodel import *
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy

class SineWaveMPC():
	def __init__(self):
		self.rate = 1
		self.input_variables = [ "input_sine " ]
		self.state_variables = [ "output_sine" ]
		self.read_stream_data()
		self.setup_model()

	'''
	Create MPC kernel
	'''
	def setup_model(self):
		self.model = SDBmodel()


	'''
	Collect all the data, make a model
	'''
	def run(self):
		# Collect data
		for i in range(2, 1000):
			# Input vector at step t-1
			input_vector_t_1 = self.construct_input(i-1)
			# State vector at step t-1
			state_vector_t_1 = self.construct_state(i-1,1)
			# Call mpc kernel to add data
			self.model.add_data( input_vector_t_1 , state_vector_t_1)

			# Input vector at step t
			input_vector_t = self.construct_input(i)

		# Make model matrices
		self.model.assemble_matrices()

		# Identify model
		self.model.identify_model()

		# Get model parameters
		params = self.model.get_model()

		print params[:]

		# Validate model
		self.validate_model()


	'''
	Reads data to be supplied to build the model
	'''
	def read_stream_data(self):
		self.points = {}

		# Generate offline input data
		for point in self.input_variables:
			self.points[point] = [ math.sin(math.radians(i)) for i in range(1000) ]

		# Generate offline state data
		for point in self.state_variables:
			self.points[point] = [  math.sin(math.radians(i)) * 5 for i in range(1000) ]

		self.predictions = []
		self.model_params = []
		self.actual_outputs = []

	'''
	Constructs an input vector at a particular timestep
	'''
	def construct_input(self, step):
		input_vector = []
		try:
			for point in self.input_variables:
				input_vector.append( self.points[point][ step - 1] )
		except:
			return None
		return input_vector


	'''
	Constructs the state vector at a particular timestep
	'''
	def construct_state(self, step):
		state_vector = []
		try:
			for point in self.state_variables:
				state_vector.append( self.points[point][ step - 1 ])
		except:
			return None
		return state_vector


	def validate_model(self):
		# initial state
		initial_state = []
		for point in self.state_variables:
			initial_state.append( self.points[point][0])

		trajectory = [initial_state[:]]
		# propogate state with inputs
		for i in range(1, 1000):
			next_state = []
			current_input = []
			for point in self.input_variables:
				current_input.append( self.points[point][i] )
			next_state = self.model.predict(trajectory[-1],current_input)
			trajectory.append(next_state)

		plt.subplot(2,1,1)
		plt.plot( numpy.arange( len(trajectory) ), trajectory, label="predicted trajectory")
		plt.plot( numpy.arange( len(self.actual_outputs) ), self.actual_outputs, label="actual-output")
		plt.xlabel("Step")
		plt.legend()
		plt.show()

		return trajectory

if __name__ == "__main__":
	swm = SineWaveMPC()
	swm.run()
