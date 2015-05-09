import urlparse
from smap.archiver.client import SmapClient
#from mpc import *
from ConfigParser import ConfigParser
from smap.contrib import dtutil
import smap
import matplotlib.pyplot as plt
import numpy
from SDBmodel import *

class SimpleOfflineMPC:

	def __init__(self, config_filename):
		Config = ConfigParser()
		Config.read(config_filename)
		self.setup(Config)

	def setup(self, opts):
		self.rate = float( opts.get('mpc', 'rate'   ).strip() )
		self.input_variables = opts.get('mpc', 'input_variables').split(',')
		self.state_variables = opts.get('mpc', 'state_variables').split(',')
		self.read_stream_data( int(opts.get('mpc', 'time_period')) )
		self.setup_model()

	'''
	Create MPC kernel
	'''
	def setup_model(self):
		self.model = SDBmodel()


	'''
	Function that runs periodically to update the model
	'''
	def run(self):
		# regression steps; use two autoregressive steps
		regsteps = 2;

		# Collect data
		for step in range(2, 1000):
			# Input vector at step t-1
			input_vector_t_1 = self.construct_input(step-1)
			# State vector at step t-1
			state_vector_t_1 = self.construct_state(step-1,regsteps)
			# Call mpc kernel to add data
			self.model.add_data( input_vector_t_1 , state_vector_t_1)

			# # Print data for DEBUG
			# if step == 3:
			# 	print state_vector_t_1
			# 	print input_vector_t_1


		# Make model matrices
		n = regsteps;
		m = len(self.input_variables);
		fitmatrices = self.model.assemble_ARMAfit_matrices(n,m)

		# # print fitmatrices.A
		# # print fitmatrices.b
		# print numpy.shape(fitmatrices.A)
		# print numpy.shape(fitmatrices.b)

		# Identify model
		self.model.identify_model(fitmatrices.A,fitmatrices.b)

		# Get model parameters
		params = self.model.get_model()

		print params[:]

		# Put together lin model matrices
		matrices = 	self.model.assemble_linmodel_matrices(params,n,m);

		print "A matrix of ARMA"
		print matrices.A
		print "B matrix of ARMA"
		print matrices.B

		# Do One-Step Prediction
		prediction = [];
		actual  = [];
		for step in range(2, 1000):
			newpredict = 0;
			newinput = self.construct_state(step,regsteps);
			newinput.extend(self.construct_input(step));
			# print newinput
			for i in range(0,n+m):
				newpredict = newpredict + params[i]*newinput[i];
			prediction.append(newpredict)
			# prediction.append(self.model.predict(self.construct_state(step,regsteps),self.construct_input(step),n,m))
			state_t = self.construct_state(step,regsteps)
			actual.append(numpy.array([state_t[0]]))

		# print prediction[1:5]
		# print actual[1:5]
		# print numpy.shape(prediction)
		# print numpy.shape(actual)
		plt.subplot(2,1,1)
		plt.plot( numpy.arange( len(prediction) ), prediction, label="predicted trajectory")
		plt.plot( numpy.arange( len(actual) ), actual, label="actual-output")
		plt.xlabel("Step")
		plt.legend()
		plt.title("One step prediction")

		# Do full day prediction
		prediction = [];
		actual  = [];
		# print fitmatrices.A[0:5]
		state = self.construct_state(2,regsteps);
		for step in range(2, 1000):
			newpredict = 0;
			newinput = state[:];
			newinput.extend(self.construct_input(step));
			for i in range(0,n+m):
				newpredict = newpredict + params[i]*newinput[i];
			prediction.append(newpredict)
			state = self.propagate_state(state,newinput[n:],matrices.A,matrices.B)
			state_t = self.construct_state(step,regsteps)
			actual.append(numpy.array([state_t[0]]))

		plt.subplot(2,1,2)
		plt.plot( numpy.arange( len(prediction) ), prediction, label="predicted trajectory")
		plt.plot( numpy.arange( len(actual) ), actual, label="actual-output")
		plt.xlabel("Step")
		plt.legend()
		plt.title("Full day prediction")
		plt.show()

	'''
	Reads data to be supplied to build the model
	'''
	def read_stream_data(self, num_days=1):
		self.points = {}
		c = SmapClient("http://new.openbms.org/backend")
		for point in self.input_variables:
			q = "apply window(mean, field='second',width='%d') to data in (\"03/01/2015\" -%ddays, \"03/07/2015\") where Metadata/Name='%s'" % \
						( self.rate,  num_days,  point )

			print q
			result = c.query(q)
			readings = result[0]["Readings"]
			self.points[point] = [ r[1] for r in result[0]["Readings"] ]

		for point in self.state_variables:
			query = "apply window(mean, field='second',width='%d') to data in (\"03/01/2015\" -%ddays, \"03/07/2015\") where Metadata/Name='%s'" % \
						( self.rate,  num_days,  point )

			result = c.query(query)
			readings = result[0]["Readings"]
			self.points[point] = [ r[1] for r in result[0]["Readings"] ]

		self.predictions = []
		self.model_params = []
		self.actual_outputs = []



	'''
	Constructs an input vector at a particular timestep
	'''
	def construct_input(self, step):
		input_vector = []
		for point in self.input_variables:
			input_vector.append( self.points[point][ step - 1] )
		# for point in self.state_variables:
		# 	input_vector.append( self.points[point][ step - 1] )
		return input_vector

	'''
	Constructs the state vector at a particular timestep using autoregression
	'''
	def construct_state(self, step, regsteps):
		state_vector = []
		for point in self.state_variables:
			for i in range(0,regsteps):
				state_vector.append(self.points[point][ step - 1 + i])
		return state_vector[:]

	def save_predictions(self, val):
		self.predictions.append( val )

	def save_model_params(self, val):
		self.model_params.append( val )

	def save_actual_output(self, val):
		self.actual_outputs.append( val )

	def propagate_state(self, state, newinput, A, B):
		state = numpy.dot(A,state) + numpy.dot(B,newinput);
		return state[:].tolist()



if __name__ == "__main__":
	som = SimpleOfflineMPC(sys.argv[1])
	som.run()
