import urlparse
from smap.archiver.client import SmapClient
from mpc import *
from ConfigParser import ConfigParser
from smap.contrib import dtutil
import smap
import matplotlib.pyplot as plt
import numpy

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
		self.mpc_model = MPC()


	'''
	Function that runs periodically to update the model
	'''
	def run(self):
		for i in range(2, 100):
			# Input vector at step t-1
			input_vector_t_1 = self.construct_input(i-1)
			# State vector at step t-1
			state_vector_t_1 = self.construct_state(i-1)
			# Call mpc kernel to add data
			self.mpc_model.add_data( input_vector_t_1 , state_vector_t_1)

			# Input vector at step t
			input_vector_t = self.construct_input(i)

			# Predict by calling mpc kernel
			prediction = self.mpc_model.predict( input_vector_t )

			# Get model parameters
			params = self.mpc_model.get_model()

			# Do post processing
			self.post_processing(i , prediction, self.construct_state(i)[0], params)


	'''
	Reads data to be supplied to build the model
	'''
	def read_stream_data(self, num_days=1):
		self.points = {}
		c = SmapClient("http://new.openbms.org/backend")
		for point in self.input_variables:
			q = "apply window(mean, field='second',width='%d') to data in (now -%ddays, now) where Metadata/Name='%s'" % \
						( self.rate, num_days, point )

			print q
			result = c.query(q)
			readings = result[0]["Readings"]
			self.points[point] = [ r[1] for r in result[0]["Readings"] ]

		for point in self.state_variables:
			query = "apply window(mean, field='second',width='%d') to data in (now -%ddays, now) where Metadata/Name='%s'" % \
						( self.rate, num_days, point )

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
		for point in self.state_variables:
			input_vector.append( self.points[point][ step - 1] )
		return input_vector

	'''
	Constructs the state vector at a particular timestep
	'''
	def construct_state(self, step):
		state_vector = []
		for point in self.state_variables:
			state_vector.append( self.points[point][ step - 1 ])
		return state_vector

	'''
	Do post processing
	'''
	def post_processing(self, step, prediction, state_t, params ):

		# Do post processing
		print "Prediction at step %d : %f . Actual : %f Error : %f" % \
			( step , prediction[0] , state_t , prediction[0] - state_t )

		self.save_predictions(prediction)
		self.save_model_params(params)
		self.save_actual_output( state_t )

	def save_predictions(self, val):
		self.predictions.append( val )

	def save_model_params(self, val):
		self.model_params.append( val )

	def save_actual_output(self, val):
		self.actual_outputs.append( val )

	def plot(self):
		plt.subplot(2,1,1)
		plt.plot( numpy.arange( len(self.predictions) ), self.predictions, label="predictions")
		plt.plot( numpy.arange( len(self.actual_outputs) ), self.actual_outputs, label="actual-output")
		plt.xlabel("Step")
		plt.legend()

		plt.subplot(2,1,2)
		for i in range(len(self.input_variables) + len(self.state_variables) ):
			stream = None
			if i < len(self.input_variables):
				stream = self.input_variables[i]
			else:
				stream = self.state_variables[ i - len(self.input_variables) ]
			ydata = [ self.model_params[j][i] for j in range(len(self.model_params) ) ]
			plt.plot( numpy.arange( len(self.model_params) ), ydata, label="model_params_" + str(stream))
		plt.xlabel("Step")
		plt.legend()
		plt.show()


if __name__ == "__main__":
	som = SimpleOfflineMPC(sys.argv[1])
	som.run()
	som.plot()