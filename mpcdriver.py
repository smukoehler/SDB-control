import urlparse
import datetime
import urllib2
from smap.driver import SmapDriver
from smap.util import periodicSequentialCall
from smap.contrib import dtutil
from sklearn import linear_model
from smap.archiver.client import RepublishClient
from functools import partial
from mpc import *

class SimpleMPC(SmapDriver):
	def setup(self, opts):
		self.rate = float( opts.get('rate' , 120 ) )
		self.archiver_url = opts.get('archiver')
		self.input_variables = opts.get('input_variables', None)
		self.state_variables = opts.get('state_variables', None).split(',')
		self.read_stream_data()
		self.setup_model()
	'''
	Create MPC kernel
	'''
	def setup_model(self):
		self.mpc_model = MPC()

	'''
	Function that runs periodically to update the model
	'''
	def start(self):
		self._loop = periodicSequentialCall(self.predict)
		self._loop.start(self.rate)
		for clientlist in self.repubclients.itervalues():
			for c in clientlist:
				c.connect()


	def predict(self):
		# Input vector at step t-1
		input_vector_t_1 = self.construct_input(-1)
		# State vector at step t-1
		state_vector_t_1 = self.construct_state(-1)

		if input_vector_t_1 == None or state_vector_t_1 == None:
			return

		# Call mpc kernel to add data
		self.mpc_model.add_data( input_vector_t_1 , state_vector_t_1 )

		# Input vector at time t
		input_vector_t = self.construct_input(0)

		# predict by calling at mpc kernel
		prediction = self.mpc_model.predict( input_vector_t )

		# Get model parameters
		params = self.mpc_model.get_model()

		# Do post processing
		self.post_processing(i , prediction, self.construct_state(i)[0], params)


	'''
	Reads data to be supplied to build the model
	'''
	def read_stream_data(self):
		self.points = {}
		self.repubclients = {}	

		for name in self.input_variables:
			point = name.strip()
			self.points[point] = []
			self.repubclients[point] = [RepublishClient(self.archiver_url, partial(self.cb, point), restrict="Metadata/Name = '" + str(point) + "'")]
		for name in self.state_variables:
			point = name.strip()
			self.points[point] = []
			self.repubclients[point] = [RepublishClient(self.archiver_url, partial(self.cb, point), restrict="Metadata/Name = '" + str(point) + "'")]

	def cb(self, point, _, data):
		value = data[-1][-1][1]
		print 'Received',point,'=',value
		self.points[point].append(value)

	'''
	Constructs an input vector at a particular timestep
	'''
	def construct_input(self, step):
		input_vector = []
		try:
			for point in self.input_variables:
				input_vector.append( self.points[point][ step - 1] )
			for point in self.state_variables:
				input_vector.append( self.points[point][ step - 2] )
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

	'''
	Do post processing
	'''
	def post_processing(self, step, prediction, state_t, params ):

		# Do post processing
		for i in range(len(self.state_variables)):
			self.add('/' + self.state_variables[i] + "-predicted" , prediction[i] )
			for j in range(len(self.input_variables)):
				self.add('/' + self.state_variables[i] + "-mpc-param-effect-of-" + self.input_variables[j],  params[j])
			for j in range(len(self.state_variables)):
				self.add('/' + self.state_variables[i] + "-mpc-param-effect-of-" + self.state_variables[j], params[ len(self.input_variables) + j])


