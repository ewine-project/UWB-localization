import collections
import numpy as np
import pandas as pd
import math
import os
from numpy import arange
from numpy import random
from numpy import zeros
from numpy import random
from numpy import split
from numpy import array
from numpy import size
from numpy import reshape
from numpy import vstack
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


class DataSet(object):

	def __init__(self, samples, labels, reshape=False, dtype=False):
		"""Construct a DataSet"""
		assert samples.shape[0] == labels.shape[0], (
				'images.shape: %s labels.shape: %s' % (samples.shape, labels.shape))
		self._num_examples = samples.shape[0]

		# Convert shape from [num examples, rows, columns, depth]
		# to [num examples, rows*columns] (assuming depth == 1)
		if reshape:
			assert samples.shape[3] == 1
			samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])
		self._samples = samples
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		# Shuffle the data
		perm = arange(self._num_examples)
		random.shuffle(perm)
		self._samples = self._samples[perm]
		self._labels = self._labels[perm]

	@property
	def samples(self):
		return self._samples

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = arange(self._num_examples)
			random.shuffle(perm)
			self._samples = self._samples[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._samples[start:end], self._labels[start:end]


def import_raw_from_files(data_dir):
	"""
		Read input files and store data into an array
		format: |LOS|NLOS|data...|
	"""
	# search dir for files
	rootdir = data_dir
	output_arr = []
	first = 1
	for dirpath, dirnames, filenames in os.walk(rootdir):
		for file in filenames:
			filename = os.path.join(dirpath, file)
			print(filename)
			output_data = []
			# read data from file
			df = pd.read_csv(filename, sep=',', header=1)
			input_data = df.as_matrix()
			# 2 for label; omit last field because it is empty (nan)
			output_data = zeros((len(input_data), 1 + input_data.shape[1] - 1))
			# set the NLOS status from filename
			for i in range(input_data.shape[0]):
				if input_data[i, 0] == 0:
					output_data[i, 0] = 1
				else:
					output_data[i, 0] = 0
			# put data into output array
			output_data[:, 1:] = input_data[:, :-1]
			if first > 0:
				first = 0
				output_arr = output_data
			else:
				output_arr = vstack((output_arr, output_data))

	return output_arr


def import_raw_csv_file(file_path):
	"""
	Read input .csv file to numpy array.

	Parameters
	----------
	file_path : str
		absolute path to input .csv file

	Returns
	-------
	output_arr : numpy.array
		array of values from .csv file
	"""
	print(file_path)
	# read data from file
	df = pd.read_csv(file_path, sep=',', header=None)
	output_arr = df.as_matrix()

	return output_arr


# parse input data from input file format
def import_localization_data(file_paths):
	"""
	Calculate ranging error and import CIR data

	Parameters
	----------
	file_path : str
		absolute path to input .csv file

	Returns
	-------
	error_arr : numpy.array
		array of ranging errors from input data
	cir_arr : numpy.array
		array of CIR vectors from .csv file (length=152)
	"""
	# import from file
	input_arr = import_raw_csv_file(file_paths[0])
	# randomize input array
	random.shuffle(input_arr)

	# create blank output_arrays
	error_arr = zeros((len(input_arr), 1))
	cir_arr = zeros((len(input_arr), 152))

	for i in range(len(input_arr)):
		fp_idx = int(input_arr[i][6])
		# calculate ranging error
		error_arr[i] = math.fabs(math.sqrt(math.pow(input_arr[i][1] - input_arr[i][3], 2) +
										   math.pow(input_arr[i][2] - input_arr[i][4], 2)) - input_arr[i][5])
		# pack cir to output cir array
		cir_arr[i] = input_arr[i][fp_idx + 15: fp_idx + 15 + 152] / input_arr[i][12]

	if len(file_paths) > 1:
		for item in file_paths[1:]:
			temp = import_raw_csv_file(item)
			# randomize input array
			random.shuffle(temp)

			# create blank output_arrays
			error_temp = zeros((len(temp), 1))
			cir_temp = zeros((len(temp), 152))

			for i in range(len(temp)):
				fp_idx = int(temp[i][6])
				# calculate ranging error
				error_temp[i] = math.fabs(math.sqrt(math.pow(temp[i][1] - temp[i][3], 2) +
													math.pow(temp[i][2] - temp[i][4], 2)) - temp[i][5])
				# pack cir to output cir array
				cir_temp[i] = temp[i][fp_idx + 15: fp_idx + 15 + 152] / temp[i][12]

			error_arr = np.vstack((error_arr, error_temp))
			cir_arr = np.vstack((cir_arr, cir_temp))

	return error_arr, cir_arr


def import_cir(data_loc, data_len=152):
	"""
		import data from folder, complex is for complex data files import
	"""
	# import data from files
	data = import_raw_from_files(data_loc)

	output_data = zeros((len(data), 2 + data_len))
	# labels
	output_data[:, 0:2] = data[:, 0:2]
	# read only data_len samples of CIR from the first path index
	for i in range(len(data)):
		fp_index = int(data[i, 3])
		output_data[i, 2:] = data[i, (fp_index + 2 + 10):(fp_index + 2 + 10 + data_len)] / float(data[i, 10])

	# randomize array
	random.shuffle(output_data)
	random.shuffle(output_data)
	random.shuffle(output_data)
	# return data and labels in separate arrays
	return output_data[:, 2:], output_data[:, 0:2]


def nlos_classification_dataset(raw_data_location, split_factor=0.6, scaling=False):
	"""
		raw_data_location: path to files with raw data
		complex: import complex CIR measurements
		split_factor: ratio between learning and test data set size (from 0 to 1)
		preprocessing: data normalization etc.
	"""
	Datasets = collections.namedtuple('Datasets', ['train', 'test'])
	data_in, labels = import_cir(raw_data_location)

	# split and reshape data
	train_data, test_data = split(data_in, array([int(size(data_in, 0) * split_factor)]))
	train_labels, test_labels = split(labels, array([int(size(labels, 0) * split_factor)]))
	train_labels = reshape(train_labels, (len(train_labels), 2))
	test_labels = reshape(test_labels, (len(test_labels), 2))

	# scale data if desired
	# scale data
	if scaling == True:
		scaler = StandardScaler()
		scaler.fit(train_data)
		joblib.dump(scaler, '../NLOSClassificationModel/scaler_152_real.pkl')
		train_data = scaler.transform(train_data)
		test_data = scaler.transform(test_data)

	train = DataSet(train_data, train_labels)
	test = DataSet(test_data, test_labels)

	return Datasets(train=train, test=test)


def error_regression_dataset(file_paths_list, split_factor=0.6, scaling=False):
	"""
	Import error regression dataset
	:param file_paths_list: list of file paths to load
	:param split_factor:
	:param scaling:
	:return:
	"""
	Datasets = collections.namedtuple('Datasets', ['train', 'test'])
	# get data from files
	error_arr, cir_arr = import_localization_data(file_paths_list)

	# split and reshape data
	train_data, test_data = split(cir_arr, array([int(size(cir_arr, 0) * split_factor)]))
	train_labels, test_labels = split(error_arr, array([int(size(error_arr, 0) * split_factor)]))
	train_labels = reshape(train_labels, (len(train_labels), 1))
	test_labels = reshape(test_labels, (len(test_labels), 1))

	# scale data if desired
	# scale data
	if scaling == True:
		scaler = StandardScaler()
		scaler.fit(train_data)
		# dump standard scaler to file
		joblib.dump(scaler, '../RangingErrorModel/scaler_152_regression.pkl')
		train_data = scaler.transform(train_data)
		test_data = scaler.transform(test_data)

	train = DataSet(train_data, train_labels)
	test = DataSet(test_data, test_labels)

	return Datasets(train=train, test=test)


def act_dist(data):
	""" calculate real distance from input data """
	dist = math.sqrt(math.pow(data[1] - data[3], 2) + math.pow(data[2] - data[4], 2) + math.pow(tag_h - anch_h, 2))
	return dist
