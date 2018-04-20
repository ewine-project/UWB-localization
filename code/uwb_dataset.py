"""
Created on Apr 20, 2018

@author: Klemen Bregar 
"""

import os
import pandas as pd
from numpy import vstack


def load_data_from_file(filepath):
	"""
	Read selected .csv file to numpy array
	"""
	output_arr = []
	print("Loading " + filepath + "...")
	# read data from file
	df = pd.read_csv(filepath, sep=',', header=0)
	output_arr = df.as_matrix()

	return output_arr


def load_data_from_folder(folderpath):
	"""
	Read selected .csv file to numpy array
	"""
	rootdir = folderpath
	output_arr = []
	first = 1
	for dirpath, dirnames, filenames in os.walk(rootdir):
		for file in filenames:
			filename = os.path.join(dirpath, file)
			print("Loading " + filename + "...")
			output_data = []
			# read data from file
			df = pd.read_csv(filename, sep=',', header=0)
			input_data = df.as_matrix()
			# append to array
			if first > 0:
				first = 0
				output_arr = input_data
			else:
				output_arr = vstack((output_arr, input_data))

	return output_arr


if __name__ == '__main__':
	# import raw data from folder with dataset
	print("Importing datasets to numpy array")
	print("-------------------------------")
	print("dataset1: ")
	data1 = load_data_from_file('../dataset1/tag_room0.csv')
	print("shape0: %d" % data1.shape[0])
	print("shape1: %d" % data1.shape[1])
	data2 = load_data_from_file('../dataset1/tag_room1.csv')
	print("shape0: %d" % data2.shape[0])
	print("shape1: %d" % data2.shape[1])
	print("dataset2: ")
	data3 = load_data_from_file('../dataset2/tag_room0.csv')
	print("shape0: %d" % data3.shape[0])
	print("shape1: %d" % data3.shape[1])
	data4 = load_data_from_folder('../dataset2/tag_room1/')
	print("shape0: %d" % data4.shape[0])
	print("shape1: %d" % data4.shape[1])
	print("-------------------------------")

