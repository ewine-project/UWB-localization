# assessing impact of NLOS measurements on 2-D position accuracy
import sys
sys.path.append("../tools/")

import localization as loc
import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
	start_time = time.time()

	file1 = '../data/localization/dataset1/tag_room0.csv'
	file2 = '../data/localization/dataset1/tag_room1.csv'
	file3 = '../data/localization/dataset2/tag_room0.csv'

	# read data from file
	print("Loading: %s" % file1)
	df = pd.read_csv(file1, sep=',', header=None)
	input_data1 = df.as_matrix()
	print("Loading: %s" % file2)
	df = pd.read_csv(file2, sep=',', header=None)
	input_data2 = df.as_matrix()
	print("Loading: %s" % file3)
	df = pd.read_csv(file3, sep=',', header=None)
	input_data3 = df.as_matrix()

	# get node position
	tag_pos1 = [input_data1[0][0], input_data1[0][1]]
	print(tag_pos1)
	tag_pos2 = [input_data2[0][0], input_data2[0][1]]
	print(tag_pos2)
	tag_pos3 = [input_data3[0][0], input_data3[0][1]]
	print(tag_pos3)

	numit = 3333

	# LS (LOS classification)
	for a in range(3, 11):

		print("%d anchors" % a)
		err_vect = []

		temp_vect1 = loc.localization_los_nlos_ls(input_data1, tag_pos1, a, numit)
		temp_vect2 = loc.localization_los_nlos_ls(input_data2, tag_pos2, a, numit)
		temp_vect3 = loc.localization_los_nlos_ls(input_data3, tag_pos3, a, numit)
		err_vect = np.concatenate((temp_vect1, temp_vect2, temp_vect3))

		# calculate error statistics
		max_err_ls_class = np.amax(err_vect)
		min_err_ls_class = np.amin(err_vect)
		meanerr_ls_class = np.mean(err_vect)
		med_err_ls_class = np.median(err_vect)
		stddev_ls_class = np.std(err_vect)
		print("mean:%f median: %f stddev: %f min_err: %f max_err: %f" % (
		meanerr_ls_class, med_err_ls_class, stddev_ls_class, min_err_ls_class, max_err_ls_class))
		print("#####################################")

	print("--- %s seconds ---" % (time.time() - start_time))