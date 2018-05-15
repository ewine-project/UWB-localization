import multilateration as mlt
import numpy as np
import random


def get_cir(data):
	""" cut 152 bytes of cir from data """
	fp_idx = int(data[8])

	cir = np.array([data[fp_idx + 15: fp_idx + 15 + 152] / data[17]])

	return cir


def select_anchors_classification(data, n, classifier):
	"""randomly selects n different anchors for evaluation. Only LoS-classified anchors are selected

	Returns:
	--------
	nodes: numpy matrix with nodes positions and ranges
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	"""
	nodelist = []

	# node search
	for i in range(0, n):
		stat = 1
		while stat > 0:
			# random number
			randidx = random.randint(0, len(data) - 1)
			# get anchor position
			temp_node = np.array([data[randidx][2], data[randidx][3]])

			eq_stat = 0

			# check for LOS/NLOS with classifier
			filt = classifier.Classify(get_cir(data[randidx]))

			# add anchor if appropriate
			if np.size(nodelist) == 0:
				if filt != 1:
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					nodelist.append([temp_node, temp_len])
					stat = 0
			else:
				# check for equal elements in matrix
				for item in nodelist:
					eq_stat = eq_stat | (np.linalg.norm(temp_node) == np.linalg.norm(item[0]))
				if (filt == 0) and (eq_stat != 1):
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					nodelist.append([temp_node, temp_len])
					stat = 0

	nodes = np.matrix(nodelist)

	return nodes


def select_anchors(data, n, NLOS):
	"""randomly selects n different anchors for evaluation. If NLOS is 1, LOS and NLOS measurements are used for evaluation

	Returns:
	--------
	nodes: numpy matrix with nodes positions and ranges
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	"""
	nodelist = []

	# node search
	for i in range(0, n):
		stat = 1
		while stat > 0:
			# random number
			randidx = random.randint(0, len(data) - 1)
			# get anchor position
			temp_node = np.array([data[randidx][2], data[randidx][3]])

			eq_stat = 0

			filt = 1
			# check LOS/NLOS
			if NLOS & int(data[randidx][5]):
				filt = 0
			if NLOS & (not int(data[randidx][5])):
				filt = 0
			if (not NLOS) & (not int(data[randidx][5])):
				filt = 0

			# add anchor if appropriate
			if np.size(nodelist) == 0:
				if filt != 1:
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					nodelist.append([temp_node, temp_len])
					stat = 0
			else:
				# check for equal elements in matrix
				for item in nodelist:
					eq_stat = eq_stat | (np.linalg.norm(temp_node) == np.linalg.norm(item[0]))
				if (filt == 0) and (eq_stat != 1):
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					nodelist.append([temp_node, temp_len])
					stat = 0

	nodes = np.matrix(nodelist)

	return nodes


def select_anchors_wls(data, n, regressor, NLOS):
	"""randomly selects n different anchors for evaluation. If NLOS is 1, LOS and NLOS measurements are used for evaluation

	Returns:
	--------
	nodes: numpy matrix with nodes positions and ranges
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	weight_vector: vector with predicted range errors
	"""
	nodelist = []
	weight_vector = []

	# node search
	for i in range(0, n):
		stat = 1
		while stat > 0:
			# random number
			randidx = random.randint(0, len(data) - 1)
			# get anchor position
			temp_node = np.array([data[randidx][2], data[randidx][3]])

			eq_stat = 0

			filt = 1
			# check LOS/NLOS
			if NLOS & int(data[randidx][5]):
				filt = 0
			if NLOS & (not int(data[randidx][5])):
				filt = 0
			if (not NLOS) & (not int(data[randidx][5])):
				filt = 0

			# add anchor if appropriate
			if np.size(nodelist) == 0:
				if filt != 1:
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					regress_err = regressor.predict(get_cir(data[randidx]))
					nodelist.append([temp_node, temp_len])
					weight_vector.append(regress_err[0][0])
					stat = 0
			else:
				# check for equal elements in matrix
				for item in nodelist:
					eq_stat = eq_stat | (np.linalg.norm(temp_node) == np.linalg.norm(item[0]))
				if (filt == 0) and (eq_stat != 1):
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					regress_err = regressor.predict(get_cir(data[randidx]))
					nodelist.append([temp_node, temp_len])
					weight_vector.append(regress_err[0][0])
					stat = 0

	nodes = np.matrix(nodelist)

	return nodes, weight_vector


def select_anchors_regression_wls(data, n, regressor, NLOS):
	"""randomly selects n different anchors for evaluation. If NLOS is 1, LOS and NLOS measurements are used for evaluation

	Returns:
	--------
	nodes: numpy matrix with nodes positions and ranges
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	weight_vector: vector with predicted range errors
	"""
	nodelist = []
	weight_vector = []

	# node search
	for i in range(0, n):
		stat = 1
		while stat > 0:
			# random number
			randidx = random.randint(0, len(data) - 1)
			# get anchor position
			temp_node = np.array([data[randidx][2], data[randidx][3]])

			eq_stat = 0

			filt = 1
			# check LOS/NLOS
			if NLOS & int(data[randidx][5]):
				filt = 0
			if NLOS & (not int(data[randidx][5])):
				filt = 0
			if (not NLOS) & (not int(data[randidx][5])):
				filt = 0

			# add anchor if appropriate
			if np.size(nodelist) == 0:
				if filt != 1:
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					regress_err = regressor.predict(get_cir(data[randidx]))
					temp_len = temp_len - regress_err
					nodelist.append([temp_node, temp_len])
					weight_vector.append(regress_err)
					stat = 0
			else:
				# check for equal elements in matrix
				for item in nodelist:
					eq_stat = eq_stat | (np.linalg.norm(temp_node) == np.linalg.norm(item[0]))
				if (filt == 0) and (eq_stat != 1):
					# get measured anchor range to tag
					temp_len = data[randidx][4]
					regress_err = regressor.predict(get_cir(data[randidx]))
					temp_len = temp_len - regress_err
					nodelist.append([temp_node, temp_len])
					weight_vector.append(regress_err)
					stat = 0

	nodes = np.matrix(nodelist)

	return nodes, weight_vector


def localization_los_nlos_ls(input_data, tag_pos, num_anch, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes = select_anchors(input_data, num_anch, True)
			position = mlt.multilateration_ls(nodes)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect


def localization_los_ls(input_data, tag_pos, num_anch, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes = select_anchors(input_data, num_anch, False)
			position = mlt.multilateration_ls(nodes)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect


def localization_ls_classification(input_data, tag_pos, num_anch, classifier, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes = select_anchors_classification(input_data, num_anch, classifier)
			position = mlt.multilateration_ls(nodes)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect


def localization_ls_regression(input_data, tag_pos, num_anch, regressor, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes, weights = select_anchors_regression_wls(input_data, num_anch, regressor, True)
			position = mlt.multilateration_ls(nodes)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect


def localization_wls(input_data, tag_pos, num_anch, regressor, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes, weights = select_anchors_wls(input_data, num_anch, regressor, True)
			position = mlt.multilateration_wls(nodes, weights)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect


def localization_wls_regression(input_data, tag_pos, num_anch, regressor, numit):
	"""
	This function estimates location of a tag based on anchor positions and ranges

	:param input_data: dataset array
	:param tag_pos: tag position vector
	:param num_anch: number of anchors used for evaluation
	:param numit: number of iterations
	:return err_vect: vector of numit evaluation errors
	"""
	err_vect = []

	for i in range(0, numit):
		# calculate position error
		singular = 1
		# keep finding anchors until matrix is not singular
		while singular:
			nodes, weights = select_anchors_regression_wls(input_data, num_anch, regressor, True)
			position = mlt.multilateration_wls(nodes, weights)
			temp_err = 0.
			# if anchor matrix is not singular keep result
			if position[0] == 0:
				singular = 0
				temp_err = np.linalg.norm(tag_pos - position[1])
				err_vect.append(temp_err)

	return err_vect
