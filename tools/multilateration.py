# file with evaluation methods
import numpy as np
from numpy import size
from numpy import matrix
from numpy.linalg import svd
from numpy import zeros
from numpy import diag
from numpy import reciprocal
from numpy import reshape
from numpy import asarray
from numpy import power
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from math import pow


def multilateration_ls(nodes):
	""" calculate location of node0 from differenct number of anchors
	input nodes matrix:
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	"""
	# get number of anchors
	num_anch = size(nodes[:, 0])

	# build A matrix
	A_arr = []
	for i in range(0, num_anch):
		# -2*xi -2*yi 1
		A_arr.append([-2 * float(nodes[i, 0][0]), -2 * float(nodes[i, 0][1]), 1])
	A = matrix(A_arr)

	# build b matrix
	b_arr = []
	for i in range(0, num_anch):
		# di^2 - xi^2 - yi^2
		b_arr.append([pow(nodes[i, 1], 2) - pow(nodes[i, 0][0], 2) - pow(nodes[i, 0][1], 2)])
	b = matrix(b_arr)

	if matrix_rank(A_arr) > 2:
		singularity = 0

		position = inv(A.T * A) * A.T * b
		position = asarray(reshape(position, (1, len(position))))

	else:
		position = matrix([[0, 0, 0]])
		singularity = 1

	return singularity, position[0][:2]


def multilateration_wls(nodes, weight_vector):
	""" calculate location of node0 from differenct number of anchors
	input nodes matrix:
	+                     +
	| array([x1,y1]) | r1 |
	| array([x2,y2]) | r2 |
	| array([x3,y3]) | r3 |
	| ...            |... |
	| array([xn,yn]) | rn |
	+                     +
	"""
	# get number of anchors
	num_anch = size(nodes[:, 0])

	# build A matrix
	A_arr = []
	for i in range(0, num_anch):
		# -2*xi -2*yi 1
		A_arr.append([-2 * float(nodes[i, 0][0]), -2 * float(nodes[i, 0][1]), 1])
	A = matrix(A_arr)

	# build b matrix
	b_arr = []
	for i in range(0, num_anch):
		# di^2 - xi^2 - yi^2
		b_arr.append([pow(nodes[i, 1], 2) - pow(nodes[i, 0][0], 2) - pow(nodes[i, 0][1], 2)])
	b = matrix(b_arr)

	if matrix_rank(A_arr) > 2:
		singularity = 0
		W = asarray(weight_vector)
		W = 4 * diag(W)
		W = inv(W)

		position = inv(A.T * W * A) * A.T * W * b
		position = asarray(reshape(position, (1, len(position))))

	else:
		position = matrix([[0, 0, 0]])
		singularity = 1

	return singularity, position[0][:2]
