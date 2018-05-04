# file with classification methods
import tensorflow as tf
from sklearn.externals import joblib
import os
import sys

class Regression(object):
	"""
	Estimate range error based on CIR and
	correct the range with predicted error.
	"""

	def __init__(self):
		self.sess = tf.Session()
		# import trained graph from file
		self.saver = tf.train.import_meta_graph("../RangingErrorModel/checkpoint_regression-19999.meta")
		self.saver.restore(self.sess, "../RangingErrorModel/checkpoint_regression-19999")
		# init/load inputs, outputs and settings of a graph
		self.logits = tf.get_collection("logits")[0]
		self.x = tf.get_collection("inputs")[0]
		self.labels = tf.get_collection("labels")[0]
		self.keep_prob = tf.get_collection("keep_prob")[0]
		# eval output from classifier
		self.regression_op = self.logits
		# define "placeholder" for output
		self.label_feed = [[0.0]]
		# load scaler
		self.scaler = joblib.load(('../RangingErrorModel/scaler_152_regression.pkl'))

	def predict(self, sample_feed):
		self.prediction = self.sess.run(self.regression_op,
										feed_dict={self.x: self.Scale(sample_feed), self.labels: self.label_feed,
												   self.keep_prob: 1.0})
		return self.prediction[:][:]

	def Scale(self, sample_feed):
		return self.scaler.transform(sample_feed)

	def closeSession(self):
		self.sess.close()