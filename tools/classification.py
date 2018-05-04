# file with classification methods
import tensorflow as tf
from sklearn.externals import joblib
import os
import sys


# sys.path.insert(0, './tf_models')
class CNNClassifier(object):
	""" implements convolutional neural network classifier implemented and learned in TensorFlow"""

	def __init__(self):
		# dirpath = os.path.dirname(__file__)

		self.sess = tf.Session()

		# import trained graph from file
		self.saver = tf.train.import_meta_graph('../NLOSClassificationModel/checkpoint-19000.meta')
		self.saver.restore(self.sess, '../NLOSClassificationModel/checkpoint-19000')

		# init/load inputs, outputs and settings of a graph
		self.logits = tf.get_collection("logits")[0]
		self.x = tf.get_collection("inputs")[0]
		self.labels = tf.get_collection("labels")[0]
		self.keep_prob = tf.get_collection("keep_prob")[0]
		# eval output from classifier
		self.eval_op = tf.nn.top_k(self.logits)
		# define "placeholder" for output
		self.label_feed = [[0, 0]]
		# load scaler
		self.scaler = joblib.load(('../NLOSClassificationModel/scaler_152_real.pkl'))

	def Classify(self, sample_feed):
		self.prediction = self.sess.run(self.eval_op, feed_dict={self.x: self.Scale(sample), self.labels: self.label_feed,
																 self.keep_prob: 1.0})
		return self.prediction.indices[:][:]

	def Scale(self, sample_feed):
		return self.scaler.transform(sample_feed)

	def CloseSession(self):
		self.sess.close()

