from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os

tf.disable_eager_execution()

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0

		# Whether to use normalization over a batch or a population
		# self.is_train = tf.placeholder(tf.bool, name='is_train')
		self.is_train = tf.constant(False)

		# input image batch
		# self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
		self.inputImgs = tf.placeholder(tf.float32, shape=(1, Model.imgSize[0], Model.imgSize[1], 1), name='input')

		# setup CNN, RNN and CTC
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()

		# setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self):
		"create CNN layers and return output of these layers"
		# cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
		cnnIn4d = self.inputImgs

		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			# conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			conv_norm = tf.layers.batch_normalization(conv, training=False)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		self.cnnOut4d = pool


	def setupRNN(self):
		"create RNN layers and return output of these layers"
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			# self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=tf.constant([32], tf.int32))
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			# prepare information about language (dictionary, characters in dataset, characters forming words) 
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()

			# decode using the "Words" mode of word beam search
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	def dumpNNOutput(self, rnnOutput):
		"dump the output of the NN to CSV file(s)"
		dumpDir = '../dump/'
		if not os.path.isdir(dumpDir):
			os.mkdir(dumpDir)

		# iterate over all batch elements and create a CSV file for each one
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)
			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
		numBatchElements = len(batch.imgs)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		batch.imgs = np.expand_dims(batch.imgs, axis=3)
		# feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements}
		print(self.inputImgs)
		print(batch.imgs.shape)



		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		#===========================================================
		print('self.inputImgs.name:', self.inputImgs.name)
		print('self.inputImgs.shape:', self.inputImgs.shape)
		print('self.inputImgs:', self.inputImgs)

		# print('self.decoder.name:', self.decoder.name)
		# print('self.decoder.shape:', self.decoder.shape)
		print('self.decoder:', self.decoder)
		#===========================================================

		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			# feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		#===========================================================
		# print('self.lossPerElement.name:', self.lossPerElement.name)
		# print('self.lossPerElement.shape:', self.lossPerElement.shape)
		print('self.lossPerElement:', self.lossPerElement)

		# saver = tf.train.Saver()
		# saver.save(self.sess, '../model/model.ckpt')

		tf.train.write_graph(self.sess.graph_def, '../model', 'simpleHTR_freeze_graph.pb', as_text=False)
		tf.train.write_graph(self.sess.graph_def, '../model', 'simpleHTR_freeze_graph.pbtxt', as_text=True)

		input_name="input"
		outputs = ['CTCGreedyDecoder:0','CTCGreedyDecoder:1','CTCGreedyDecoder:2','CTCGreedyDecoder:3']
		tf.compat.v1.saved_model.simple_save(
			self.sess,
			'../model/saved_model',
			inputs={input_name: self.sess.graph.get_tensor_by_name('{}:0'.format(input_name))},
			# outputs={t.rstrip(":0").rstrip(":3"):self.sess.graph.get_tensor_by_name(t) for t in outputs}
			outputs={t:self.sess.graph.get_tensor_by_name(t) for t in outputs}
		)

		import sys
		sys.exit(0)
		#===========================================================

		# dump the output of the NN to CSV file(s)
		if self.dump:
			self.dumpNNOutput(evalRes[1])

		return (texts, probs)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
