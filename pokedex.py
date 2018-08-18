
import numpy as np
from sklearn.externals import joblib
from sklearn 			import cluster
from sklearn.svm 		import SVC
from sklearn.ensemble 	import BaggingClassifier
from sklearn.tree 	import DecisionTreeClassifier
from sklearn.ensemble 	import VotingClassifier
from sklearn.neighbors 	import KNeighborsClassifier
from sklearn.neural_network 	import MLPClassifier 
from sklearn.neural_network 	import MLPRegressor 

from sklearn.cross_validation import train_test_split


TRUE = 'TRUE'
FALSE = 'FALSE'

PARAMS = {	\
			'gamma'			: 0.001, 
			'C'				: 12, 
			'svc_vote'		: 0.6,
			'n_neighbors'	: 7,
			'n_estimators'	: 5,
			'max_samples'	: 0.5,
			'max_features'	: 0.5,
			'random_state'	: 1,
			'bag_vote'		: 0.5}
		


class Pokedex:

	#class Bill(:
	#custom MLP Regressor/Tensor? to fit classifier to higher score	
	def __init__(self, pkmn, classifier_dumpfile 	= 'classifier.pkl', cluster_dumpfile 		= 'cluster.pkl'):
		
		
		self.pkmn = pkmn
		self.caught = None 	# training pokemon
		self.wild = None 		# testing pkmn
		'''
		 Classifier  
				Classifies and Regresses features unto labels or quantifiers called expectations
		'''
		self.classifier_ensemble = \
		{
			# id 	: 	(classifier,vote_amt)

			'dt'	: 	(
							DecisionTreeClassifier(max_depth=3),
							0.3
						) ,	
			'mlp'	:	(
						MLPClassifier(hidden_layer_sizes =[5,2], alpha=0.0008, solver = 'lbfgs', learning_rate ='constant',
						activation = 'relu',
							max_iter=240
							),
							0.5
						),
			# 'svc'	: 	(
							# SVC(gamma = PARAMS['gamma'], C= PARAMS['C'],probability=True),
							# PARAMS['svc_vote']
						# ) ,	
			'bag'	: 	(
						BaggingClassifier(	KNeighborsClassifier(n_neighbors=PARAMS['n_neighbors']), 
											n_estimators=PARAMS['n_estimators'], max_samples=PARAMS['max_samples'], 
											max_features=PARAMS['max_features'], random_state=PARAMS['random_state']),
						PARAMS['bag_vote']*2
						)
	
		}
		# initialize classifier from ensemble
		self.checkBag()
		# create a NN to train / (load from file ) synapes to minimize _ensemble.score()
		#self.
		
		'''
			Cluster \
				Will cluster the features
		'''
		self.cluster_ensemble	= \
		{
			#get labels with labels_
			'kmn' 	: cluster.KMeans(algorithm='auto', copy_x=True, init='k-means++')
		}
		self.cluster = self.cluster_ensemble['kmn']

		
	def dump(self):
		try:
			joblib.dump(self.classifier, self.classifier_dumpfile)
			joblib.dump(self.cluster, self.cluster_dumpfile)
		except:
			return
			
	def load(self):
		try:
			self.classifier  	= joblib.load(self.classifier_dumpfile)
			self.cluster  		= joblib.load(self.cluster_dumpfile)
		except:
			return	
	
	def checkBag(self):
		estimators = list(zip(self.classifier_ensemble.keys(),[ c for c,v in self.classifier_ensemble.values() ]))
		vote_amts = [ v for c,v in self.classifier_ensemble.values() ]
		self.classifier = VotingClassifier(estimators, voting='soft', weights=vote_amts)
	
	# 	----			Classification methods		----
	def catchem(self, ratio):
		'''
		Fit data
			ratio : float	
				Ratio of set to be traninig, remaining are test data 
		'''
		
		self.caught, self.wild = train_test_split(self.pkmn, train_size=ratio)

		
	# have features be a list of IDS for data[HP,] OUT[ID] to use 
	def train(self, features, expectations):
		'''	
			Check the classifiers for a pokemon. must call catchem first
		'''									# 
		pkmn_features 		= self.getrows(self.caught, features)
		pkmn_expectations 	= self.getrows(self.caught, [expectations])

		return self.classifier.fit(pkmn_features, pkmn_expectations)
	
	
	
	def score(self, features, expectations):
		'''	
			Check score to your currnet PC  
		'''
		pkmn_features 		= self.getrows(self.wild, features)
		pkmn_expectations 	= self.getrows(self.wild, [expectations])

		return self.classifier.score(pkmn_features, pkmn_expectations)
	
	
	## Neural Net methods
	def recover(self):
		'''
			Fit Regressor to modify classifier paramss
		'''
		#self.classifier_ensemble <- fit params
		# use Bill to fit on current, adjust(1-score) on each param
	

	
	# 	----			Clustering  methods		----
	def evaluate():
		return

	
	def getrows(self, data, cols):
		rows = []
		for id, type in cols:
			datum  = data[id]
			if datum.size == 1:
				rows.append(datum)
			else:
				rows.append(np.array(datum, dtype=type))
		

		return np.array(rows).T