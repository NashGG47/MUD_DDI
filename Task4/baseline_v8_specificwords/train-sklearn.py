#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import argparse
from joblib import dump


def load_data(data):
	features = []
	labels = []
	for interaction in data:
		interaction = interaction.strip()
		interaction = interaction.split('\t')
		interaction_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in interaction[1:] }
		features.append(interaction_dict)
		labels.append(interaction[0])
	return features, labels


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', choices=['nb', 'dt', 'dt70', 'svm'], required=True, help='Model to train')
	parser.add_argument('model_file', help='Output model file')
	parser.add_argument('vectorizer_file', help='Output vectorizer file')

	args = parser.parse_args()

	# model_file = sys.argv[1]
	# vectorizer_file = sys.argv[2]

	model_file = args.model_file
	vectorizer_file = args.vectorizer_file

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	classes = np.unique(y_train)

	v = DictVectorizer()
	X_train = v.fit_transform(train_features)

	# clf = MultinomialNB(alpha=0.01)

	# clf = DecisionTreeClassifier()
	#clf = DecisionTreeClassifier(max_depth=70)	
	# clf = SVC()

	if args.model == 'nb':
		clf = MultinomialNB(alpha=0.01)
	elif args.model == 'dt':
		clf = DecisionTreeClassifier()
	elif args.model == 'dt70':
		clf = DecisionTreeClassifier(max_depth=70)
	elif args.model == 'svm':
		clf = LinearSVC(max_iter=2500)

	clf.fit(X_train, y_train)

	# clf.partial_fit(X_train, y_train, classes)

	#Save classifier and DictVectorizer
	dump(clf, model_file) 
	dump(v, vectorizer_file)

	 