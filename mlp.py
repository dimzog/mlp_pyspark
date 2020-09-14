
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, date_format
from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import re
import math
import time

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

# Manually adding bad occurences
stopwords.append('would')
stopwords.append('ta')

# AMs: 03400043, 03400050
seed = 4350
np.random.seed(seed)

# Map subcategories to greater ones
mappings = {
		'Bank account or service': 'bank account or service',
		'Checking or savings account': 'checking or savings account',
		'Consumer Loan': 'consumer loan',
		'Credit card': 'credit card or prepaid card',
		'Credit card or prepaid card': 'credit card or prepaid card',
		'Credit reporting': 'credit reporting credit repair services or other personal consumer reports',
		'Credit reporting credit repair services or other personal consumer reports': 'credit reporting credit repair services or other personal consumer reports',
		'Debt collection': 'debt collection',
		'Money transfers': 'money transfer virtual currency or money service',
		'Mortgage': 'mortgage',
		'Payday loan': 'payday loan title loan or personal loan',
		'Payday loan title loan or personal loan': 'consumer loan',
		'Prepaid card': 'credit card or prepaid card',
		'Student loan': 'student loan',
		'Vehicle loan or lease': 'vehicle loan or lease',
		'Virtual currency': 'money transfer virtual currency or money service',
		'Money transfer virtual currency or money service': 'money transfer virtual currency or money service'
	}


# Manually remove categories having small sample size

# Bad categories for 250k, 500k, 1m rows
bad_categories = ['Other financial service', 'bank account or service']

# Bad categories for 2m rows
#bad_categories = ['money transfer virtual currency or money service', 'vehicle loan or lease', 'Other financial service']


def clear_comment(text):
	"""
	Clears the comments: Removes punctuation, strange characters, digits, empty strings, stopwords,
	and keeps unique words in each comment

	Params:
	text (string): the comment from a customer

	Returns:
	out (string): Cleared comment
	"""

	out = text.lower()

	# Keep only words containing Characters
	out = re.sub('[^a-z]+|[xx]+', ' ', out)

	# Lemmatize words (e.x eggs -> egg) and remove stopwords
	out = ' '.join([lemmatizer.lemmatize(t) for t in nltk.word_tokenize(out) if t not in stopwords and len(t)>1])

	return out



if __name__ == '__main__':

	spark = SparkSession.builder.appName('part2_ml').getOrCreate()
	sc = spark.sparkContext

	# Load Data
	data = sc.textFile('hdfs://master:9000/project/customer_complaints_500k.csv').\
			map(lambda x: x.split(','))

	# Filter Data -> output format: (category, comment)
	clean_data = data.filter(lambda x: (len(x) == 3) and (x[0].startswith('201')) and (x[1] != '') and (x[2] != '') and (x[1] not in bad_categories)).\
				map(lambda x: (mappings[x[1]], clear_comment(x[2]))).\
				filter(lambda x: x[1] != '').\
				cache()


	# Find most common words, keep k of them -> output format: (word, count)
	k = 500
	vocab = clean_data.flatMap(lambda x: x[1].split(' ')).\
			map(lambda x: (x, 1)).\
			reduceByKey(lambda x, y: x+y).\
			sortBy(lambda x: x[1], ascending=False).\
			map(lambda x: x[0]).\
			take(k)


	# Broadcast to all
	shared_vocab = sc.broadcast(vocab)

	# output format: ((category, comment), doc_id)
	keep = clean_data.map(lambda x: (x[0], x[1].split(' '))).\
			map(lambda x: (x[0], [y for y in x[1] if y in shared_vocab.value])).\
			filter(lambda x: x[1] != '').\
			zipWithIndex().\
			cache()

	# Release Memory
	clean_data.unpersist()

	# Count number of Total Documents
	N = keep.count()

	print('\nTotal Documents in Dataset: {}.\n'.format(N))

	# Calculate IDF -> output format: (word, IDF)
	idf = keep.flatMap(lambda x: [(y, 1) for y in list(set(x[0][1]))]).\
			reduceByKey(lambda x, y: x+y).\
			map(lambda x: (x[0], math.log(N/x[1]))).\
			collect()

	# Broadcast it, so tf can use it
	shared_idf = sc.broadcast(idf)

	# Calculate TF-IDF
	# flatMap -> output format: ((word, category, doc_id, len_of_comment), 1)
	# reduceByKey -> output format: ((word, category, doc_id, len_of_comment), count)
	# map -> output format: ((word, category, doc_id), (count/len_of_comment) * IDF)
	# map -> output format: ((word, category, doc_id), (word_id, TFIDF))
	# map -> output format: (doc_id, category), [word_id, TFIDF])
	# reduceByKey -> output format: (doc_id, category), list_of([word_id, TFIDF]))
	# map -> output format: (category, sorted_list with word_index as key and tfidf metric value as value)
	# map -> output format: (category, SparseVector(k,key:word_id - value:TFIDF))
	tfidf = keep.flatMap(lambda x: [((y, x[0][0], x[1], len(x[0][1])), 1) for y in x[0][1]]).\
			reduceByKey(lambda x, y: x+y).\
			map(lambda x: ((x[0][0], x[0][1], x[0][2]), (x[1]/x[0][3]) * [idf_v[1] for idf_v in shared_idf.value if idf_v[0] == x[0][0]][0])).\
			map(lambda x: (x[0], (shared_vocab.value.index(x[0][0]), x[1]))).\
			map(lambda x: ((x[0][2], x[0][1]), [x[1]])).\
			reduceByKey(lambda x, y: x+y).\
			map(lambda x: (x[0][1], sorted(x[1], key=lambda y: y[0]))).\
			map(lambda x: (x[0], SparseVector(k, [y[0] for y in x[1]], [y[1] for y in x[1]])))

	# Release Memory
	#keep.unpersist()

	# Print Requested Outputs
	print('\n\nRequested Outputs\n')
	for i in tfidf.take(5):
		print(i)
	print('\n')

	#keep.unpersist()
	# Create DataFrame
	df = tfidf.toDF(['category', 'features'])

	stringIndexer = StringIndexer(inputCol='category', outputCol='label')
	stringIndexer.setHandleInvalid('skip')
	stringIndexerModel = stringIndexer.fit(df)
	df = stringIndexerModel.transform(df)

	# Grab unique labels
	uniq = df.select('label').distinct().collect()

	# Split Ratio for each Label
	fractions = {i: 0.8 for i in range(len(uniq)+1)}

	# Split to train-test
	train_set = df.sampleBy('label', fractions=fractions, seed=seed).cache()
	test_set = df.subtract(train_set)

	# Get number of documents for each set
	print('\n\nSize of train set: ', train_set.count(), '\n\n')
	print('\n\nSize of test set: ', test_set.count(), '\n\n')

	# Samples per Category for each set
	train_set.groupBy('category').count().show()
	test_set.groupBy('category').count().show()

	# input layer:k size, output layer:unique_cat size
	layers = [k, 200, len(uniq)]

	# Trainer
	trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=64, seed=seed)

	start_time = time.time()

	# Train the model
	model = trainer.fit(train_set)
	print('\n\n--- Time Elapsed for Training: {:0.2f} seconds ---\n\n'.format(time.time() - start_time))

	# compute accuracy on the test set
	result = model.transform(test_set)
	predictionAndLabels = result.select('prediction', 'label')
	evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
	print('\nTest set accuracy = {:0.2f} %\n'.format(evaluator.evaluate(predictionAndLabels) * 100))

	# Stop the session
	spark.stop()
