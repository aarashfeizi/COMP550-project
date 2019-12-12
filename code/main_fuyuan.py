import random
import pickle
import numpy as np
import sklearn
import nltk
from sklearn.metrics import classification_report,accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def read_rawdata(dataset_path):
	print("-------------------------")
	print("dataset path:\t"+dataset_path)
	with open(dataset_path, 'rb') as data:
		dataset = pickle.load(data)
	train = dataset['train']
	test = dataset['test']
	return train, test


def preprocessing_sentence(text, methods):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    if 'lemmatization' in methods:
        new_tokens = set()
        for token in tokens:
            new_token = lemmatizer.lemmatize(token)
            new_tokens.add(new_token)
        tokens = list(new_tokens)
    if 'stemming' in methods:
        new_tokens = set()
        for token in tokens:
            new_token = stemmer.stem(token)
            new_tokens.add(new_token)
        tokens = list(new_tokens)
    if 'stopwords' in methods:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
    newtext = " ".join(str(item) for item in tokens)
    return newtext

def preprocessing_papers(dataset, methods=['lemmatization', 'stemming', 'stopwords']):
    # preprocessing each sentence
    # lemmatization, stemming and removing stopwords
    # infrequent words are removed in feature extraction process, not here
    item = dataset[0]
    string = ''

    new_dataset = dataset

    return new_dataset, string


def get_numerical_feature(train, test, low_frequent_threshold=1):
    # Extract feature here
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
#    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    train_feature = vectorizer.fit_transform(train[0])
    test_feature = vectorizer.transform(test[0])
    
    word_frequency = np.ravel(train_feature.sum(axis=0))
    length1 = word_frequency.shape[0]

    print("Total tokens: " + str(length1))
    print("Average word frequency: " + str(word_frequency.sum()/length1))
    print("Frequency threshold: " + str(low_frequent_threshold))
    
    is_frequent = np.greater_equal(word_frequency, [low_frequent_threshold]*length1)

    frequent_indices = np.where(is_frequent == True)[0]
    print("Tokens left: "+ str(len(frequent_indices)))
    
    # Because train feature is CSR format, np.delete() not working well
#    new_train_feature = np.delete(train_feature, not_frequent_indices, axis=1)
    new_train_feature = train_feature[:,frequent_indices]
    new_test_feature = test_feature[:,frequent_indices]

    return new_train_feature, new_test_feature


def feature_extraction(train, test, total_string):
    my_threshold = 1
    all_tokens = nltk.word_tokenize(total_string)
    train_feature, test_feature = get_numerical_feature(train, test, my_threshold)
    return train_feature, test_feature

def svm(train, test, train_feature, test_feature):
    svm = sklearn.svm.LinearSVC(max_iter=100000)
    svm.fit(train_feature, train[1])
    prediction = svm.predict(test_feature)    
    
    print("----- Support Vector Machine -----")
    print("accuracy: {0}".format(accuracy_score(test[1], prediction)))

def logisticregression(train, test, train_feature, test_feature):
    lr = LogisticRegression(max_iter=100000, solver='liblinear')
    lr.fit(train_feature, train[1])
    prediction = lr.predict(test_feature)
    
    print("----- Logistic regression -----")
    print("accuracy: {0}".format(accuracy_score(test[1], prediction)))

def decision_tree(train, test, train_feature, test_feature):
    tree = DecisionTreeClassifier()
    tree.fit(train_feature, train[1])
    prediction = tree.predict(test_feature)

    print("----- Decision Tree -----")
    print("accuracy: {0}".format(accuracy_score(test[1], prediction)))

def naivebayes(train, test, train_feature, test_feature):
    nb = MultinomialNB()
    nb.fit(train_feature, train[1])
    prediction = nb.predict(test_feature)

    print("----- Naive Bayes -----")
    print("accuracy: {0}".format(accuracy_score(test[1], prediction)))


def xgboost(train, test, train_feature, test_feature):
	dtrain = xgb.DMatrix(train)
	dtest = xgb.DMatrix(test)


def train_models(train, test, train_feature, test_feature):
    svm(train, test, train_feature, test_feature)
    logisticregression(train, test, train_feature, test_feature)
    decision_tree(train, test, train_feature, test_feature)
    naivebayes(train, test, train_feature, test_feature)





def main(dataset_path):
	train, test = read_rawdata(dataset_path)
	my_methods = []
	train_new, total_string = preprocessing_papers(train, methods=my_methods)
	test_new, _ = preprocessing_papers(test, methods=my_methods)
	train_feature, test_feature = feature_extraction(train_new, test_new, total_string)
	train_models(train_new, test_new, train_feature, test_feature)

if __name__ == "__main__":
	main('../dataset/abstract.pickle')
	main('../dataset/abstract+introduction.pickle')
	main('../dataset/all.pickle')

