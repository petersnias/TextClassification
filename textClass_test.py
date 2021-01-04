#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import random


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


#Usefule refence:  https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

baseDir = "./data/"

weFN = 'wiki-news-300d-1M.vec'



for subdir, dirs, files in os.walk(baseDir):
    for file in files:
        fn = subdir + "/" + file #This may not be right
        if ".DS_Store" in fn:
            continue
        print(fn)
        ss = subdir.split('/')
        ss1 = file.split('.')
        title = ss[2] + '_' + ss[3] + '_' + ss1[0]

        #Get DATA
        dataDF= pd.read_csv(fn)

        # split data into training and testing data
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(dataDF['UTTERANCE'], dataDF['LABEL'])

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)

        # create a count vectorizer object

        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(dataDF['UTTERANCE'])

        # transform the training and calidation data using count vectorizer object
        xtrain_count = count_vect.transform(train_x)
        xvalid_count = count_vect.transform(valid_x)

        # word leverl tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(dataDF['UTTERANCE'])
        xtrain_tfidf = tfidf_vect.transform(train_x)
        xvalid_tfidf = tfidf_vect.transform(valid_x)

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                           max_features=5000)

        tfidf_vect_ngram.fit(dataDF['UTTERANCE'])
        xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
        xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

        # characters level tf-idf
        tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                max_features=5000)
        tfidf_vect_ngram_char.fit(dataDF['UTTERANCE'])
        xtrain_tfidf_ngram_chars = tfidf_vect_ngram_char.transform(train_x)
        xvalid_tfidf_ngram_chars = tfidf_vect_ngram_char.transform(valid_x)


        #load the pre-trained word embeddings vectors (need to download this)

        embeddings_index = {}
        for i, line in enumerate(open(weFN)):
            values = line.split()
            embeddings_index[values[0]] = numpy.asanyarray(values[1:],dtype='float32')


        #create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(dataDF['UTTERANCE'])
        word_index = token.word_index

        # convert text to sequence of tokens and pad them to ensure equal length vectors
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
        valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

        # create token-embedding mapping
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        dataDF['char_count'] = dataDF['UTTERANCE'].apply(len)
        dataDF['word_count'] = dataDF['UTTERANCE'].apply(lambda x: len(x.split()))
        dataDF['word_density'] = dataDF['char_count'] / (dataDF['word_count']+1)
        dataDF['punctuation_count'] = dataDF['UTTERANCE'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
        dataDF['title_word_count'] = dataDF['UTTERANCE'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        dataDF['upper_case_word_count'] = dataDF['UTTERANCE'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

        pos_family = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }

        # function to check and get the part of speech tag count of a words in a given sentence
        def check_pos_tag(x, flag):
            cnt = 0
            try:
                wiki = textblob.TextBlob(x)
                for tup in wiki.tags:
                    ppo = list(tup)[1]
                    if ppo in pos_family[flag]:
                        cnt += 1
            except:
                pass
            return cnt

        dataDF['noun_count'] = dataDF['UTTERANCE'].apply(lambda x: check_pos_tag(x, 'noun'))
        dataDF['verb_count'] = dataDF['UTTERANCE'].apply(lambda x: check_pos_tag(x, 'verb'))
        dataDF['adj_count'] = dataDF['UTTERANCE'].apply(lambda x: check_pos_tag(x, 'adj'))
        dataDF['adv_count'] = dataDF['UTTERANCE'].apply(lambda x: check_pos_tag(x, 'adv'))
        dataDF['pron_count'] = dataDF['UTTERANCE'].apply(lambda x: check_pos_tag(x, 'pron'))

        # train a LDA Model
        lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
        X_topics = lda_model.fit_transform(xtrain_count)
        topic_word = lda_model.components_
        vocab = count_vect.get_feature_names()

        # view the topic models
        n_top_words = 10
        topic_summaries = []
        for i, topic_dist in enumerate(topic_word):
            topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
            topic_summaries.append(' '.join(topic_words))


        def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
            # fit the training dataset on the classifier
            classifier.fit(feature_vector_train, label)

            # predict the labels on validation dataset
            predictions = classifier.predict(feature_vector_valid)

            if is_neural_net:
                predictions = predictions.argmax(axis=-1)

            return metrics.accuracy_score(predictions, valid_y)


        scores = cross_val_score(naive_bayes.MultinomialNB(), xtrain_count, train_y, cv=10)
        print(scores)
        # Naive Bayes on Count Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
        print("NB, Count Vectors: ", accuracy)

        # Naive Bayes on Word Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
        print("NB, WordLevel TF-IDF: ", accuracy)

        # Naive Bayes on Ngram Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        print("NB, N-Gram Vectors: ", accuracy)

        # Naive Bayes on Character Level TF IDF Vectors
        accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
        print("NB, CharLevel Vectors: ", accuracy)

        # Linear Classifier on Count Vectors
        accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
        print("LR, Count Vectors: ", accuracy)

        # Linear Classifier on Word Level TF IDF Vectors
        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
        print("LR, WordLevel TF-IDF: ", accuracy)

        # Linear Classifier on Ngram Level TF IDF Vectors
        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        print("LR, N-Gram Vectors: ", accuracy)

        # Linear Classifier on Character Level TF IDF Vectors
        accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
        print("LR, CharLevel Vectors: ", accuracy)

        # SVM on Ngram Level TF IDF Vectors
        accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        print("SVM, N-Gram Vectors: ", accuracy)

        # RF on Count Vectors
        accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
        print("RF, Count Vectors: ", accuracy)

        # RF on Word Level TF IDF Vectors
        accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
        print("RF, WordLevel TF-IDF: ", accuracy)

        # Extereme Gradient Boosting on Count Vectors
        accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
        print("Xgb, Count Vectors: ", accuracy)

        # Extereme Gradient Boosting on Word Level TF IDF Vectors
        accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
        print("Xgb, WordLevel TF-IDF: ", accuracy)

        # Extereme Gradient Boosting on Character Level TF IDF Vectors
        accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
        print("Xgb, CharLevel Vectors: ", accuracy)

