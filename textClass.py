#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import random


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from scipy import stats


import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import warnings


#Usefule refence:  https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/


#Supress all warnings
warnings.filterwarnings("ignore")

baseDir = "./dataIHIET/"

weFN = 'wiki-news-300d-1M.vec'
#NUMBER OF EXPERIMENTS - TASK BOUNDARY vs NON-TASK BOUNDARY ; HUMAN vs COMPUTER ; HUMAN INTERRUPTIONS vs NON INTERRUPTIONS ;
#TASK BOUNDARY vs HUMAN INTERRUPTIONS
#NUMBER OF DATASETS - ALL ; TEAM ; SPEAKER ; INTERRUPTER
#NUMBER OF FEATURES: 4 - WORD COUNT ; WORD TFIDF ; NGRAM_TFIDF ; CHAR NGRAM TFIDF
#NUMBER OF MODELS: 5 - NAIVE BAYES ; LOGISTIC REGRESSION ; SUPPORT VECTOR MACHINE ; RANDOM FOREST ; GRADIENT BOOST

numFolds = 20

OUT = pd.DataFrame()

DATA = pd.DataFrame(columns={'EXP','DATA_TYPE','DATA_LABEL','N','WC_RF','WORD_TFIFD_RF','NGRAM_TFIDF_RF','CHAR_NGRAM_TFIDF_RF'})


for subdir, dirs, files in os.walk(baseDir):
    for file in files:
        fn = subdir + "/" + file #This may not be right
        if ".DS_Store" in fn:
            continue
        print('Processing features for ' + fn)
        ss = subdir.split('/')
        ss1 = file.split('.')
        title = ss[2] + '_' + ss[3] + '_' + ss1[0]

        #Get DATA
        dataDF= pd.read_csv(fn)

        x = dataDF['UTTERANCE']
        y = dataDF['LABEL']

        N = len(x)

        listEXP = []
        listDATA_TYPE = []
        listDATA_LABEL = []
        listN = []
        # Populate the Experiment and Dataset
        for i in range(numFolds):
            listEXP.append(ss[2])
            listDATA_TYPE.append(ss[3])
            listDATA_LABEL.append(ss1[0])
            listN.append(N)

        DATA['EXP'] = listEXP
        DATA['DATA_TYPE'] = listDATA_TYPE
        DATA['DATA_LABEL'] = listDATA_LABEL
        DATA['N'] = listN

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(y)

        # create a count vectorizer object

        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(dataDF['UTTERANCE'])

        # transform the training and calidation data using count vectorizer object
        x_count = count_vect.transform(x)

        # word leverl tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(dataDF['UTTERANCE'])
        x_tfidf = tfidf_vect.transform(x)

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),max_features=5000)

        tfidf_vect_ngram.fit(dataDF['UTTERANCE'])
        x_tfidf_ngram = tfidf_vect_ngram.transform(x)

        # characters level tf-idf
        tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                max_features=5000)
        tfidf_vect_ngram_char.fit(dataDF['UTTERANCE'])
        x_tfidf_ngram_chars = tfidf_vect_ngram_char.transform(x)


        def train_model(classifier, feature_vector_train, label):
            score = cross_val_score(classifier, feature_vector_train, label, cv=numFolds, scoring='f1')
            return score
            ''''
            loo = LeaveOneOut()
            loo.get_n_splits(feature_vector_train,label)
            tp, tn, fp, fn = 0, 0, 0, 0
            ytestArray = []
            ypredArray = []

            for train_index, test_index in loo.split(feature_vector_train,label):
                x_train, x_test = feature_vector_train[train_index], feature_vector_train[test_index]
                y_train, y_test = label[train_index], label[test_index]

                kclass = classifier.fit(x_train,y_train)
                y_pred = kclass.predict(x_test)
                ytestArray.append(y_test)
                ypredArray.append(y_pred)
            
            tn, fp, fn, tp = confusion_matrix(ytestArray, ypredArray).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = stats.hmean([precision, recall])
            '''''
            #acc = cross_val_score(classifier, feature_vector_train, label, cv=20,scoring='accuracy')
            #prec = cross_val_score(classifier, feature_vector_train, label, cv=20, scoring='precision')
            #recall = cross_val_score(classifier, feature_vector_train, label, cv=20, scoring='recall')
            #y_pred = cross_val_predict(classifier, feature_vector_train, label, cv=10)
            #conf_mat = confusion_matrix(label,y_pred)

            #return score


        def replaceitem(x):
            for i in range(len(x)):
                if x[i] < 0.5:
                    x[i] = 1.0 - x[i]
                else:
                    x[i] = x[i]
            return x


        print('Generating models for ' + fn)

        #try:
        # RF on Count Vectors
        scores = train_model(ensemble.RandomForestClassifier(), x_count, y)
        #scores = replaceitem(scores)
        DATA['WC_RF'] = scores

        # RF on Word Level TF IDF Vectors
        scores = train_model(ensemble.RandomForestClassifier(), x_tfidf, y)
        #scores = replaceitem(scores)
        DATA['WORD_TFIFD_RF'] = scores

        # RF Ngram Level TF IDF Vectors
        scores = train_model(ensemble.RandomForestClassifier(), x_tfidf_ngram, y)
        #scores = replaceitem(scores)
        DATA['NGRAM_TFIDF_RF'] = scores

        # RF on Ngram Level TF IDF Vectors
        scores = train_model(ensemble.RandomForestClassifier(), x_tfidf_ngram_chars, y)
        #scores = replaceitem(scores)
        DATA['CHAR_NGRAM_TFIDF_RF'] = scores

        d = [OUT, DATA]

        OUT = pd.concat(d)

        OUT.to_csv('dataPointRF_20_IHIET.csv')

       # except:pass





