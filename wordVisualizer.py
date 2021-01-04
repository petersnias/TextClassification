#Visualization code from
#https://www.kaggle.com/nicapotato/explore-the-spooky-n-grams-wordcloud-bayes


# Packages
import os
import numpy as np
import pandas as pd
import nltk
import random
import string as str

# Pre-Processing
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import *

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

# N- Grams
from nltk.util import ngrams
from collections import Counter

# Topic Modeling
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Word 2 Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# Models
import datetime
from nltk import naivebayes

import warnings
warnings.filterwarnings("ignore")

tokenizer = RegexpTokenizer(r'\w+')
#stop_words = set(stopwords.words('englishTangram'))

def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words if not w in stop_words] #3
    #words = [w for w in words]  # 3, don't get rid of STOPWORDS
    #words = [ps.stem(w) for w in words] #4
    return words

def wordfreqviz(text, x, title):
    word_dist = nltk.FreqDist(text)
    top_N = x
    rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')
    mpl.style.use('ggplot')
    rslt.plot.bar(rot=0)
    plt.savefig("./plots/" + title + "_WORDHIST.png")

def wordfreq(text, x):
    word_dist = nltk.FreqDist(text)
    top_N = x
    rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')
    print(rslt)

    # Function
def cloud(text, title):
    # Setting figure parameters
    mpl.rcParams['figure.figsize'] = (10.0, 10.0)  # (6.0,4.0)
    # mpl.rcParams['font.size']=12                #10
    mpl.rcParams['savefig.dpi'] = 100  # 72
    mpl.rcParams['figure.subplot.bottom'] = .1
    # Processing Text
    stopwords = set(STOPWORDS)  # Redundant
    wordcloud = WordCloud(width=1400, height=800,
                          background_color='black',
                          collocations = False,
                          stopwords=stopwords,
                          ).generate(" ".join(text))

    # Output Visualization
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=50, color='y')
    plt.savefig("./plots/" + title + "_WC.png")
    #plt.imshow(plt.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
    #fig.savefig("wordcloud.png", dpi=900)

## Helper Functions
def get_ngrams(text, n):
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]

def gramfreq(text,n,num):
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)
    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name
    return df.sort_values(["frequency"],ascending=[0])[:num]

def gram_table(sentences, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramfreq(preprocessing(sentences),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Occurence"]
        out = pd.concat([out, table], axis=1)
    return out


lemm = WordNetLemmatizer()


class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print("=" * 70)


def LDA(data):
    # Storing the entire training text in a list
    text = list(data.values)
    # Calling our overwritten Count vectorizer
    tf_vectorizer = LemmaCountVectorizer(max_df=0.95, min_df=2,
                                         stop_words='english',
                                         decode_error='ignore')
    tf = tf_vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation(n_topics=6, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)

    n_top_words = 10
    print("\nTopics in LDA model: ")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)


#Dat visualization
baseDir = "./data/"
topWords = 20 #Top words to be visulatized


for subdir, dirs, files in os.walk(baseDir):
    for file in files:
        fn = subdir + "/" + file
        if ".DS_Store" in fn:
            continue
        print(fn)
        ss = subdir.split('/')
        ss1 = file.split('.')
        #title = ss[2] + '_' + ss[3] + '_' + ss1[0]
        #print(title)
        #inData = pd.read_csv(fn)
        dataDF= pd.read_csv(fn)

        #x = dataDF['UTTERANCE']
        #y = dataDF.dataDF['LABEL'==0]

        sentencesPos = dataDF.loc[dataDF['LABEL']==1]
        SP = sentencesPos['UTTERANCE']
        title = ss[2] + '_' + ss[3] + '_' + ss1[0] + '_POS'
        words = preprocessing(SP)
        wordfreqviz(words, topWords, title)
        cloud(words, title)
        ngramTable = gram_table(SP, gram=[1, 2, 3], length=20)
        ngramTable.to_csv("./plots/" + title + "_NGRAM.csv")




        sentencesNeg = dataDF.loc[dataDF['LABEL']==0]
        SN = sentencesNeg['UTTERANCE']
        title = ss[2] + '_' + ss[3] + '_' + ss1[0] + '_NEG'
        words = preprocessing(SN)
        wordfreqviz(words, topWords, title)
        cloud(words, title)
        ngramTable = gram_table(SN, gram=[1, 2, 3], length=20)
        ngramTable.to_csv("./plots/" + title + "_NGRAM.csv")






#
# # Read Data
# neg = pd.read_csv(baseDir + dataFolder + "/" + subDataFolder + "/0.txt")
# pos = pd.read_csv(baseDir + dataFolder + "/" + subDataFolder + "/1.txt")
#
#
# negSent = neg.iloc[:,0] #Get sentence data
#
# #Process and visualize all the NEGATIVE WORDS
# words = preprocessing(negSent)
# wordfreqviz(words, topWords)
# wordfreq(words, topWords)
# cloud(words, 'test')
# a = gramfreq(words,2,25)
# print(a)
# b = gram_table(negSent, gram=[1,2,3,4], length=20)
# print(b)
#
# LDA(negSent)
#
#
# #Visualize and process all the positive words
# posSent = pos.iloc[:,0]





#ps = PorterStemmer()



#import sys
#from os import listdir
#from os.path import isfile, join

#mypath = './data/TB_NONTB/ALL'
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(onlyfiles)

#inputFile = "./data/"

#df = pd.read_csv("../input/train.csv", index_col="id")
#test = pd.read_csv("../input/test.csv", index_col="id")
