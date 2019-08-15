from flask import render_template, request, redirect, url_for, session
from app import app

import collections
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize

import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stopwords = factory.get_stop_words()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import pandas as pd

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global result
    global nclusters
    global clusters
    teks = request.form['teks']
    nclusters_ = int(request.form['kluster'])
    result_ = sent_tokenize(teks)
    x = len(result_)        
    steming = []
    stopy1 = []
    stopy2 = []
    stopy3 = []
    for i in range(x):
        steming.append(i)
        steming[i] = stemmer.stem(result_[i])
        stopy1.append(i)
        stopy1[i] = stopword.remove(steming[i])
        stopy2.append(i)
        stopy2[i] = stopword.remove(stopy1[i])
        stopy3.append(i)
        stopy3[i] = stopword.remove(stopy2[i])
    vectorizer = TfidfVectorizer(smooth_idf=False, norm=None)
    tfidf1 = vectorizer.fit_transform(stopy2)
    tabletemp = pd.DataFrame(columns = ["id","teks","score"])
    for temp1 in range(x):
        ids = temp1
        count = 0
        tekss = result_[temp1]
        for temp2 in tfidf1.A[temp1]:
            count += temp2
        tabletemp = tabletemp.append({
            'id' : ids, 
            'teks' : tekss,
            'score' : count
        }, ignore_index=True)
    avescore = 0
    totaldata = tabletemp.id.count()
    for score in tabletemp.score:
        avescore += score
    ave = avescore/totaldata
    tabletemp1 = tabletemp.drop(tabletemp[tabletemp.score < ave].index)
    n_clusters = nclusters_
    if tabletemp1.id.count() < n_clusters:
        tabletemp.nlargest(n_clusters,'score')
        afscore = []
        for m in tabletemp.id:
            afscore.append(result[m])
    else:
        afscore = []
        for m in tabletemp1.id:
            afscore.append(result_[m])
    tfidf = vectorizer.fit_transform(afscore)
    kmeans = KMeans(n_clusters, random_state=0)
    kmeans.fit(tfidf)
    clusters_ = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters_[label].append(i)
    result = result_
    nclusters = nclusters_
    clusters = clusters_
    close = kmeans.fit(tfidf)
    closest, _ = pairwise_distances_argmin_min(close.cluster_centers_, tfidf)
    closest.sort()
    return render_template('result.html', 
        result = result,
        nclusters = nclusters,
        clusters = clusters,
        teks = teks,
        closest = closest)

@app.route('/result')
def result():
    global nclusters
    global result
    global clusters
    return render_template('detail.html', 
        result = result, 
        nclusters = nclusters, 
        clusters = clusters)