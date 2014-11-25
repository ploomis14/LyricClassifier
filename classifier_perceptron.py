"""Song Lyric Classifier - Perceptron Edition"""
"""Program is left uncommented and will not run unless you have already generated the test files in the other one. It's only here
as documentation of our attempted perceptron version of the classifier."""

from lxml import html
from collections import defaultdict
import nltk
from nltk.corpus import cmudict
import requests
import argparse
import string
import random
import os.path
import re
import math
import sys
import curses 
from curses.ascii import isdigit 
import numpy

UNK_PROB = .000000000001
START_TAG = "<s>"
END_TAG = "</s>"
CORPUS_SIZE = 6

reload(sys)
sys.setdefaultencoding("utf-8")

def create_train_data(genres):
    """
    Make training files for each genre from the first two pages of song data
    """
    compile_corpus_for_genre_perceptron(genres,"train.txt",10)


def compile_corpus_for_genre_perceptron(genres,filename,npages):
    """
    Scrapes website for song lyrics and compiles corpora with lyrics from each genre of music.
    Compile a corpus of lyrics for a certain genre of music
    """
    f = open(filename,'w')
    for genre in genres:
        for i in range(npages):
            page = requests.get('http://genius.com/tags/'+genre+'/all?page='+str(i))
            tree = html.fromstring(page.text)
            songs = tree.xpath('//*[@class=" song_link"]/@href')
            for song in songs:
                f.write(genre + " ")
                lyric_page = requests.get(song)
                song_tree = html.fromstring(lyric_page.text)
                verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
                for verse in verses:
                    if verse.strip("[]") == verse:  #skip annotations like "[Chorus 1]" and "[Sample: <song>]"
                        f.write(verse.strip("\n") + " ")
                f.write('\n')
    f.close()
    
def create_test_data(genre):
    """
    Make testing files for each genre from one page of songs
    """
    i = 0
    page = requests.get('http://genius.com/tags/'+genre+'/all?page=3')
    tree = html.fromstring(page.text)
    songs = tree.xpath('//*[@class=" song_link"]/@href')
    for song in songs:
        f = open(genre + "test%s.txt" % str(i), 'w')
        lyric_page = requests.get(song)
        song_tree = html.fromstring(lyric_page.text)
        verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
        for verse in verses:
            if verse.strip("[]") == verse:  #skip annotations like "[Chorus 1]" and "[Sample: <song>]"
                f.write(verse+'\n')
        i+=1
        f.close()
        
        
class LyricClassifier:
    def __init__(self):
        """
        Constructs models from training data, gets testing data by scraping songs on the website and
        outputing each to a file
        """
        self.genre_list = ['rock', 'rap']
        create_train_data(self.genre_list)

        rock = create_ngram_model('rocktrain.txt')
        rap = create_ngram_model('raptrain.txt')

        
        
        self.genre_model_list = [rock, rap]

    
class Perceptron:
    def __init__(self, numfeats):
        self.numfeats = numfeats
        self.w = numpy.zeros((numfeats+1,))   #+1 including intercept
        self.w[0] = 1 #list of zeros
        
    def train(self, traindata, trainlabels, max_epochs):

        #TODO: fill in
        epochs = 0
        mistakes = 1
        alpha = 1
        average = numpy.zeros((numfeats+1,))
        
        #loop until mistakes are eliminated or max epochs are hit
        while epochs < max_epochs and mistakes != 0:
            mistakes = 0
            epochs += 1
            
            
            #perceptron learning algorithm
            #to implement with random searching, use i in range random_iter(range...)
            for i in range(len(traindata)):
                temp = numpy.append([1], traindata[i])
                 
                if numpy.vdot(self.w, temp) > 0 and trainlabels[i] < 0:
                    for j in range(len(self.w)):
                        self.w[j] -= alpha * temp[j]
                    mistakes += 1
                     
                if numpy.vdot(self.w, temp) < 0 and trainlabels[i] > 0:
                    for j in range(len(self.w)):
                        self.w[j] += alpha * temp[j]
                    mistakes += 1
                    
            #keep a running average of perceptrons      
            for k in range(len(average)):
                average[k] += self.w[k]
                
        #once the loop is finished, update the main perceptron with the average one         
        for k in range(len(average)):
            self.w[k] = average[k]/epochs
        
#         for k in range(len(self.w)-1):
#             print self.w[k]
        return mistakes
    

    def test(self, testdata, testlabels):

        #TODO: fill in
        mistakes = 0
          
        for i in range(len(testdata)):
            if (numpy.vdot(self.w, numpy.append([1], testdata[i])) > 0 and testlabels[i] < 0) or (numpy.vdot(self.w, numpy.append([1], testdata[i])) < 0 and testlabels[i] > 0):
                mistakes += 1
                
        return mistakes

    def identify_song(self, filename):
        data = rawdata_to_vectors(filename, ndims = None)[0][0]
        
        while len(data) < len(self.w) - 1:
            data = numpy.append(data,[0])
            
#         print data
#         print len(data), len(self.w)
        if numpy.vdot(self.w, numpy.append([1], data)) > 0:
            print "Classified " + filename + " as rock"
            return True
        else:
            print "Classified " + filename + " as rap"
            return True
        
def rawdata_to_vectors(filename, ndims):
    """reads raw data, maps to feature space, 
    returns a matrix of data points and labels"""
    
    spam = open(filename).readlines()
    
    labels = numpy.zeros((len(spam),), dtype = numpy.int)  #gender labels for each user
        
    contents = []
    for li, line in enumerate(spam):
        line = line.split(' ')
        contents.append(line[1:])  #tokenized text of tweets, postags of tweets
        genre = line[0]
        #print genre
        if genre =='rock':
            labels[li] = 1
        else:
            labels[li] = -1

    representations, numfeats = bagofwords(contents)   #TODO: change to call your feature extraction function
    print "Featurized data"

    #convert to a matrix representation
    points = numpy.zeros((len(representations), numfeats))
    for i, rep in enumerate(representations):
        for feat in rep:
            points[i, feat] = rep[feat]

        #normalize to unit length
        l2norm = numpy.linalg.norm(points[i, :])
        if l2norm>0:
            points[i, :]/=l2norm

    if ndims:
        points = dimensionality_reduce(points, ndims)
        
    print "Converted to matrix representation"

    return points, labels
        
def bagofwords(contents):
    """represents data in terms of word counts.
    returns representations of data points as a dictionary, and number of features"""
    feature_counts = defaultdict(int)  #total count of each feature, so we can ignore 1-count features
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    for words in contents:
        for word in words:
            feature_counts[word]+=1
            
    for i, content in enumerate(contents):
        words = content
        representations.append(defaultdict(float))
        for word in words:
            if word in ['<s>', '</s>'] or feature_counts[word]==1:
                continue
            if word in features:
                feat_index = features[word]
            else:
                cur_index+=1
                features[word] = cur_index
                feat_index = cur_index
            representations[i][feat_index]+=1

    return representations, cur_index+1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics')
    args = parser.parse_args()
    
    if args.classify:         
        classifier = LyricClassifier()
        
        for genre in classifier.genre_list:
            create_test_data(genre)
            
        points, labels = rawdata_to_vectors('train.txt', ndims=None)
        print points, labels
        
        ttsplit = int(numpy.size(labels)/10)  #split into train and test 90-10
        traindata, testdata = numpy.split(points, [ttsplit*9])
        trainlabels, testlabels = numpy.split(labels, [ttsplit*9])
        
        numfeats = numpy.size(traindata, axis=1)
        classifier = Perceptron(numfeats)
        
        print "Training..."
        trainmistakes = classifier.train(traindata, trainlabels, max_epochs = 28)
        print "Finished training, with", trainmistakes/numpy.size(trainlabels), "% error rate"
        
        testmistakes = classifier.test(testdata, testlabels)
        print testmistakes/numpy.size(testlabels), "% error rate on test data"
        
        for item in ['rap', 'rock']:
            for n in range(20):
                classifier.identify_song(item.lower() + 'test%s.txt' % n)
