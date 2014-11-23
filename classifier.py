"""Song Lyric Classifier"""

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

UNK_PROB = .000000000001

reload(sys)
sys.setdefaultencoding("utf-8")

def create_train_data(genres):
    """
    Make training files for each genre from the first two pages of song data
    """
    for genre in genres:
        compile_corpus_for_genre(genre,genre+"train.txt",2)

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

        for genre in self.genre_list:
            create_test_data(genre)
        
        self.genre_model_list = [rock, rap]

    def generate_key(self, seq):
        key = ""
        for item in seq:
            key += item + " "
        
        return key

    def classify(self, filename):
        maxprob = float('-inf')
        best_fit = {}
        #find p(lyrics) for each model
        for model in self.genre_model_list:
            #print model
            totalprob = 0.0
            backpointers = ['<s>','<s>','<s>']
            for line in open(filename):
                line = line.lower().split()
                line.append(END_TAG)
                
                #for each word in the file, push it into the backpointers list - everything moves one to the left
                for word in line:
                    word = word.strip('()')
                    prob = 0.0
                    for i in range(len(backpointers) - 1):
                        backpointers[i] = backpointers[i+1]
                    
                    backpointers[len(backpointers)-1] = word
                    
                    
                    #use interpolation to compute probabilities of n-grams found in the backpointers list
                    j = 0
                    l = .9
                    while j < len(backpointers):
                        key = self.generate_key(backpointers[j:len(backpointers)])
                        #print key
                        prob += l*model[key.strip()]
                        #print key
                        #print model[key.strip()]
                        j+=1
                        l*=.1
                    
                    #if no n-gram matches, we're looking at an unkown word
                    if prob == 0:
                        prob += UNK_PROB
                    
                    totalprob += math.log(prob)
                
                #end of a line gets a start of sentence tag
                for i in range(len(backpointers) - 1):
                    backpointers[i] = backpointers[i+1]
                backpointers[len(backpointers)-1] = '<s>'
            
            #update the best-performing model
            #print totalprob
            if totalprob > maxprob:
                maxprob = totalprob
                best_fit = model
        
        if best_fit == {}:
            return 0
        index = self.genre_model_list.index(best_fit)
        return self.genre_list[index]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics')
    args = parser.parse_args()
    
    if args.classify:
        genre_list = ['rap','rock']
        classifier = LyricClassifier()
        errors = 0.0
        total = 0.0
        for item in genre_list:
            for n in range(20):
                guess = classifier.classify(item.lower() + 'test%s.txt' % n)
                print "guessed "+guess+" for "+item+" song"
                if item != guess:
                    errors += 1
                total += 1

        print "Accuracy: " + str(100*((total-errors)/total))
