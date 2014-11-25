"""Song Lyric Generator/Classifier
    """
from lxml import html
from collections import defaultdict

import requests
import argparse

import os.path

import math
import sys


UNK_PROB = .000001
START_TAG = "<s>"
END_TAG = "</s>"
CORPUS_SIZE = 6


reload(sys)
sys.setdefaultencoding("utf-8")

def compile_corpus_for_genre(genre,filename,npages):
    """
    Scrapes website for song lyrics and compiles corpora with lyrics from each genre of music.
    Compile a corpus of lyrics for a certain genre of music
    """
    f = open(filename,'w')
    for i in range(npages):
        page = requests.get('http://genius.com/tags/'+genre+'/all?page='+str(i))
        tree = html.fromstring(page.text)
        songs = tree.xpath('//*[@class=" song_link"]/@href')
        for song in songs:
            lyric_page = requests.get(song)
            song_tree = html.fromstring(lyric_page.text)
            verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
            for verse in verses:
                if verse.strip("[]") == verse:  #skip annotations like "[Chorus 1]" and "[Sample: <song>]"
                    f.write(verse+'\n')
    f.close()

def create_ngram_model(filename):
    """
    Accumulate trigram, bigram, and unigram counts using a corpus of lyrics from a certain genre of music
    Returns a dictionary containing the ngram counts collected from the corpus file
    """
    countsdict = defaultdict(float)
    nextword = defaultdict(set)
    unigrams = 0
    bigrams = 0
    trigrams = 0

    for line in open(filename):
        line = line.lower()
        words = line.split()
        for i in range(len(words)):
            # unigrams
            unigrams += 1
            words[i] = words[i].strip('?.,/()!')
            countsdict[words[i].strip()]+=1.0

            # bigrams
            bigrams += 1
            if i == 0:
                countsdict[START_TAG+" "+words[i].strip()]+=1.0
                countsdict[START_TAG]+=1.0
                nextword[START_TAG].update([words[i].strip()])
            elif i >= 1:
                countsdict[words[i-1].strip()+" "+words[i].strip()]+=1.0
                nextword[words[i-1].strip()].update([words[i].strip()])    
                if i == len(words)-1:
                    countsdict[words[i].strip()+" "+END_TAG]+=1.0
                    # nextword[words[i].strip()].update([END_TAG]) 
                    countsdict[END_TAG]+=1.0
                
            # trigrams
            trigrams += 1
            if i >= 2:
                countsdict[words[i-2].strip()+" "+words[i-1].strip()+" "+words[i].strip()]+=1.0

                if i == len(words)-1:
                    countsdict[words[i-1].strip()+" "+words[i].strip()+" "+END_TAG]+=1.0
        
    #normalize the counts
    for key in countsdict:
        if len(key.split())==1:
            countsdict[key] /= unigrams
        if len(key.split())==2:
            countsdict[key] /= bigrams
        if len(key.split())==3:
            countsdict[key] /= trigrams

    return countsdict,nextword

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
    for j in range(4):
        page = requests.get('http://genius.com/tags/'+genre+'/all?page=%s' % (CORPUS_SIZE + j))
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
        self.genre_list = ['rock', 'rap', 'pop']
        create_train_data(self.genre_list)

        rock = create_ngram_model('rocktrain.txt')[0]
        rap = create_ngram_model('raptrain.txt')[0]
        pop = create_ngram_model('poptrain.txt')[0]

        for genre in self.genre_list:
            create_test_data(genre)
        
        self.genre_model_list = [rock, rap, pop]

    def generate_key(self, seq): #turns a subset of a list into a key for the n-grams dictionary
        key = ""
        for item in seq:
            key += item + " "
        
        return key

    def classify(self, filename):
        maxprob = float('-inf')
        best_fit = {}
        
        #find p(lyrics) for each model
        for model in self.genre_model_list:
            
            totalprob = 0.0
            backpointers = ['<s>','<s>','<s>']
            
            #loop through each line and modify it so it's legible to the n-gram model
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
                        prob += l*model[key.strip()]
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
        
        #if something went horribly wrong, randomly label the song - this line shouldn't get used
        if best_fit == {}:
            return 0
        
        #return the guessed genre
        index = self.genre_model_list.index(best_fit)
        return self.genre_list[index]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics. If supplied file does not exist, program will classify test data')
    args = parser.parse_args()
    
    
    if args.classify: #classify the file supplied, OR run on a bunch of test data and report results
        
        genre_list = ['rap','rock', 'pop']
        classifier = LyricClassifier()
        if os.path.exists(args.classify) and os.path.getsize(args.classify) != 0:
            print "Song classified as " + classifier.classify(args.classify)
            
        else:
            #generate test data if non-existent
            for genre in genre_list:
                if not os.path.exists(genre+"test0.txt"):
                    create_test_data(genre)
            
            #loop over test data, classify, count errors        
            errors = 0.0
            total = 0.0
            for item in genre_list:
                for n in range(80):
                    guess = classifier.classify(item.lower() + 'test%s.txt' % n)
                    print "guessed "+guess+" for "+item+" song"
                    if item != guess:
                        errors += 1
                    total += 1
    
            print "Accuracy: " + str(100*((total-errors)/total))
            


