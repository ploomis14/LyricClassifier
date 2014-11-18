"""Song Lyric Generator/Classifier
    """
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
START_TAG = "<s>"
END_TAG = "</s>"
CORPUS_SIZE = 4
LINES_PER_VERSE = 4
MAX_SYLLABLES = 8
MIN_SYLLABLES = 5
VERSES_PER_SONG = 3

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

class LyricGenerator:
    def __init__(self, genre):
        """
        Collects corpus of lyrics, constructs ngram model for genre
        """
        filename = genre+'.txt'
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            compile_corpus_for_genre(genre,filename,CORPUS_SIZE)
        (self.model,self.nextword) = create_ngram_model(filename)
    
    def approx_nsyl(self,word):
        """Credit - Jason Sundram, http://runningwithdata.com/post/3576752158/w
        Return the max syllable count in the case of multiple pronunciations"""
        d = cmudict.dict()
        if word not in d.keys():
            return 0
        return max([len([y for y in x if y[-1].isdigit()]) for x in d[word.lower()]])

    def generate_line(self):
        """Use an ngram model to generate a single line of lyrics for a certain genre of music"""
        # Choose a random word to start the lyric. Choose from the set of words that follow a start tag.
        start = random.choice([ngram for ngram in self.model.keys() if START_TAG in ngram.split()]).split()[1]
        sequence = start
        i = 1

        # Continuity among the syllable length of lines
        # There should be increased probability of ending a line when the maximum syllable length is exceeded
        end_of_line_prob = 0.0
        syllables = 1

        while(1):
            # Choose the next word in the generated sequence based on bigram probabilities
            nextword = ""
            bestProb = prob = 0.0
            for token in self.nextword[sequence.split()[-1]]:
                if sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token in self.model.keys():
                    prob = self.model[sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token]/self.model[sequence.split()[i-2]+" "+sequence.split()[i-1]]
                else:
                    prob = self.model[sequence.split()[-1]+" "+token]/self.model[sequence.split()[-1]]
                
                if prob > bestProb:
                    bestProb = prob
                    nextword = token

            print sequence.split()
            end_of_line_prob = self.model[sequence.split()[-1]+" "+END_TAG]/self.model[sequence.split()[-1]]
            
            if syllables > MAX_SYLLABLES:
                end_of_line_prob += 0.6
            if syllables < MIN_SYLLABLES:
                end_of_line_prob -= 0.2

            # Exit the loop when the probability of ending the verse is greater than the probability of adding another word
            if end_of_line_prob > bestProb or nextword == "":
                break
            sequence = sequence+" "+nextword
            syllables += self.approx_nsyl(nextword)
            i += 1

        return sequence

    def rhyme(self,w, pos):
        """Given a word and its POS tag, return a rhyming word that has the same part of speech"""
        entries = nltk.corpus.cmudict.entries()
        syllables = [(word, syl) for word, syl in entries if word == w and pos == nltk.pos_tag([word])]
        rhymes = [word for (token, syllable) in syllables for word, pron in entries if pron[-2:] == syllable[-2:]]
        if len(rhymes) == 0:
            return w
        return rhymes[random.randint(0,len(rhymes))]

    def output_lyrics(self,filename):
        """Outputs verses to file (groups of four lines where the last word of two consecutive lines matches)"""
        output_file = open(filename,'w')
        previous_line = ""
        lyrics = ""

        for v in range(1,VERSES_PER_SONG+1):
            for i in range(1,LINES_PER_VERSE+1):
                current_line = self.generate_line()
                output = current_line+"\n"
                # Exchange the last word of the current line for a word the rhymes with the previous line
                if i%2 == 0:
                    prev_word = previous_line.split()[-1]
                    pos = nltk.pos_tag([prev_word])
                    rhyme_word = self.rhyme(prev_word,pos)
                    if len(rhyme_word) > 0:
                        output = current_line.rsplit(' ', 1)[0]+" "+rhyme_word+"\n"
                lyrics = lyrics+output
                print output
                previous_line = current_line
            lyrics = lyrics+"\n"
        output_file.write(lyrics)
        output_file.close()


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
    parser.add_argument('-generate', type = str, required = False, choices = ['rock','rap','pop'], help = 'generate genre lyrics (rock, rap, pop)')
    parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics')
    args = parser.parse_args()
    
    if args.generate:
        generator = LyricGenerator(args.generate)
        print "generating "+args.generate+" lyrics..."
        generator.output_lyrics('generate-'+args.generate+'.txt')
    
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
