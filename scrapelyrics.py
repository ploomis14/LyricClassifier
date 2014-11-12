"""Song Lyric Generator
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

START_TAG = "<s>"
END_TAG = "</s>"
CORPUS_SIZE = 4
LINES_PER_VERSE = 6

#approximates the number of syllables in a word
def approx_nsyl(word):  #Credit to Danielle Sucher - http://www.daniellesucher.com/2012/04/nantucket-an-accidental-limerick-detector/
    digraphs = ["ai", "au", "ay", "ea", "ee", "ei", "ey", "oa", "oe", "oi", "oo", "ou", "oy", "ua", "ue", "ui"]
    # Ambiguous, currently split: ie, io
    # Ambiguous, currently kept together: ui
    digraphs = set(digraphs)
    count = 0
    array = re.split("[^aeiouy]+", word.lower())
    for i, v in enumerate(array):
        if len(v) > 1 and v not in digraphs:
            count += 1
        if v == '':
            del array[i]
    count += len(array)
    if re.search("(?&lang;=\w)(ion|ious|(?&lang;!t)ed|es|[^lr]e)(?![a-z']+)", word.lower()):
        count -= 1
    if re.search("'ve|n't", word.lower()):
        count += 1
    return count


def compile_corpus_for_genre(genre):
    """
        Scrapes website for song lyrics and compiles corpora with lyrics from each genre of music.
        Compile a corpus of lyrics for a certain genre of music
        """
    f = open(genre+'train.txt','w')
    for i in range(CORPUS_SIZE):
        page = requests.get('http://genius.com/tags/'+genre+'/all?page='+str(i))
        tree = html.fromstring(page.text)
        songs = tree.xpath('//*[@class=" song_link"]/@href')
        for song in songs:
            lyric_page = requests.get(song)
            song_tree = html.fromstring(lyric_page.text)
            verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
            for verse in verses:
                if verse.strip("[]") == verse:  #skip annotations like "[Chorus 1]" and "[Sample: <song>]"
                    f.write(verse.encode('utf-8')+'\n')
    
    
    f.close()
    create_test_data(genre, CORPUS_SIZE)

def create_test_data(genre, page):
    i = 0
    page = requests.get('http://genius.com/tags/'+genre+'/all?page='+str(page))
    tree = html.fromstring(page.text)
    songs = tree.xpath('//*[@class=" song_link"]/@href')
    for song in songs:
        f = open(genre + "test%s.txt" % str(i), 'w')
        lyric_page = requests.get(song)
        song_tree = html.fromstring(lyric_page.text)
        verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
        for verse in verses:
            if verse.strip("[]") == verse:  #skip annotations like "[Chorus 1]" and "[Sample: <song>]"
                f.write(verse.encode('utf-8')+'\n')
        i+=1
        f.close()

def generate_key(seq):
    key = ""
    for item in seq:
        key += item + " "
    
    return key

def classify(models, filename):
    maxprob = 0.0
    best_fit = {}
    #find p(lyrics) for each model
    for model in models:
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
                    key = generate_key(backpointers[j:len(backpointers)])
                    #print key
                    prob += l*model[key.strip()]
                    #print key
                    #print model[key.strip()]
                    j+=1
                    l*=.1
                
                #if no n-gram matches, we're looking at an unkown word
                if prob == 0:
                    prob += .0000000001
                
                totalprob += prob
            
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
    return models.index(best_fit)

def create_ngram_model(filename):
    """
        Accumulate trigram, bigram, and unigram counts using a corpus of lyrics from a certain genre of music
        Returns a dictionary containing the ngram counts collected from the corpus file
        """
    countsdict = defaultdict(float)
    unigrams = 0
    bigrams = 0
    trigrams = 0
    for line in open(filename):
        line = line.lower()
        words = line.split()
        for i in range(len(words)):
            # unigrams
            words[i] = words[i].strip('?.,/()!')
            countsdict[words[i].strip()]+=1.0
            unigrams += 1
            # bigrams
            if i == 0:
                countsdict[START_TAG+" "+words[i].strip()]+=1.0
                bigrams+=1
        
            elif i >= 1:
                countsdict[words[i-1].strip()+" "+words[i].strip()]+=1.0
                bigrams+=1
                
                if i == len(words)-1:
                    countsdict[words[i].strip()+" "+END_TAG]+=1.0
                    bigrams+=1
            
            # trigrams
            if i >= 2:
                countsdict[words[i-2].strip()+" "+words[i-1].strip()+" "+words[i].strip()]+=1.0
                trigrams+=1
                if i == len(words)-1:
                    countsdict[words[i-1].strip()+" "+words[i].strip()+" "+END_TAG]+=1.0
                    trigrams+=1
    #normalize the counts
    for key in countsdict:
        #print key
        if len(key.split())==1:
            countsdict[key] /= unigrams
        if len(key.split())==2:
            countsdict[key] /= bigrams
        if len(key.split())==3:
            countsdict[key] /= trigrams


    return countsdict

def generate_line(model):
    """
        Use an ngram model to generate a single line of lyrics for a certain genre of music
        """
    # Choose a random word to start the lyric. Choose from the set of words that follow a start tag.
    start = random.choice([ngram for ngram in model.keys() if START_TAG in ngram.split()]).split()[1]
    unigrams = [unigram for unigram in model.keys() if len(unigram.split()) == 1]
    totalUnigramCount = 0.0
    for unigram in unigrams:
        totalUnigramCount += model[unigram]
    sequence = start
    i = 1
    while(1):
        # Choose the next word in the generated sequence based on bigram probabilities
        nextword = ""
        bestProb = 0.0
        for token in unigrams:
            if sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token in model.keys():
                prob = model[sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token]/model[sequence.split()[i-1]+" "+token]
            elif sequence.split()[i-1]+" "+token in model.keys():
                prob = model[sequence.split()[i-1]+" "+token]/model[sequence.split()[i-1]]
            else:
                prob = model[token]/totalUnigramCount
            
            if prob > bestProb:
                bestProb = prob
                nextword = token
        # Exit the loop when the probability of ending the verse is greater than the probability of adding another word
        if model[sequence.split()[i-1]+" "+END_TAG]/model[sequence.split()[i-1]] > bestProb:
            break
        sequence = sequence+" "+nextword
        i += 1
    return sequence

def rhyme(w, pos):
    """
        Given a word and its POS tag, return a rhyming word that has the same part of speech
        """
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == w and pos == nltk.pos_tag([word])]
    rhyme = ""
    for (token, syllable) in syllables:
        for word, pron in entries:
            if pron[-2:] == syllable[-2:]:
                rhyme = word
                break;
    return rhyme

def output_lyrics(model,filename):
    """
        Outputs verses to file (groups of four lines where the last word of two consecutive lines matches)
        """
    output_file = open(filename,'w')
    previous_line = ""
    
    for i in range(1,LINES_PER_VERSE):
        current_line = generate_line(model)
        output = current_line+"\n"
        # Exchange the last word of the current line for a word the rhymes with the previous line
        if i%2 == 0:
            prev_word = previous_line.rsplit(' ', 1)[1]
            pos = nltk.pos_tag([prev_word])
            rhyme_word = rhyme(prev_word,pos)
            if len(rhyme_word) > 0:
                output = current_line.rsplit(' ', 1)[0]+" "+rhyme_word+"\n"
        
        print output
        previous_line = current_line
    output_file.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-generate', type = str, required = False, choices = ['rock','rap','pop'], help = 'generate genre lyrics (rock, rap, pop)')
    parser.add_argument('-classify', type = str, required = False,  help = 'classify the genre of an unlabeled set of lyrics')
    args = parser.parse_args()
    
    if args.generate:
        if not os.path.exists(args.generate+'.txt') or os.path.getsize(args.generate+'.txt') == 0:
            print "compiling "+args.generate+" corpus..."
            compile_corpus_for_genre(args.generate)
        filename = args.generate+'train.txt'
        model = create_ngram_model(filename)
        #generate_line(model)
        print "generating "+args.generate+" lyrics..."
        output_lyrics(model,'generate-'+args.generate+'.txt')
    
    if args.classify:
        ROCK = create_ngram_model('rocktrain.txt')
        RAP = create_ngram_model('raptrain.txt')
        POP = create_ngram_model('poptrain.txt')
        
        
        
        genre_list = ['Rock', 'Rap', 'Pop']
        genre_model_list = [ROCK, RAP, POP]
        
        errors = 0.0
        total = 0.0
        for item in genre_list:
            for n in range(20):
                guess = genre_list[classify(genre_model_list, item.lower() + 'test%s.txt' % n)]
                print item, guess
                if item != guess:
                    errors += 1
                total += 1


        print "Accuracy: " + str((total-errors)/total)
#print "Lyrics classified as " + genre_list[classify(genre_model_list, args.classify)] + "."