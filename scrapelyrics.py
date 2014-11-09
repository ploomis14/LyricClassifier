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

START_TAG = "<s>"
END_TAG = "</s>"
CORPUS_SIZE = 4
LINES_PER_VERSE = 4

def compile_corpus_for_genre(genre):
    """
    Scrapes website for song lyrics and compiles corpora with lyrics from each genre of music.
    Compile a corpus of lyrics for a certain genre of music
    """
    f = open(genre+'.txt','w')
    for i in range(CORPUS_SIZE):
        page = requests.get('http://genius.com/tags/'+genre+'/all?page='+str(i))
        tree = html.fromstring(page.text)
        songs = tree.xpath('//*[@class=" song_link"]/@href')
        for song in songs:
            lyric_page = requests.get(song)
            song_tree = html.fromstring(lyric_page.text)
            verses = song_tree.xpath('//*[@data-editorial-state="accepted"]/text()')
            for verse in verses:
                f.write(verse.encode('utf-8')+'\n')
    f.close()

def generate_key(seq):
    key = ""
    for item in seq:
        key += item + " "
    
    return key

def classify(models, filename):
    maxprob = 0 
    
    #find p(lyrics) for each model
    for model in models:
        totalprob = 0
        backpointers = ['<s>','<s>','<s>']
        for line in open(filename):
            line = line.split().lower()
            
            #for each word in the file, push it into the backpointers list - everything moves one to the left
            for word in line:
                prob = 0
                for i in range(len(backpointers) - 1):
                    backpointers[len(backpointers) - i - 2] = backpointers[len(backpointers) - i - 1]
                    
                backpointers[len(backpointers)-1] = word
                
                
                #use interpolation to compute probabilities of n-grams found in the backpointers list
                j = 0
                l = .9
                while j < len(backpointers):
                    key = generate_key(backpointers[j:len(backpointers)-1])
                    prob += l*model[key]
                    j+=1
                    l*=.1
                
                #if no n-gram matches, we're looking at an unkown word
                if prob == 0:
                    prob += l *model['<UNK>']
                
                totalprob += prob
        
        #update the best-performing model
        if totalprob > maxprob:
            maxprob = totalprob
            best_fit = model
        
        return models.index(best_fit)
                
def create_ngram_model(filename):
    """
    Accumulate trigram, bigram, and unigram counts using a corpus of lyrics from a certain genre of music
    Returns a dictionary containing the ngram counts collected from the corpus file
    """
    countsdict = defaultdict(float)
    for line in open(filename):
        words = line.split()
        for i in range(len(words)):
            # unigrams
            countsdict[words[i].strip()]+=1.0
            # bigrams
            if i == 0:
                countsdict[START_TAG+" "+words[i].strip()]+=1.0
            elif i >= 1:
                countsdict[words[i-1].strip()+" "+words[i].strip()]+=1.0
                if i == len(words)-1:
                    countsdict[words[i].strip()+" "+END_TAG]+=1.0
            # trigrams
            if i >= 2:
                countsdict[words[i-2].strip()+" "+words[i-1].strip()+" "+words[i].strip()]+=1.0
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

def output_lyrics(model,filename):
    """
    Outputs verses to file (groups of four lines where the last word of two consecutive lines matches)
    """
    words = [token for token in model.keys() if len(token.split()) == 1]
    pron_dict = cmudict.dict()
    output_file = open(filename,'w')
    previous_line = generate_line(model)
    for i in range(LINES_PER_VERSE):
        current_line = generate_line(model)
        
        # Exchange the last word of the current line for a word the rhymes with the previous line
        if i%2 == 0:
            prev_word = previous_line.rsplit(' ', 1)[1]
            prev_pron = pron_dict[last_word.lower()][0]
            
            for word in words:
                pron = pron_dict[word.lower()][0]
                if pron[len(pron)-1] == prev_pron[len(prev_pron)-1]:
                    output_file.write(current_line.rsplit(' ', 1)[0]+" "+word)
                    break;
        else:
            output_file.write(current_line)

        previous_line = current_line
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-generate', type = str, required = False, choices = ['rock','rap','pop'], help = 'generate genre lyrics (rock, rap, pop)')
    parser.add_argument('-classify', type = str, required = False,  help = 'classify the genre of an unlabeled set of lyrics')
    args = parser.parse_args()

    if args.generate:
        if not os.path.exists(args.generate+'.txt') or os.path.getsize(args.generate+'.txt') == 0:
            print "compiling "+args.generate+" corpus..."
            compile_corpus_for_genre(args.generate)
        filename = args.generate+'.txt'
        model = create_ngram_model(filename)
        #generate_line(model)
        print "generating "+args.generate+" lyrics..."
        output_lyrics(model,'generate-'+args.generate+'.txt')
    
    if args.classify:
        ROCK = create_ngram_model('rock.txt')
        HIPHOP = create_ngram_model('rap.txt')
        POP = create_ngram_model('pop.txt')
        
        genre_list = ['Rock', 'Rap', 'Pop']
        genre_model_list = [ROCK, RAP, POP]
        
        print "Lyrics classified as " + genre_list[classify(genre_model_list)] + "."
            
    
