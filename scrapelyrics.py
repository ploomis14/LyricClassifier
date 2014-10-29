"""Song Lyric Generator
"""

from lxml import html
from collections import defaultdict
import requests, argparse, string, random

START_TAG = "<s>"
VERSE_LENGTH = 10

def compile_corpus_for_genre(genre):
    """
    Scrapes the website metrolyrics.com for song lyrics and compiles corpora with lyrics from each genre of music.
    Compile a corpus of lyrics for a certain genre of music
    """
    f = open(genre+'.txt','w')
    page = requests.get('http://www.metrolyrics.com/top100-'+genre+'.html')
    tree = html.fromstring(page.text)
    songs = tree.xpath('//*[@class="song-link hasvidtoplyric"]/@href')
    for song in songs:
        lyric_page = requests.get(song)
        song_tree = html.fromstring(lyric_page.text)
        verses = song_tree.xpath('//*[@class="verse"]/text()')
        for verse in verses:
            f.write(verse.encode('utf-8')+'\n')
    f.close()

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
            # trigrams
            if i >= 2:
                countsdict[words[i-2].strip()+" "+words[i-1].strip()+" "+words[i].strip()]+=1.0
    return countsdict

def generate_lyrics(model):
    """
    Use an ngram model to generate lines of lyrics for a certain genre of music
    """
    # Choose a random word to start the lyric. Choose from the set of words that follow a start tag.
    start = random.choice([ngram for ngram in model.keys() if START_TAG in ngram.split()]).split()[1]
    unigrams = [ngram for ngram in model.keys() if len(ngram.split()) == 1]
    sequence = start
    for i in range(1, VERSE_LENGTH):
        # Choose the next word in the generated sequence based on bigram probabilities
        nextword = ""
        bestBigramProb = 0.0
        for token in unigrams:
            if sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token in model.keys():
                prob = model[sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token]/model[sequence.split()[i-1]+" "+token]
            elif sequence.split()[i-1]+" "+token in model.keys():
                prob = model[sequence.split()[i-1]+" "+token]/model[sequence.split()[i-1]]
            else:
                prob = model[token]/sum(model.values())
            
            if prob > bestBigramProb:
                bestBigramProb = prob
                nextword = token
        sequence = sequence+" "+nextword
    print sequence

if __name__=='__main__':
    #genres = ['rock','hiphop','pop']
    #for g in genres:
    #    compile_corpus_for_genre(g)
    parser = argparse.ArgumentParser()
    parser.add_argument('-generate', type = str, required = False, choices = ['rock','hiphop','pop'], help = 'generate genre lyrics (genre)')
    args = parser.parse_args()
    if args.generate:
        model = create_ngram_model(args.generate+'.txt')
        generate_lyrics(model)
