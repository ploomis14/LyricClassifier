"""Song Lyric Generator"""

import grequests
from lxml import html
from collections import defaultdict
import nltk
from nltk.corpus import cmudict
import requests
import argparse
import string
import random
import os
import os.path
import re
import math
import sys
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import nltk
from nltk import bigrams
from nltk import trigrams
import traceback

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

def approx_nsyl(word):
	"""Credit - Jason Sundram, http://runningwithdata.com/post/3576752158/w
	Return the max syllable count in the case of multiple pronunciations"""
	d = cmudict.dict()
	if word not in d.keys():
		return 0
	return max([len([y for y in x if y[-1].isdigit()]) for x in d[word.lower()]])

def generate_line(key):
	"""
	Use an ngram model to generate a single line of lyrics for a certain genre of music
	"""
	# Choose a random word to start the lyric. Choose from the set of words that follow a start tag.
	while True:
		start = random.choice([ngram for ngram in model.keys() if START_TAG in ngram.split()]).split()[1]
		if not start.isdigit() and len(start)>1:
			break
		
	sequence = start
	i = 1
	# print "finished loading"

	# Continuity among the syllable length of lines
	# There should be increased probability of ending a line when the maximum syllable length is exceeded
	end_of_line_prob = 0.0
	syllables = 1
	try:
		while(1):
			# Choose the next word in the generated sequence based on bigram probabilities
			nextword = ""
			bestProb = prob =0.0
			for token in NGRAM[sequence.split()[-1]]:
				# print "token",sequence.split()[-1],token
				if sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token in model.keys():
					prob = model[sequence.split()[i-2]+" "+sequence.split()[i-1]+" "+token]/model[sequence.split()[i-2]+" "+sequence.split()[i-1]]
				else:
					prob = model[sequence.split()[i-1]+" "+token]/model[sequence.split()[i-1]]
				
				
				if prob > bestProb:
					bestProb = prob
					nextword = token

			print key,i,sequence.split()

			end_of_line_prob = model[sequence.split()[-1]+" "+END_TAG]/model[sequence.split()[-1]]
			if syllables > MAX_SYLLABLES:
				end_of_line_prob += 0.8
			if syllables < MIN_SYLLABLES:
				end_of_line_prob -= 0.5

			# Exit the loop when the probability of ending the verse is greater than the probability of adding another word
			if end_of_line_prob > bestProb or nextword == "":
				break
			sequence = sequence+" "+nextword
			syllables += approx_nsyl(nextword)

			i += 1
	except Exception, e:
		traceback.print_exc(e)
		raise e
	

	return {key:sequence}

def rhyme(w, pos):
	"""Given a word and its POS tag, return a rhyming word that has the same part of speech"""
	entries = nltk.corpus.cmudict.entries()
	syllables = [(word, syl) for word, syl in entries if word == w and pos == nltk.pos_tag([word])]
	rhymes = [word for (token, syllable) in syllables for word, pron in entries if pron[-2:] == syllable[-2:]]
	if len(rhymes) == 0:
		return w
	return rhymes[random.randint(0,len(rhymes)-1)]

def lines_generated(kwargs):
	retval = {}
	for i in kwargs:
		retval.update(i)
	print retval

	output_file = open(FILENAME,'w')
	previous_line = ""
	lyrics = ""

	for v in range(1,VERSES_PER_SONG+1):
		for i in range(1,LINES_PER_VERSE+1):
			current_line = retval[i-1+VERSES_PER_SONG*(v-1)]
			output = current_line+"\n"
			# Exchange the last word of the current line for a word the rhymes with the previous line
			if i%2 == 0:
				prev_word = previous_line.split()[-1]
				pos = nltk.pos_tag([prev_word])
				rhyme_word = rhyme(prev_word,pos)
				if len(rhyme_word) > 0:
					output = current_line.rsplit(' ', 1)[0]+" "+rhyme_word+"\n"
			lyrics = lyrics+output
			print output
			previous_line = current_line
		lyrics = lyrics+"\n"
	output_file.write(lyrics)
	output_file.close()

def output_lyrics(filename):
	"""
	Outputs verses to file (groups of four lines where the last word of two consecutive lines matches)
	"""
	print filename
	global FILENAME
	FILENAME = filename
	
	inputs = [i for i in range(VERSES_PER_SONG*LINES_PER_VERSE)]
	import multiprocessing
	print "inputs",inputs
	p = multiprocessing.Pool(8)
	try:
	   p.map_async(generate_line, inputs, callback=lines_generated)
	   p.close()
	   p.join()
	except Exception as e:
	   traceback.print_exc(e)
	   print e

def nltk_process(genre, cached_models):
	filename = genre+'.txt'
	global model, NGRAM
	print "compiling corpus for "+genre+"..."
	if not os.path.exists(filename) or os.path.getsize(filename) == 0:
		compile_corpus_for_genre(genre,filename,CORPUS_SIZE)
	print "done."
	
	if genre not in cached_models.keys():
		model, NGRAM = create_ngram_model(filename)
		cached_models[genre] = model
		cached_models[genre+"ngram"] = NGRAM
	
	model = cached_models[genre]
	NGRAM = cached_models[genre+"ngram"]

	print "generating "+genre+" lyrics..."
	output_lyrics('generate-'+genre+'.txt')
	return cached_models
		
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-generate', type = str, required = False, choices = ['rock','rap','pop'], help = 'generate genre lyrics (rock, rap, pop)')
	parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics')
	args = parser.parse_args()
	
	if args.generate:
		nltk_process(args.generate)