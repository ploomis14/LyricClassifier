#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ziyuanliu
# @Date:   2014-11-17 12:24:49
# @Last Modified by:   ziyuanliu
# @Last Modified time: 2014-11-17 22:28:22

import grequests
from scrapelyrics import *
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import nltk
from nltk import bigrams
from nltk import trigrams
import traceback 

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

def approx_nsyl(word):
	"""
	Approximates the number of syllables in a word
	"""
	d = cmudict.dict()
	if word not in d.keys():
		return 0
	x = d[word.lower()][0]
	return len(list(y for y in x if isdigit(y[-1])))

def generate_line(key):
	"""
	Use an ngram model to generate a single line of lyrics for a certain genre of music
	"""
	# Choose a random word to start the lyric. Choose from the set of words that follow a start tag.
	start = random.choice([ngram for ngram in model.keys() if START_TAG in ngram.split()]).split()[1]
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
				end_of_line_prob += 0.6
			if syllables < MIN_SYLLABLES:
				end_of_line_prob -= 0.2

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


	

def nltk_process(genre):
	#check folder 
	filename = genre+'.txt'
	if not os.path.exists(filename) or os.path.getsize(filename) == 0:
		compile_corpus_for_genre(genre,filename,CORPUS_SIZE)

	global model, NGRAM
	model, NGRAM = create_ngram_model(filename)	
	print "generating "+genre+" lyrics..."
	output_lyrics('generate-'+genre+'.txt')
		
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-generate', type = str, required = False, choices = ['rock','rap','pop'], help = 'generate genre lyrics (rock, rap, pop)')
	parser.add_argument('-classify', help = 'classify the genre of an unlabeled set of lyrics')
	args = parser.parse_args()
	
	if args.generate:
		nltk_process(args.generate)

		# filename = args.generate+'.txt'
		# if not os.path.exists(filename) or os.path.getsize(filename) == 0:
		# 	print "getting corpus"
		# 	compile_corpus_for_genre(args.generate,filename,CORPUS_SIZE)
		# global NGRAM_MODEL
		# NGRAM_MODEL = nltk_process(filename)

		
	




