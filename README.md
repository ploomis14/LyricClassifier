SONG LYRIC GENERATOR/CLASSIFIER
-------------------------------
Description: This application scrapes music websites to compile corpora of lyrics for certain genres of music. It uses ngram models to generate verses of lyrics for a specified genre. It also classifies songs by genre.

gui.py - User interface for generator. Usage - python gui.py

generator.py - The scrapelyrics program will collect ngram probabilities from the corpus of the genre entered. It will use the ngram model to generate lines of lyrics. Generated lyrics will be output to a file named "[genre]-generate.txt". Run using gui.py

classifier_ngram.py - Contains class and supporting functions for lyric classifier. Will test on a set of 20 songs and output resulting accuracy.
Usage - python classifier_ngram.py -classify CLASSIFY