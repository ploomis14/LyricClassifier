SONG LYRIC GENERATOR/CLASSIFIER
-------------------------------
Name: scrapelyrics

Usage:
    python scrapelyrics.py -generate [rock,pop,rap]
    python scrapelyrics.py -classify  [filename]

Description: This application scrapes music websites to compile a corpora of lyrics for certain genres of music.
Generate - The scrapelyrics program will collect ngram probabilities from the corpus of the genre entered. It will use the ngram model to
generate lines of lyrics. Generated lyrics will be output to a file named "[genre]-generate.txt"
Classify - The scrapelyrics program will read a text file containing song lyrics, and predict the genre of the song.
