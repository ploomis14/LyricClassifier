"""Song Lyric Generator
"""

from lxml import html
import requests, argparse

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
            f.write(verse.encode('utf-8'))
    f.close()

def create_ngram_model(filename):
    """
    Accumulate trigram and unigram counts using a corpus of lyrics from a certain genre of music
    """
    print filename

if __name__=='__main__':
    genres = ['rock','hiphop','pop']
    for g in genres:
        compile_corpus_for_genre(g)
    parser = argparse.ArgumentParser()
    parser.add_argument('-generate', type = str, required = False, choices = ['rock','hiphop','pop'], help = 'generate genre lyrics (genre)')
    args = parser.parse_args()
    if args.generate:
        create_ngram_model(args.generate+'.txt')
