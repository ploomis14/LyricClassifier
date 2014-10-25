"""Song Lyric Collector
scrapes the website metrolyrics.com for song lyrics and compiles corpora with lyrics from each genre of music
"""

from lxml import html
import requests

"""Compile a corpus of lyrics for a certain genre of music
"""
def compile_corpus_for_genre(genre):
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

if __name__=='__main__':
    genres = ['pop','rock','hiphop']
    for genre in genres:
        compile_corpus_for_genre(genre)