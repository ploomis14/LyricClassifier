#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ziyuanliu
# @Date:   2014-11-16 20:35:36
# @Last Modified by:   ziyuanliu
# @Last Modified time: 2014-11-17 22:11:28

from Tkinter import *
from ttk import *
# from scrapelyrics import *
from optimizedgenerator import nltk_process
import tkMessageBox as box


class GUI(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        self.OPTIONS = [
            "pop",
            "rock",
            "rap"
        ]

        self.initUI()
    
    def initUI(self):
        self.parent.title("Song Generator")
        self.style = Style()
        self.style.theme_use("default")

        self.pack(fill=BOTH, expand=1)

        quitButton = Button(self, text="Generate",
            command=self.generatelyrics)
        quitButton.place(x=250, y=20)

        lb = Listbox(self)
        for i in self.OPTIONS:
            lb.insert(END, i)
            
        lb.bind("<<ListboxSelect>>", self.onSelect)    
        lb.place(x=20, y=20)

        self.lyrics = ""
        self.genre = StringVar()

    def onSelect(self, val):
        sender = val.widget
        idx = sender.curselection()
        value = sender.get(idx)   
        self.genre.set(value)

    def generatelyrics(self):
        print "generating lyrics for",self.genre.get()
        nltk_process(self.genre.get())
        # generator = LyricGenerator(self.genre.get())
        filename = 'generate-'+self.genre.get()+'.txt'
        # generator.output_lyrics(filename)
        f = open(filename,'r')
        self.lyrics = f.read()
        box.showinfo("Lyrics for {0}".format(self.genre.get()), self.lyrics)



def main():
  
    root = Tk()
    root.geometry("350x350+300+300")
    app = GUI(root)
    root.mainloop()  


if __name__ == '__main__':
    main()
