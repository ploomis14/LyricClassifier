"""GUI for lyric generator"""

from Tkinter import *
from ttk import *
import os
import os.path
from generator import nltk_process

class GUI(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        self.OPTIONS = [
            "pop",
            "rock",
            "rap"
        ]

        self.cached_models = {}

        self.initUI()
    
    def initUI(self):
        self.parent.title("Song Generator")
        self.style = Style()
        self.style.theme_use("default")

        self.pack(fill=BOTH, expand=1)

        quitButton = Button(self, text="Generate Lyrics",
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
        filename = 'generate-'+self.genre.get()+'.txt'
        self.cached_models = nltk_process(self.genre.get(), self.cached_models)
        if os.path.exists(filename):
            f = open(filename,'r')
            self.lyrics = f.read()
            text = text = Text(self.parent)
            text.insert(END,self.lyrics)
            text.place(x=20,y=200)

def main():
    root = Tk()
    root.geometry("500x500+500+500")
    app = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()