"""GUI for lyric generator"""

from Tkinter import *
from ttk import *
import os
import os.path
from generator import nltk_process
import tkMessageBox

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

        self.generateButton = Button(self, text="Generate Lyrics",
            command=self.generatelyrics,state = DISABLED)
        self.generateButton.place(x=250, y=20)

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
        self.generateButton['state'] = 'enabled'

    def generatelyrics(self):
        genre = self.genre.get()
        if genre is not None and len(genre)>0:

            filename = 'generate-'+self.genre.get()+'.txt'
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(processes=1)

            async_result = pool.apply_async(nltk_process, (self.genre.get(), self.cached_models))
            self.cached_models = async_result.get()


            # self.cached_models = nltk_process(self.genre.get(), self.cached_models)
            if os.path.exists(filename):
                f = open(filename,'r')
                self.lyrics = f.read()
                text = Text(self.parent)
                text.insert(END,self.lyrics)
                text.height = 20
                text.width = 60
                text.pack(side="left", fill="both", expand=True)
                text.place(x=20,y=200)
                
        

def main():
    root = Tk()
    root.geometry("610x600+200+100")
    app = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()