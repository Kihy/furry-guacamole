from tkinter import *
from PIL import Image, ImageTk

class App_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Le Snake")

        # image=Image.open("images/background.jpg")
        # image=image.resize((1600,900),Image.ANTIALIAS)
        # self.photo = ImageTk.PhotoImage(image)

        self.canvas = Canvas(self.master,width=1600, height=900,bg='white')

        # w.create_image(10,10,image=self.photo)
        self.canvas.pack(fill=BOTH,expand=YES)

    def draw_circle(self, position, radius):
        self.canvas.create_oval(position[0]+radius,position[1]+radius,
            position[0]-radius,position[1]-radius,fill='red')
