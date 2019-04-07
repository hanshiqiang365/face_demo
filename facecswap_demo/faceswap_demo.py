#author: hanshiqiang365 （微信公众号：韩思工作室）
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
#import numpy as np
import cv2
import pygame
#from scipy.spatial import Delaunay
import faceswap_demo_cmd as fs

def main(file1,file2,blur_rate):
    try:
     blur_rate = float(blur_rate)
    except:
        blur_rate = 0.5
        
    record ={}
    fs.process_faceswap(file1,file2,record,blur_rate)

    output_file = "resources/result.jpg"
    return output_file


root = Tk()
root.title('AI Face++ Swap Tool （ 换脸） —— Developed by hanshiqiang365 （微信公众号 - 韩思工作室）')
root.geometry('1200x500')
root.iconbitmap('demo.ico')

pygame.mixer.init()
pygame.mixer.music.load("demo_bgm.wav")
pygame.mixer.music.play(-1)

decoration = PIL.Image.open('demo_bg.jpg').resize((1200, 500))
render = ImageTk.PhotoImage(decoration)
img = Label(image=render)
img.image = render
img.place(x=0, y=0)

global file1_, file2_, rate, seg_img_path


# 原图1展示
def show_original1_pic():
    global file1_
    file1_ = askopenfilename(title='选择文件')
    print(file1_)
    Img = PIL.Image.open(r'{}'.format(file1_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS) 
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.config(image=img_png_original)
    label_Img_original1.image = img_png_original 
    cv_orinial1.create_image(5, 5,anchor='nw', image=img_png_original)


# 原图2展示
def show_original2_pic():
    global file2_
    file2_ = askopenfilename(title='选择文件')
    print(file2_)
    Img = PIL.Image.open(r'{}'.format(file2_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS)
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original2.config(image=img_png_original)
    label_Img_original2.image = img_png_original 
    cv_orinial2.create_image(5, 5,anchor='nw', image=img_png_original)


# 换脸效果展示
def show_swapface_pic():
    global file1_,seg_img_path,file2_
    print(entry.get())
    mor_img_path = main(file1_,file2_,entry.get())
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)
    img_png_seg = ImageTk.PhotoImage(Img)
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg 


def quit():
    root.destroy()


Button(root, text = "打开图片1", width=15, height=2, font=10, command = show_original1_pic).place(x=50,y=180)
Button(root, text = "打开图片2", width=15, height=2, font=10, command = show_original2_pic).place(x=50,y=240)
Button(root, text = "换脸图片生成", width=15, height=2, font=10, command = show_swapface_pic).place(x=50,y=300)
Button(root, text = "退出", width=15, height=2, font=10, command = quit).place(x=50,y=360)

Label(root,text = "换脸系数",width=15, height=1, font=10).place(x=50,y=10)

entry = Entry(root, text = "0.5", width=15, font=20)
entry.place(x=230,y=10)


Label(root,text = "图片1",width=15, height=2, font=10).place(x=300,y=100)
cv_orinial1 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial1.create_rectangle(8,8,260,260,width=1,outline='green')
cv_orinial1.place(x=230,y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=230,y=150)


Label(root,text="图片2",width=15, height=2, font=10).place(x=620,y=100)
cv_orinial2 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial2.create_rectangle(8,8,260,260,width=1,outline='green')
cv_orinial2.place(x=550,y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=550,y=150)

Label(root, text="换脸效果", width=15, height=2, font=10).place(x=940,y=100)
cv_seg = Canvas(root, bg='white', width=270,height=270)
cv_seg.create_rectangle(8,8,260,260,width=1,outline='green')
cv_seg.place(x=870,y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=870,y=150)


root.mainloop()
