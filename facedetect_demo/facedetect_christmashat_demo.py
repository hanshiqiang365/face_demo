#author: hanshiqiang365 （微信公众号：韩思工作室）
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
import cv2
import random
import pygame

def main(file,blur_rate):
    try:
     blur_rate = float(blur_rate)
    except:
        blur_rate = 1.25

    face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sample_image = cv2.imread(file) # file
    faces = face_patterns.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=8,minSize=(50, 50))

    hats = []
    for i in range(4):
        hats.append(cv2.imread('resources/christmas_hat%d.png' % i, -1))

    for face in faces:
        hat = random.choice(hats)
        scale = face[3] / hat.shape[0] * blur_rate #1.25
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)

        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2)
        y_offset = int(face[1] - hat.shape[0] / 2)

        x1, x2 = max(x_offset, 0), min(x_offset + hat.shape[1], sample_image.shape[1])
        y1, y2 = max(y_offset, 0), min(y_offset + hat.shape[0], sample_image.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1

        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h

        for c in range(0, 3):
            sample_image[y1:y2, x1:x2, c] = (alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * sample_image[y1:y2, x1:x2, c])

    output_file = "resources/christmashat_result.png"
    cv2.imwrite(output_file, sample_image)
    
    return output_file


root = Tk()
root.title('AI Face++ Detect + Christmas Hat（人脸探测加上圣诞帽 ） —— Developed by hanshiqiang365 （微信公众号 - 韩思工作室）')
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

global file_, rate, seg_img_path

# 原图展示
def show_original1_pic():
    global file_
    file_ = askopenfilename(title='选择文件')
    print(file_)
    Img = PIL.Image.open(r'{}'.format(file_))
    Img = Img.resize((430,270),PIL.Image.ANTIALIAS) 
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.config(image=img_png_original)
    label_Img_original1.image = img_png_original 
    cv_orinial1.create_image(5, 5,anchor='nw', image=img_png_original)

# 特效展示
def show_detectface_pic():
    global file_,seg_img_path
    print(entry.get())
    mor_img_path = main(file_,entry.get())
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    Img = Img.resize((430, 270), PIL.Image.ANTIALIAS)
    img_png_seg = ImageTk.PhotoImage(Img)
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg 


def quit():
    root.destroy()


Button(root, text = "打开图片", width=15, height=2, font=10, command = show_original1_pic).place(x=50,y=180)
Button(root, text = "特效图片生成", width=15, height=2, font=10, command = show_detectface_pic).place(x=50,y=270)
Button(root, text = "退出软件", width=15, height=2, font=10, command = quit).place(x=50,y=360)

Label(root,text = "变形系数",width=15, height=1, font=10).place(x=50,y=10)

entry = Entry(root, width=15, font=20)
entry.insert(END, '1.25')
entry.place(x=230,y=10)


Label(root,text = "原版图片",width=30, height=2, font=10).place(x=310,y=100)
cv_orinial1 = Canvas(root,bg = 'white',width=430,height=270)
cv_orinial1.create_rectangle(8,8,425,265,width=1,outline='green')
cv_orinial1.place(x=230,y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=230,y=150)

Label(root, text="特效图片", width=30, height=2, font=10).place(x=780,y=100)
cv_seg = Canvas(root, bg='white', width=430,height=270)
cv_seg.create_rectangle(8,8,425,265,width=1,outline='green')
cv_seg.place(x=700,y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=700,y=150)


root.mainloop()
