#author: hanshiqiang365 （微信公众号：韩思工作室）

from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
import numpy as np
import cv2
import dlib
import pygame
from scipy.spatial import Delaunay


predictor_model = 'shape_predictor_68_face_landmarks.dat'


def get_points(image): 

    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])
        
    x = image.shape[1] - 1
    y = image.shape[0] - 1

    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)


def get_triangles(points):
    return Delaunay(points).simplices


def affine_transform(input_image, input_triangle, output_triangle, size):
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image


def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha): 
    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                     rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask


def morph_faces(filename1, filename2, alpha=0.5):
    img1 = cv2.imread(str(filename1))
    img2 = cv2.imread(str(filename2))
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_CUBIC)
    print('img1.shape',img1.shape)
    print('img2.shape',img2.shape)

    points1 = get_points(img1)
    print('pionts1:',len(points1),points1)
    points2 = get_points(img2)
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    import pandas as pd
    p = pd.DataFrame(points)
    p.to_csv('resources/facemorhper.csv')

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    triangles = get_triangles(points)
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)

    return np.uint8(img_morphed)


def main(file1,file2,alpha):
    try:
     alpha = float(alpha)
    except:
        alpha = 0.5
    img_morphed = morph_faces(file1, file2, alpha)
    output_file = 'resources/{}_{}_{}.jpg'.format(
        file1.split('.')[0][-2:], file2.split('.')[0][-1:], alpha)
    cv2.imwrite(output_file, img_morphed)
    return output_file


root = Tk()
root.title('AI Face++ Morhper Tool （ 人脸融合） —— Developed by hanshiqiang365 （微信公众号 - 韩思工作室）')
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


# 人脸融合效果展示
def show_morpher_pic():
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


Button(root, text = "打开图片1", width=12, height=1, font=10, command = show_original1_pic).place(x=50,y=180)
Button(root, text = "打开图片2", width=12, height=1, font=10, command = show_original2_pic).place(x=50,y=240)
Button(root, text = "人脸融合生成", width=12, height=1, font=10, command = show_morpher_pic).place(x=50,y=300)
Button(root, text = "退出", width=12, height=1, font=10, command = quit).place(x=50,y=360)

Label(root,text = "融合系数",width=12, height=1, font=10).place(x=50,y=10)

entry = Entry(root, text = "0.5", width=15, font=20)
entry.place(x=230,y=10)


Label(root,text = "图片1",width=12, height=1, font=10).place(x=300,y=100)
cv_orinial1 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial1.create_rectangle(8,8,260,260,width=1,outline='green')
cv_orinial1.place(x=230,y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=230,y=150)


Label(root,text="图片2",width=12, height=1, font=10).place(x=620,y=100)
cv_orinial2 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial2.create_rectangle(8,8,260,260,width=1,outline='green')
cv_orinial2.place(x=550,y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=550,y=150)

Label(root, text="融合效果", width=12, height=1, font=10).place(x=940,y=100)
cv_seg = Canvas(root, bg='white', width=270,height=270)
cv_seg.create_rectangle(8,8,260,260,width=1,outline='green')
cv_seg.place(x=870,y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=870,y=150)


root.mainloop()
