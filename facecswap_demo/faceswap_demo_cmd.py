#author: hanshiqiang365（微信公众号：韩思工作室）
import cv2
import dlib
import numpy as np
from PIL import Image

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def transformation_from_points(points1, points2):

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

def get_eye_faceswap(img_src,record):
    img = cv2.imread(img_src)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        face_matrix = np.matrix([[p.x, p.y] for p in landmarks.parts()])
        record[img_src] = face_matrix

def get_eye(img_src,record):
    img = cv2.imread(f"{img_src}.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        face_matrix = np.matrix([[p.x, p.y] for p in landmarks.parts()])
        record[img_src] = face_matrix

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
def correct_colours(im1, im2, landmarks1,blur_rate):
    blur_amount = blur_rate * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def exchange(p1,p2,record):
    im1 = Image.open(f"{p1}.jpg")
    im1_eye = Image.open(f"{p2}_eye_resized.jpg")
    im1.paste(im1_eye,(record[p1][0],record[p1][1]))
    im1.save(f"{p1}_result.jpg")

    im2 = Image.open(f"{p2}.jpg")
    im2_eye = Image.open(f"{p1}_eye_resized.jpg")
    im2.paste(im2_eye,(record[p2][0],record[p2][1]))
    im2.save(f"{p2}_result.jpg")

def process_faceswap(src1,src2,record,blur_rate):
    get_eye_faceswap(src1, record)
    get_eye_faceswap(src2, record)

    M = transformation_from_points(record[src1][ALIGN_POINTS],
                                   record[src2][ALIGN_POINTS])
    im1 = cv2.imread(src1)
    im2 = cv2.imread(src2)
    mask = get_face_mask(im2, record[src2])
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, record[src1]), warped_mask],
                           axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, record[src1],blur_rate)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    cv2.imwrite("resources/result.jpg", output_im)

def process(src1,src2,record,blur_rate):
    get_eye(src1, record)
    get_eye(src2, record)

    M = transformation_from_points(record[src1][ALIGN_POINTS],
                                   record[src2][ALIGN_POINTS])
    im1 = cv2.imread(f"{src1}.jpg")
    im2 = cv2.imread(f"{src2}.jpg")
    mask = get_face_mask(im2, record[src2])
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, record[src1]), warped_mask],
                           axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, record[src1],blur_rate)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    cv2.imwrite(f"{src1}_result.jpg", output_im)

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

record ={}

#修改此处命名“图片1名称”“图片2名称”后运行即可
src1,src2 = "resources/testpic1","resources/testpic2"
process(src1,src2,record,COLOUR_CORRECT_BLUR_FRAC)
process(src2,src1,record,COLOUR_CORRECT_BLUR_FRAC)
