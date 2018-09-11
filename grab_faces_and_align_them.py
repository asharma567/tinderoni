'''
todos
-----
- handle errors, have it display the type of exception
- handle multiple people, it does it just by skipping it right now
- figuring out gender
- learn how to use logger

DONE
- run it again to test it out
- error handling
'''

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import argparse
import dlib
import face_recognition
import argparse
import pickle
import os
import cv2
import imutils
import matplotlib.pyplot as plt


PREDICTOR = dlib.shape_predictor('/Users/ajay/Downloads/face-alignment/shape_predictor_68_face_landmarks.dat')
FA = FaceAligner(PREDICTOR, desiredFaceWidth=256)   

def aligner(face_pic, gray, box):   
    
    faceAligned = FA.align(face_pic, gray, box)    
    return faceAligned

def get_bounding_box_face(raw_image):
    
    rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image,model='cnn')
    
    return boxes


DETECTOR = dlib.get_frontal_face_detector()

def get_bounding_box_face2(raw_image):
    
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    boxes = DETECTOR(gray, 2)
    return [rect_to_bb(box) for box in boxes], boxes

def crop_out_face(image, box):
    
    x, y, w, h = box
    
    return imutils.resize(image[y:y + h, x:x + w], width=256)


def save_to_drive(img_path, aligned_face): 
    file_name = img_path.split('/')[1]
    cv2.imwrite(
        'tinder_pics_likes_faces_deduped/' + file_name.strip('.jpeg') + '_face.png',
        aligned_face 
        )
    print ('saved: ' + file_name)
    return None

def iterate_through_folder(folder_name='tinder_pics_likes_deduped'):
    imagePaths = list(paths.list_images(folder_name))
    # imagePaths_saved_already = list(paths.list_images('tinder_pics_likes_faces'))
    
    for i, image_path in enumerate(imagePaths):
        print (i/float(len(imagePaths)))
        main(image_path)

def main(img_path):

    try:
        image = cv2.imread(img_path)
        image = imutils.resize(image, width=800)
        boxes, rects = get_bounding_box_face2(image)
        
        if len(boxes) == 1:
            box = boxes[0]
            faceOrig = crop_out_face(image, box)
            aligned_face = aligner(image, image, rects[0] )
            save_to_drive(img_path, aligned_face)
        
    except:
        print ('error -- ', img_path)

    
if __name__ == "__main__":
    iterate_through_folder('tinder_pics_likes_deduped')
    # main('tinder_pics_likes/1526004657_Dian_5.jpeg')


