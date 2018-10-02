'''
todos
-----
- handle errors, have it display the type of exception
- handle multiple people, it does it just by skipping it right now
- learn how to use logger

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


PREDICTOR = dlib.shape_predictor('face-alignment/shape_predictor_68_face_landmarks.dat')
FA = FaceAligner(PREDICTOR, desiredFaceWidth=256)   

def aligner(face_pic, gray, box):   
    
    faceAligned = FA.align(face_pic, gray, box)    
    return faceAligned

def get_bounding_box_face(raw_image):
    
    rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image, model='cnn')
    
    return boxes


DETECTOR = dlib.get_frontal_face_detector()

def get_bounding_box_face2(raw_image):
    
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    boxes = DETECTOR(gray, 2)
    return [rect_to_bb(box) for box in boxes], boxes

def crop_out_face(image, box):
    
    x, y, w, h = box
    
    return imutils.resize(image[y:y + h, x:x + w], width=256)


def save_to_drive(img_path, aligned_face, save_to_folder): 
    file_name = img_path.split('/')[1]
    cv2.imwrite(
        save_to_folder + '/' + file_name.split('.')[0] + '_face.jpg',
        aligned_face 
        )

    print ('saved: ' + save_to_folder + '/' + file_name.split('.')[0] + '_face.jpg')
    return None

def multithread_map(fn, work_list, num_workers=50):
    from concurrent.futures import ThreadPoolExecutor
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))


def iterate_through_folder(folder_name):
    imagePaths = list(paths.list_images(folder_name))

    multithread_map(main, imagePaths)

def main(img_path):

    root_folder = img_path.split('/')[0]

    # try:
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=800)
    boxes, rects = get_bounding_box_face2(image)
    
    if len(boxes) == 1:
        box = boxes[0]
        faceOrig = crop_out_face(image, box)
        aligned_face = aligner(image, image, rects[0] )
        save_to_drive(img_path, aligned_face, root_folder + '_faces')
        
    # except:
        # print ('error -- ', img_path)

    
if __name__ == "__main__":
    iterate_through_folder('friends_and_myself_pics')
    

