'''

'''
import numpy as np
from imutils import paths
import cv2


def save_to_drive(aligned_face, path): 

    file_name = path.split('/')[1]
    cv2.imwrite(
        'tinder_pics_likes_deduped/' + file_name,
        aligned_face 
        )
    print ('saved: ' + file_name)
    return None


def is_duplicate(control_image, images):
	if images == []: return False	
	for comparison_image in images:
		
		#if a duplicate is found, we break out of the loop and start on the next one
		if np.array_equal(control_image, comparison_image):
			print ('DUP')
			
			return True

	return False

def dedupe(images, paths):
	for i, control_image in enumerate(images):
		if not is_duplicate(control_image, images[i + 1:]):
			save_to_drive(control_image, paths[i])

def iterate_through_folder(folder_name):
    #the entire thing has to be read into memory which is obviously not optimal
    imagePaths = list(paths.list_images(folder_name))
    images = []
    for i, image_path in enumerate(imagePaths):
        # print (i / float(len(imagePaths)))
        
        arr = cv2.imread(image_path)
        images.append(arr)
        

    return images, imagePaths

photos, imagePaths = iterate_through_folder('tinder_pics_likes')
dedupe(photos, imagePaths)