# USAGE
# python cluster_faces.py --encodings encodings.pickle

# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import pandas as pd
import hdbscan

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

#!!unsure how this will affect the alignment
encodings = [arr for arr in encodings if arr is not None]

# cluster the embeddings
print("[INFO] clustering...")

# clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.535, min_samples=2)
# clt.fit(encodings)

params = {'min_cluster_size': 2}
clt = hdbscan.HDBSCAN(**params)
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
print ('value_counts', pd.Series(clt.labels_).value_counts())
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
	# find all indexes into the `data` array that belong to the
	# current label ID, then randomly sample a maximum of 25 indexes
	# from the set
	print("[INFO] faces for face ID: {}".format(labelID))

	idxs = np.where(clt.labels_ == labelID)[0]
	# idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

	# initialize the list of faces to include in the montage
	faces = []

	# loop over the sampled indexes
	for i in idxs:
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		
		# try:
		# 	(top, right, bottom, left) = data[i].get("loc")
		# except: 
		# 	continue

		# face = image[top:bottom, left:right]
		face = image

		# force resize the face ROI to 96x96 and then add it to the
		# faces montage list
		face = cv2.resize(face, (96, 96))
		faces.append(face)

	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	montage = build_montages(faces, (96, 96), (10, 10))[0]
	
	# show the output montage
	print ('here')
	title = "Face ID #{}".format(labelID)
	title = "Unknown Faces" if labelID == -1 else title
	cv2.imshow(title, montage)
	cv2.waitKey(0)