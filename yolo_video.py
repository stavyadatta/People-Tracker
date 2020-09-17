# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import dlib
from centroid import CentroidTracker
from trackableObject import TrackableObject

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output",
	help="path to output video")
ap.add_argument("-y", "--yolo",
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])


print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
totalFrames = 0
trackers = []
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}


try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("{} total frames in video".format(total))


except:
	print("could not determine # of frames in video")
	print("no approx. completion time can be provided")
	total = -1
integer = 0
while True:
	integer+=1
	(grabbed, frame) = vs.read()
	rects = []

	
	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if totalFrames % 30 == 0:  
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)   
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		boxes = []
		confidences = []
		classIDs = []
		idNums = []
		idNum = -1

		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				if LABELS[classID] == "person":
					confidence = scores[classID]
					if confidence > args["confidence"]:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						

						tracker = dlib.correlation_tracker()
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						xEnd = int(centerX + (width/2))
						yEnd = int(centerY + (height/2))
						rect = dlib.rectangle(x, y, xEnd, yEnd)
						#print(x, y, xEnd, yEnd, "coordineates for yolo ")
						#print(rect, "Rectangle printed")
						tracker.start_track(rgb, rect)
						print(integer, "tracker")
						idNum+=1
						idNums.append(idNum)
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						trackers.append(tracker)
	else:	
		print("not working is it?")
		for tracker in trackers:
			print("Inside the tracker")
			tracker.update(rgb)
			pos = tracker.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY)) 
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	#rects = cv2.dnn.NMSBoxes(rects, confidences, args["confidence"], args["threshold"])
	objects = ct.update(rects)
	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)
		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# if len(idxs) > 0:
	# 	for i in idxs.flatten():
	# 		(x, y) = (boxes[i][0], boxes[i][1])
	# 		(w, h) = (boxes[i][2], boxes[i][3])

	# 		color = [int(c) for c in COLORS[classIDs[i]]]
	# 		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	# 		text = "{}: {:.4f}".format(idNums[i],
	# 			confidences[i])
			# cv2.putText(frame, text, (x, y - 5),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/testCode1.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("single frame took {:.4f} seconds".format(elap))
			print("estimated total time to finish: {:.4f}".format(
				elap * total))

	writer.write(frame)
	totalFrames +=1

print("cleaning up...")
writer.release()
vs.release()