from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	# A,B and C are the distances between the two arguments
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2 * C)
	return ear

# args takes the arguments
argsParser = argparse.ArgumentParser()
# --shape-predictor argument defines the path to the face landmarks predictor
argsParser.add_argument("-p", "--shape-predictor", required=True)
# --video argument defines the path to the video input
argsParser.add_argument("-v", "--video", type=str, default="")
args = vars(argsParser.parse_args())

# EAR threshold
EAR_THRESHOLD = 0.2
# EAR consecutive frames
EAR_FRAMES = 1

# count the blink frames
blinkFrames = 0
# count the total blinks
total = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Get the left and right eyes coordinates
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Uncomment the following two lines for video
vs = FileVideoStream(args["video"]).start()
fileStream = True

# Uncomment the following two lines for camera
# vs = VideoStream(src=0).start()
# fileStream = False

# While reading the video stream
while True:
	# The break condition: video ends
	if fileStream and not vs.more():
		break
	# Read and transfer each frame into gray picture
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detect rects coordinates in gray picture
	rects = detector(gray, 0)
	for rect in rects:
		# Find all shapes
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Find the left and right eye shapes using their coordinates
		leftEye = shape[leftStart:leftEnd]
		rightEye = shape[rightStart:rightEnd]

		# Calculate left and right EARs
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Calculate the average EAR
		ear = (leftEAR + rightEAR) / 2

		# Find the left and right eye hulls to draw
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# Draw the contours with green color
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# Count the frame when ear is less than the threshold
		if ear < EAR_THRESHOLD:
			blinkFrames += 1
		else:
			# When ear is greater than the threshold, check if the eyes have closed for enough frames
			if blinkFrames >= EAR_FRAMES:
				# Add the total blinks
				total += 1
			blinkFrames = 0
		# Show the blinks total in the video with blue color
		cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		# Show the EAR values in the video with red color
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# Show each frame after the above process
	cv2.imshow("Eye Blink Detection", frame)

	# Click space key to stop video
	key = cv2.waitKey(1) & 0xFF
	if key == ord(" "):
		break

cv2.destroyAllWindows()
vs.stop()