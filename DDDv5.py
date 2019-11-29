from scipy.spatial import distance as dist
from threading import Thread,Timer
from TransportCommunication.src.python import  ethClient 
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import WebcamVideoStream
import dlib
import cv2

#calculating eye aspect ratio
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	X   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	Y1  = dist.euclidean(mou[2], mou[10])
	Y2  = dist.euclidean(mou[4], mou[8])
	# taking average
	Y   = (Y1+Y2)/2.0
	# compute mouth aspect ratio
	mar = Y/X
	return mar

def send_alert():
	print("Hello")
	# message = ethClient.MessageForController('DDD', 'small', '1')
	# message.setData("1")
	# ethClient.sendMessage(message)


writer = None
index = 0

def save_video(): 
    global index,writer
    index += 1
    writer = None
    Timer(30,save_video).start()


video = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat'
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10 
MOU_AR_THRESH = 0.5 
COUNTER = 0
yawnStatus = False
yawns = 0
pulse = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# ethClient.newClient("192.168.43.175")
# loop over captuing video
save_video()
while True:
	ret, frame = video.read()
	frame = imutils.resize(frame, width=800)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("high{}.avi".format(index), fourcc,10,(frame.shape[1], frame.shape[0]), True)
	else:
		writer.write(frame)     
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prev_yawn_status = yawnStatus
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouEAR = mouth_aspect_ratio(mouth)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0


		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0,255, 255), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter

		if ear < EYE_AR_THRESH:
			COUNTER += 1
			pulse+=1
			cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# if the eyes were closed for a sufficient number of
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				pulse += 1
				if pulse >= 30: 
					pulse = 0
					Thread(target = send_alert,args=()).start()

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (680, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# yawning detections

		if mouEAR > MOU_AR_THRESH:
			cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			yawnStatus = True 
			# cv2.putText(frame, "Drowsiness ALERT", (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
		else:
			yawnStatus = False

		if prev_yawn_status == True and yawnStatus == False:
			yawns+=1

		cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (680, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# cv2.putText(frame,"BTP",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

video.stop()




# do a bit of cleanup
cv2.destroyAllWindows()



  
  # check if the video writer is None
  