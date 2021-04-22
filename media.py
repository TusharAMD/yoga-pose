import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

""" def returnfunc(trainerDict):
  return trainerDict """
# For static images:
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
  #for idx, file in enumerate(file_list):
  image = cv2.imread("3.jpg")
  image_height, image_width, _ = image.shape
  # Convert the BGR image to RGB before processing.
  results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  h,w,c=image.shape
  
  trainerDict=dict()
  #print(results.pose_landmarks)
  if(results.pose_landmarks):
    idp=0
    for landmark in results.pose_landmarks.landmark:
      cx,cy=int(landmark.x*w),int(landmark.y*h)
      #print(idp,landmark.x,landmark.y,cx,cy)
      trainerDict[idp]=[landmark.x,landmark.y]
      idp=idp+1
    #print(trainerDict)
    file1 = open("Trainer.txt","w")
    file1.write(str(trainerDict))
    file1.close()
  #dicti=dict(results.pose_landmarks)
  #print(dicti)
  #returnfunc(trainerDict)
  # Draw pose landmarks on the image.
  annotated_image = image.copy()
  # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
  # upper_body_only is set to True.
  mp_drawing.draw_landmarks(
      annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 255, 255)))
  
  cv2.imwrite("Frame1.png",annotated_image)
#cv2.imshow("Frame",annotated_image)
#cv2.waitKey(1)

'''
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
'''