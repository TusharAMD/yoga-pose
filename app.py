from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from media import trainerDict
from Treepose import trainerDict_tpose

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    # with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    #     cap = cv2.VideoCapture(1)
    #     while(True):
    #         ret, image = cap.read()

    #         if ret ==  True:
    #             image_height, image_width, _ = image.shape
    #             # Convert the BGR image to RGB before processing.
    #             results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #             h,w,c=image.shape
    #             if(results.pose_landmarks):
    #                 idp=0
    #                 for landmark in results.pose_landmarks.landmark:
    #                     cx,cy=int(landmark.x*w),int(landmark.y*h)

    #                 if idp==24:
    #                     if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
    #                         cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
    #                         print("Hips Correct")
    #                     else:
    #                         cv2.putText(image,"Correct your Hips",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
    #                 elif idp==12:
    #                     if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
    #                         cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
    #                         print("Shoulders Correct")
    #                     else:
    #                         cv2.putText(image,"Correct your Shoulders",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
    #                 elif idp==14:
    #                     if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
    #                         cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
    #                         print("Elbow Correct")
    #                     else:
    #                         cv2.putText(image,"Correct your Elbow",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
    #                 #print(idp,landmark.x,landmark.y,cx,cy)
    #                 #trainerDict[idp]=[landmark.x,landmark.y]
    #                 idp=idp+1

    #                 #print(trainerDict)


    #                 # file1 = open("Trainer.txt","w")
    #                 # file1.write(str(trainerDict))
    #                 # file1.close()
    #             #dicti=dict(results.pose_landmarks)
    #             #print(dicti)
                
    #             # Draw pose landmarks on the image.
    #             annotated_image = image.copy()
    #             # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
    #             # upper_body_only is set to True.
    #             mp_drawing.draw_landmarks(
    #                 annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 255, 255)))
                
    #             #cv2.imwrite("Frame4.png",annotated_image)
    #             #cv2.imshow("Frame",annotated_image)
    #             #cv2.waitKey(1)
    #             frame = annotated_image.tobytes()
    #             yield (b'--frame\r\n'
    #                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    
    while True:
        # Capture frame-by-frame
        success, image = camera.read()  # read the camera frame
        if not success:
            break
        else:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:       
                image_height, image_width, _ = image.shape
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                h,w,c=image.shape
                if(results.pose_landmarks):
                    
                    #cv2.putText(image,"detecting...",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                    #print("Detecting...")
                    idp=0
                    for landmark in results.pose_landmarks.landmark:
                        cx,cy=int(landmark.x*w),int(landmark.y*h)
                        print(idp)
                        if idp==24:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Hips Correct")
                            else:
                                cv2.putText(image,"Correct your Hips",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                        elif idp==12:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Shoulders Correct")
                            else:
                                cv2.putText(image,"Correct your Shoulders",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                        elif idp==14:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Elbow Correct")
                            else:
                                cv2.putText(image,"Correct your Elbow",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                    #print(idp,landmark.x,landmark.y,cx,cy)
                    #trainerDict[idp]=[landmark.x,landmark.y]
                        idp=idp+1

                    #print(trainerDict)


                    # file1 = open("Trainer.txt","w")
                    # file1.write(str(trainerDict))
                    # file1.close()
                #dicti=dict(results.pose_landmarks)
                #print(dicti)
                
                # Draw pose landmarks on the image.
                annotated_image = image.copy()
                # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
                # upper_body_only is set to True.
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 255, 255)))
                
                #cv2.imwrite("Frame4.png",annotated_image)
                #cv2.imshow("Frame",annotated_image)
                ret, jpeg = cv2.imencode('.jpg', annotated_image)
                frame=jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/plank')
def index():
    """Video streaming home page."""
    return render_template('plank.html')


######################
#Tree Pose
######################



def gen_frames_tpose(): 
    while True:
        success, image = camera.read() 
        if not success:
            break
        else:
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:       
                image_height, image_width, _ = image.shape
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                h,w,c=image.shape
                if(results.pose_landmarks):
                    idp=0
                    for landmark in results.pose_landmarks.landmark:
                        cx,cy=int(landmark.x*w),int(landmark.y*h)
                        print(idp)
                        if idp==24:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Hips Correct")
                            else:
                                cv2.putText(image,"Correct your Hips",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                        elif idp==12:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Shoulders Correct")
                            else:
                                cv2.putText(image,"Correct your Shoulders",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
                        elif idp==14:
                            if abs(trainerDict[idp][0]-landmark.x)<0.1 and abs(trainerDict[idp][1]-landmark.y)<0.1:
                                cv2.circle(image,(cx,cy),25,(255,0,255),cv2.FILLED)
                                print("Elbow Correct")
                            else:
                                cv2.putText(image,"Correct your Elbow",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)

                        idp=idp+1
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 255, 255)))
                ret, jpeg = cv2.imencode('.jpg', annotated_image)
                frame=jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed_tpose')
def video_feed_tpose():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_tpose(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tpose')
def tpose():
    """Video streaming home page."""
    return render_template('tpose.html')


if __name__ == '__main__':
    app.run(debug=True)