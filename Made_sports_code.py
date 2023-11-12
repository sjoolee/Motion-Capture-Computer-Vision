import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# curl counter
counter = 0
stage = None

#VIDEO FEED
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)as pose:
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolour image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = pose.process(image)
        result = holistic.process(image)
        resultz = hands.process(image)

        # Recolour back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Face Positions
        #mp_drawing.draw_landmarks(image, result.face_landmarks,mp_holistic.FACEMESH_CONTOURS) #FACEMESH_TESSELATION VS FACEMESH_CONTOURS

        # Hand Specific Landmarks
        if resultz.multi_hand_landmarks:
            for handLms in resultz.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x * w), int(lm.y * h)  
                    if id == 0:
                        cv2.circle(image,(cx,cy),15,(255,0,255),cv2.FILLED)
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

        # Coordinates of the Frame?
        coord = [[637,352],[904,352],[631,512],[952,512]]
        dist = 1
        
       


        # The landmarks
        try:
                landmarks = results.pose_landmarks.landmark

                # Our Angles
                def calculate_angle(a,b,c):
                        a = np.array(a) # first
                        b = np.array(b) # mid
                        c = np.array(c) # end

                        radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
                        angle = np.abs(radians*180.0/np.pi)

                        if angle >180.0:
                                angle = 360-angle
                        return angle
                
                # SPECIFIC POINTS (get left side coordinates)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                # calc angles
                angle_elbow = round(calculate_angle(shoulder,elbow,wrist),1)
                angle_shoulder = round(calculate_angle(hip,shoulder,elbow),1)

                # visualize angle
                    #for elbow
                cv2.putText(image,str(angle_elbow),
                                tuple(np.multiply(elbow,[640,480]).astype(int)), #where the specific coordinates are translated (this is for the elbow)
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA) #font, font size, font colour, line width and line
                    #for shoulder
                cv2.putText(image,str(angle_shoulder),
                                tuple(np.multiply(shoulder,[640,480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)


        # curl counter logic (may only work with golf)
        # try to use time.sleep, see if it can pause the follow through swing and then reset
                if angle_shoulder > 60: #hip should be the reference point and add another condition
                    stage = "swing"
                    counter +=1
                if angle_shoulder < 20:
                    stage = "reset"
                    print(counter)

        except:
                pass

        'start with fixed starting position with one specific condition (machine learning; motion capture)'


        # Render curl counter (AI Pose Estimation|Plus AI Gym Tracker)
        # Setup status box
        cv2.rectangle(image,(0,0),(235,73),(245,245,245),-1)

        # Data for status box
        cv2.putText(image,'Swing #:',(15,12),
                    cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(139,69,19),2,cv2.LINE_AA)

        # Stage data
        cv2.putText(image,'STATE',(90,12),
                    cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(60,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(139,69,19),2,cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                                  mp_drawing.DrawingSpec(color=(119,136,153), thickness=2, circle_radius=2),
                                                                  mp_drawing.DrawingSpec(color=(218,165,32), thickness=2, circle_radius=2))
        
        cv2.imshow('Mediapipe Feed', image)                   
        if cv2.waitKey(10) & 0xFF == ord('q'):
                   break


    cap.release()
    cv2.destroyAllWindows()




    ##To Dos
    '''the main part, is finding the speed of the swing. Maybe you can get the coordinates of the wrist at one side,
then find the coordinate of the wrist at the other side, using parameters. Make the negative space with the face, thin line (not shoulder width
but less) calc. the speed of the wrist coordinate going from one parameter to the other,
and then maybe have a position marker? So it turns red and then green, once you're positioned for the negative space? Positive space would be the arms'''


    
  
