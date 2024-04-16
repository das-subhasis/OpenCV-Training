import cv2
import mediapipe as mp
import numpy
import math

# initialize Mediapipe hands
mp_hands = mp.solutions.hands
# store the Hands function
hands = mp_hands.Hands()

# used to outline hand
Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    # convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    Process = hands.process(frame_rgb)

    # create landmarks
    landmark_list = []

    count = {"RIGHT":0,"LEFT":0}

    # If multiple hands are present
    if Process.multi_hand_landmarks:
        # detect the hands
        for handlm in Process.multi_hand_landmarks:
            Draw.draw_landmarks(frame, handlm,
                                mp_hands.HAND_CONNECTIONS,
                                Draw.DrawingSpec(
                                    color=(0, 0, 255), thickness=2, circle_radius=2),
                                Draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Storing statuses for each finger
        finger_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

        # Storing indices for each finger tips
        finger_tips_inds = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

        
    
        for hand_ind, hand_info in enumerate(Process.multi_handedness):

            hand_label = hand_info.classification[0].label

            hand_landmarks = Process.multi_hand_landmarks[hand_ind]

            for tip_index in finger_tips_inds:
                
                finger_name = tip_index.name.split("_")[0]

                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y ):
                    finger_statuses[hand_label+"_"+finger_name] = True
                    count[hand_label.upper()]+=1

            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
    
            # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
            if ( hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or ( hand_label =='Left' and (thumb_tip_x > thumb_mcp_x) ):
                
                # Update the status of the thumb in the dictionary to true.
                finger_statuses[hand_label.upper()+"_THUMB"] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

    cv2.putText(frame, " Total RIGHT Fingers: %d"%count['RIGHT'], (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
    cv2.putText(frame, " Total LEFT Fingers: %d"%count['LEFT'], (10, 55),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
