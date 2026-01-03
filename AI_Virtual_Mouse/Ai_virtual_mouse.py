import cv2
import mediapipe as mp
import pyautogui
import time
import math
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
click_times = []
freeze_cursor = False
screen_w, screen_h = pyautogui.size()

print("Hand mouse control started") 

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    frame = cv2.flip(frame, 1)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) 

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Distance between thumb and index finger
            dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

            if dist < 0.06:  
                if not freeze_cursor:
                    freeze_cursor = True
                    click_times.append(time.time())

                    # Double click
                    if len(click_times) >= 2 and click_times[-1] - click_times[-2] < 0.4:
                        pyautogui.doubleClick()
                        cv2.putText(frame, "Double Click", (10,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        click_times = []
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "Single Click", (10,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            else:
                if freeze_cursor:
                    time.sleep(0.1)
                freeze_cursor = False

            # Move cursor by index finger
            if not freeze_cursor:
                screen_x = int(index_tip.x * screen_w)
                screen_y = int(index_tip.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                prev_screen_x, prev_screen_y = screen_x, screen_y

            fingers = []  
            if sum(fingers) == 4:
                scroll_mode = True
            else:
                scroll_mode = False

            # Scroll actions
            if scroll_mode:
                if index_tip.y < 0.4:
                    pyautogui.scroll(60)
                    cv2.putText(frame, "Scroll up", (10,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif index_tip.y > 0.6:
                    pyautogui.scroll(-60)
                    cv2.putText(frame, "Scroll down", (10,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Live Video", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
