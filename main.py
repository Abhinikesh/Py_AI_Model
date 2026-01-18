import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

click_times = []
click_cooldown = 0.4

screen_w, screen_h = pyautogui.size()
print("\nHand gesture control started.")

prev_screen_x, prev_screen_y = 0, 0
smoothening = 6

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # 🔧 FIX: finger state detection
            fingers = [
                1 if hand_landmarks.landmark[tip].y <
                     hand_landmarks.landmark[tip - 2].y else 0
                for tip in [8, 12, 16, 20]
            ]

            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)

            curr_x = prev_screen_x + (screen_x - prev_screen_x) / smoothening
            curr_y = prev_screen_y + (screen_y - prev_screen_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_screen_x, prev_screen_y = curr_x, curr_y

            # scroll mode
            if sum(fingers) == 4:
                scroll_mode = True
            else:
                scroll_mode = False

            # scroll actions
            if scroll_mode:
                if index_tip.y < 0.4:
                    pyautogui.scroll(50)
                    cv2.putText(frame, "Scrolling Up",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                elif index_tip.y > 0.6:
                    pyautogui.scroll(-50)
                    cv2.putText(frame, "Scrolling Down",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

            dist = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            if dist < 0.04:
                current_time = time.time()
                click_times.append(current_time)

                if len(click_times) >= 2 and click_times[-1] - click_times[-2] < 0.5:
                    pyautogui.doubleClick()
                    cv2.putText(frame, "Double Click",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    click_times = []
                else:
                    pyautogui.click()
                    cv2.putText(frame, "Click",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                time.sleep(click_cooldown)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
