import cv2
from mediapipe.python.solutions import hands, drawing_utils

cap = cv2.VideoCapture(0)

mp_hands = hands.Hands()
mp_drawing = drawing_utils

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                hands.HAND_CONNECTIONS
            )

    cv2.imshow("Handtracker", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
