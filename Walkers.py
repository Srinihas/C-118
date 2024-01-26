import cv2
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('walking.avi')

while True:
    ret, frame = cap.read()
    bodies = body_classifier.detectMultiScale(frame, 1.2, 3)
    for x, y, w, h in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow("String", frame)
    if cv2.waitKey(1) == 32:
        break
cap.release()
cv2.destroyAllWindows()