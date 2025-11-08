import cv2
import time

vs = cv2.VideoCapture(1)   # change to 1 if using external webcam
time.sleep(2)

folder = input("Enter folder name: ")
i = 1

while True:
    ret, frame = vs.read()
    if not ret:
        break

    cv2.imshow("Capture Dataset - Press q to save", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        filename = f"./Dataset/{folder}/{folder}_{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        i += 1
        time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('x'):  # press X to exit
        break

vs.release()
cv2.destroyAllWindows()
