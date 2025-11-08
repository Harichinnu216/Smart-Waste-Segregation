import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import serial_rx_tx

model = load_model("waste_cnn_model.h5")
img_size = 64

labels_dict = {
    0: "PLASTIC WASTE",
    1: "ELECTRONIC WASTE",
    2: "PAPER WASTE"
}

serialPort = serial_rx_tx.SerialPort()

def OpenCommand():
    serialPort.Open("COM4", "9600")   # Change COM port if needed

def SendDataCommand(cmd):
    if serialPort.IsOpen():
        serialPort.Send(str(cmd))

OpenCommand()
cap = cv2.VideoCapture(1)
time.sleep(2)

print("Press Q to classify, X to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (img_size, img_size))
    img = img.reshape(1, img_size, img_size, 1) / 255.0

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    label = labels_dict[class_id]
    conf = prediction[0][class_id] * 100

    cv2.putText(frame, f"{label} ({conf:.2f}%)", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Smart Waste CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detected:", label)

        if class_id == 0: SendDataCommand("1")  # Plastic
        if class_id == 1: SendDataCommand("2")  # Electronic
        if class_id == 2: SendDataCommand("3")  # Bottle

        time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
