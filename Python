from ultralytics import YOLO
import cv2
import cvzone
import math
import serial
import time

arduino = serial.Serial('COM5', 9600)
time.sleep(2)  # wait for Arduino to reset

model = YOLO("../resource/v1.3.3.2.pt")
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)



classNames = ['alu-foil', 'apple', 'banana', 'battery', 'bone', 'cans', 'corrections-fluids', 'electronics-device battery', 'foam', 'milk-carton', 'orange', 'phone-ipad', 'plastic-bottle', 'plastic-box', 'plastic-cup']


lastCommand = ''

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detected = False
    command = 'x'  # default to 'no detection'

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.59:
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                detected = True

                if currentClass in ['phone-ipad', 'battery', 'electronics-device battery', 'corrections-fluids']:
                    myColor = (0, 0, 255)
                    cvzone.putTextRect(img, f' dangerous {conf}',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=2, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    command = 'a'
                elif currentClass in ['plastic-bottle', 'cans', 'plastic-box', 'plastic-cup']:
                    myColor = (0, 255, 255)
                    cvzone.putTextRect(img, f' recycle {conf}',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=2, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    command = 'b'
                elif currentClass in ['apple', 'banana', 'orange', 'bone']:
                    myColor = (0, 128, 0)
                    cvzone.putTextRect(img, f' food waste {conf}',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=2, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    command = 'c'
                elif currentClass in ['alu-foil', 'foam']:
                    myColor = (255, 0, 0)
                    cvzone.putTextRect(img, f' general {conf}',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=2, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    command = 'd'

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 4)

    # Send command only if it changed
    if command != lastCommand:
        arduino.write((command + '\n').encode('utf-8'))
        lastCommand = command

    cv2.imshow("detector", img)
    cv2.waitKey(1)
