import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(0)  # Use default camera (index 0)

if not cap.isOpened():
    raise IOError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    imgchar = pytesseract.image_to_string(frame)
    print(imgchar)  # Print the recognized text for debugging
    imgboxes = pytesseract.image_to_boxes(frame)
    
    for boxes in imgboxes.splitlines():
        boxes = boxes.split(' ')
        x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 1)
    
    cv2.putText(frame, imgchar, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('Character Extraction from Visuals', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('e'):
        print(imgchar)
        print("\n\nDEVELOPER @ Ajee")

cap.release()
cv2.destroyAllWindows()



