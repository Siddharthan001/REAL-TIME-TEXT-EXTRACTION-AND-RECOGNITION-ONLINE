from flask import Flask, render_template, request, redirect, url_for, jsonify
import pytesseract
import cv2
import os
import threading
import time

# Initialize Flask app
app = Flask(__name__)

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Ensure directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# Global variables
detected_text = ""
detected_text_lock = threading.Lock()
camera_running = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    """Extract text from an uploaded image."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    image_path = os.path.join('static/uploads', file.filename)
    file.save(image_path)

    # Preprocess and extract text from the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imgchar = pytesseract.image_to_string(binary)

    # Draw bounding boxes on detected text
    imgboxes = pytesseract.image_to_boxes(binary)
    h, w, _ = img.shape
    for box in imgboxes.splitlines():
        box = box.split(' ')
        x, y, w_box, h_box = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        cv2.rectangle(img, (x, h - y), (w_box, h - h_box), (0, 255, 0), 2)

    # Save the output image
    output_image_path = os.path.join('static/outputs', 'output_' + file.filename)
    cv2.imwrite(output_image_path, img)

    return render_template('result.html', text=imgchar, output_image=output_image_path)

@app.route('/extract_video', methods=['POST'])
def extract_text_from_video():
    """Extract text from an uploaded video."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded video
    video_path = os.path.join('static/uploads', file.filename)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Cannot open video", 400

    frames_text = []
    cntr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cntr += 1
        if cntr % 8 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            imgchar = pytesseract.image_to_string(binary)
            frames_text.append(imgchar)

    cap.release()

    return render_template('result_video.html', text="\n".join(frames_text) or "No text detected in the video.")

def capture_text_from_camera():
    """Continuously capture text from the webcam."""
    global detected_text, camera_running
    camera_running = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        camera_running = False
        return

    try:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            imgchar = pytesseract.image_to_string(binary)

            # Update detected text safely
            with detected_text_lock:
                detected_text = imgchar.strip()

            imgboxes = pytesseract.image_to_boxes(binary)
            h, w, _ = frame.shape
            for box in imgboxes.splitlines():
                box = box.split(' ')
                x, y, w_box, h_box = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                cv2.rectangle(frame, (x, h - y), (w_box, h - h_box), (0, 0, 255), 1)

            cv2.imshow('Camera Feed', frame)

            # Stop camera if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        camera_running = False

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera thread."""
    if not camera_running:
        threading.Thread(target=capture_text_from_camera, daemon=True).start()
    return redirect(url_for('camera_page'))

@app.route('/camera', methods=['GET'])
def camera_page():
    """Render the real-time camera feed page."""
    return render_template('camera.html')

@app.route('/get_detected_text', methods=['GET'])
def get_detected_text():
    """Provide detected text as a JSON response."""
    with detected_text_lock:
        text_to_return = detected_text
    return jsonify({'text': text_to_return})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera feed."""
    global camera_running
    camera_running = False
    return redirect(url_for('camera_page'))

if __name__ == '__main__':
    app.run(debug=True)
