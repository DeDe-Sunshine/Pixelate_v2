from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
import base64

app = Flask(__name__)

def face_zoom(image):
    """
    Locate face and crop with additional surrounding area.
    Args:
        image: The input image (NumPy array).
    Returns:
        Cropped face with surrounding area, or the original image if no face is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load frontal face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no face is detected, return the original image
    if len(faces) == 0:
        return image

    # Use the first detected face
    x, y, w, h = faces[0]
    y1 = max(0, y - round(0.5 * h))
    y2 = min(image.shape[0], y + round(1.5 * h))
    x1 = max(0, x - round(0.5 * w))
    x2 = min(image.shape[1], x + round(1.5 * w))
    return image[y1:y2, x1:x2]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Load the image and convert to RGB
            #image = Image.open(file).convert("RGB")
            image = cv2.imread(file)

            # Define pixelation and blur levels for 4 images: from most blurred to fully clear
            pixel_sizes = [64, 32, 20, 1]  # From high pixelation to no pixelation
            blur_levels = [16, 12, 8, 0]   # From high blur to no blur

            # zoom and find
            #zoomed_image = facee_zoom(image)
            # Generate processed images for the column
            processed_images = face_zoom(image)
            _, buffer = cv2.imencode('.jpg', processed_images)

            # Convert to Base64
            base64_image = base64.b64encode(buffer).decode('utf-8')



            return render_template("index.html", processed_images=base64_image)

    return render_template("index.html", processed_images=None)

if __name__ == "__main__":
    app.run(debug=True)
