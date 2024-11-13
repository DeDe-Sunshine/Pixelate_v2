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

def process_image_for_column(image, pixel_sizes, blur_levels):
    """
    Process the image by applying a combination of pixelation and blur effects.
    Args:
        image: PIL image to be processed.
        pixel_sizes: List of pixelation sizes.
        blur_levels: List of blur radii.
    Returns:
        List of base64-encoded images for display.
    """
    processed_images = []

    for pixel_size, blur_level in zip(pixel_sizes, blur_levels):
        # Apply pixelation by resizing down and then back up
        pixelated = image.resize(
            (max(1, image.width // pixel_size), max(1, image.height // pixel_size)),
            Image.NEAREST
        )
        pixelated = pixelated.resize((image.width, image.height), Image.NEAREST)

        # Apply blur if specified
        if blur_level > 0:
            blurred = pixelated.filter(ImageFilter.GaussianBlur(blur_level))
        else:
            blurred = pixelated

        # Convert processed image to base64
        buffered = BytesIO()
        blurred.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        processed_images.append(img_str)

    return processed_images

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
            image = Image.open(file).convert("RGB")

            # Define pixelation and blur levels for 4 images: from most blurred to fully clear
            pixel_sizes = [64, 32, 20, 1]  # From high pixelation to no pixelation
            blur_levels = [16, 12, 8, 0]   # From high blur to no blur

            # Generate processed images for the column
            processed_images = process_image_for_column(image, pixel_sizes, blur_levels)

            return render_template("index.html", processed_images=processed_images)

    return render_template("index.html", processed_images=None)

if __name__ == "__main__":
    app.run(debug=True)
