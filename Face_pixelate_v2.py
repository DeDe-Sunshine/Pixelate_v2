import cv2
import numpy as np
import sys
import os
import random
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def face_zoom(image, filename):
    """
    locate face and crop + 1x surrounding
    Args:
        image: The input image (NumPy array).

    Returns:
        face + 1x surrounding
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))
    count = 1
    # Display and ask for selection
    if len(faces) > 1: 
        for (x, y, w, h) in faces:
            roi_color = image[(y-round(0.5*h)): y + round(1.5*h), (x-round(0.5*w)) : x + round(1.5*w)]
            window_name = 'Face '+str(count)
            fig = plt.figure()
            plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
            plt.title(window_name)
            plt.show(block=False)
            answer = input("Is this the right face? Enter Y for yes or N for no: ")
            
            if answer == "N":
                #plt.waitforbuttonpress(0)
                plt.close(fig)
                count = count+1
                continue
            elif answer == "Y": 
                #plt.waitforbuttonpress(0)
                plt.close(fig)
                return roi_color
            
        
    else: 
        roi_color = image[(y-round(0.5*h)): y + round(1.5*h), (x-round(0.5*w)) : x + round(1.5*w)]
        return roi_color

def auto_brightness_correction(image):
    """
    Automatically adjusts the brightness of an image using histogram equalization.

    Args:
        image: The input image (NumPy array).

    Returns:
        The brightness-corrected image (NumPy array).
    """

    # Convert the image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split the channels
    y, cr, cb = cv2.split(ycrcb)

    # Apply histogram equalization to the Y channel
    y_eq = cv2.equalizeHist(y)

    # Merge the channels back together
    ycrcb_eq = cv2.merge((y_eq, cr, cb))

    # Convert back to BGR color space
    corrected_image = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return corrected_image

def shuffle_pixels(image):
    """Shuffles the pixels of an image."""

    image = np.array(image)
    rows, cols, channels = image.shape

    for i in range(rows):
        for j in range(cols):
            rand_i = random.randint(0, rows - 1)
            rand_j = random.randint(0, cols - 1)

            image[i, j], image[rand_i, rand_j] = image[rand_i, rand_j], image[i, j]

    return Image.fromarray(image)

def pixelate(image, pixel_size, shuffle):
    """Pixelates an image."""

    width, height = image.size
    new_width = width // pixel_size
    new_height = height // pixel_size
    print(f'Sized to {new_width} x {new_height}')
    image = image.resize((new_width, new_height), Image.NEAREST)
    if shuffle == 1:
        image = shuffle_pixels(image)
        image = image.resize((width, height), Image.NEAREST)
    else:
        image = image.resize((width, height), Image.NEAREST)

    return image, new_width, new_height



if __name__ == "__main__":
    # Load the image 
    imagePath = filedialog.askopenfilename()
    image = cv2.imread(imagePath)
    
    # Get file name 
    file_name = os.path.basename(imagePath)
    filename_noext = file_name.split(".")
    pathname = os.path.dirname(imagePath)
    
    # Make new folder
    new_folder= "/" + filename_noext[0]
    new_directory = os.path.join(pathname, filename_noext[0])
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # move file
    shutil.move(imagePath, new_directory)

    # change working directory 
    os.chdir(new_directory)
    
    # Find face
    zoomed_face = face_zoom(image, filename_noext[0])
    
    cv2.imwrite('original_' + filename_noext[0] +'_face.jpg', zoomed_face) 

    # Correct Lighting 
    corrected_image = auto_brightness_correction(zoomed_face)
    corrected_image_path = os.path.join(new_directory, 'lighting_corrected_' + filename_noext[0] +'.jpg')
    cv2.imwrite('lighting_corrected_' + filename_noext[0] +'.jpg', corrected_image)
    
    image = Image.open(corrected_image_path)
    
    # Face image size
    width, height = image.size
    print(f'Uncompressed image is {width} x {height}')    
   
    for num in [4, 8, 16, 64]:

        # pixelate and shuffle
        pixelated_image, new_width, new_height = pixelate(image, num, shuffle=0)
        shuffled_image, _, _  = pixelate(image, num, shuffle=1)

        # save image 
        new_filename_pixelated = filename_noext[0] + '_' + str(new_width) + 'x' + str(new_height)  + '.jpg'
        new_filename_shuffled = filename_noext[0] + '_' + str(new_width) + 'x' + str(new_height)  + 'shuffled.jpg'

        pixelated_image.save(new_filename_pixelated)
        shuffled_image.save(new_filename_shuffled)