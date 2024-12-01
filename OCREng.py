import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours
import streamlit as st
from tensorflow.keras.models import load_model

# Function to preprocess the image and detect lines
def preprocess_image(image, kernel_height):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, kernel_height), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    st.image(dilated, caption='Dilated Image')
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    img_with_boxes = image.copy()
    line_list = []
    min_h, max_h = 60, 200
    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        if min_h <= h <= max_h:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (40, 100, 250), 2)
            line_list.append([x, y, x + w, y + h])
    st.image(img_with_boxes, channels="BGR", caption='Image with bounding boxes')
    return line_list

# Function to find contours
def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]
    return conts

# Function to extract ROI
def extract_roi(img, x, y, w, h, margin=2):
    roi = img[y - margin:y + h, x - margin:x + w + margin]
    return roi

# Function to perform thresholding
def thresholding(roi):
    return cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Function to resize image
def resize_img(thresh):
    return cv2.resize(thresh, (28, 28))

# Function for image normalization
def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

# Function to process each bounding box
def process_box(gray, x, y, w, h):
    roi = extract_roi(gray, x, y, w, h)
    thresh = thresholding(roi)
    resized = resize_img(thresh)
    # st.image(resized, caption='Resized Image')
    normalized = normalization(resized)
    return normalized, (x, y, w, h)

# Function to detect characters within lines
def detect_characters_within_lines(image, lines, gray_img):
    min_w, max_w = 15, 160
    min_h, max_h = 25, 140
    characters = []

    for line in lines:
        x1, y1, x2, y2 = line
        roi = gray_img[y1:y2, x1:x2]
        adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
        invertion = 255 - adaptive
        dilation = cv2.dilate(invertion, np.ones((3, 3), np.uint8))
        
        conts = find_contours(dilation)
        
        for c in conts:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
                normalized_char, box = process_box(gray_img, x1 + x, y1 + y, w, h)
                characters.append((normalized_char, box))

    return characters

# # Loading the OCR model
def load_ocr_model():
    model_path = r'C:/Users/HP/Desktop/MASTER/Projet-OCR/modelOCR.h5'
    return load_model(model_path)

# Main Streamlit app code
def main():
    st.title('OCR with English')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption='Uploaded Image')

        # Preprocess the image and detect lines
        line_list = preprocess_image(img, 200)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load OCR model
        network = load_ocr_model()

        # Processing the detections
        characters = detect_characters_within_lines(img, line_list, gray)

        # Normalize the detections and prepare for recognition
        boxes = [box[1] for box in characters]
        pixels = np.array([pixel[0] for pixel in characters], dtype='float32')

        # Recognition of characters
        digits = '0123456789'
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        characters_list = list(digits + letters)

        predictions = network.predict(pixels)

        img_copy = img.copy()
        text = ""
        previous_end_x = None
        previous_y = None

        # Calculate the average width of the characters
        average_width = np.mean([w for (x, y, w, h) in boxes])

        # Define a threshold for space based on the average width
        space_threshold = average_width * 0.7

        # Define a threshold for detecting new lines
        line_threshold = average_width * 2

        for (prediction, (x, y, w, h)) in zip(predictions, boxes):
            i = np.argmax(prediction)
            probability = prediction[i]
            character = characters_list[i]

            if previous_end_x is not None:
                # Calculate the distance between the current character's start x and the previous character's end x
                distance = x - previous_end_x

                # Add a space if the distance exceeds the space threshold
                if distance > space_threshold:
                    text += " "

                # Add a newline if the current character is in a new line
                if abs(y - previous_y) > line_threshold:
                    text += "\n"

            text += character
            previous_end_x = x + w  # Update to the end x of the current character's bounding box
            previous_y = y  # Update to the y of the current character's bounding box

            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

        st.image(img_copy, channels="BGR", caption='Processed Image with OCR')

        st.write("Detected Text:")
        st.text_area("Extracted Text", text, height=300)

if __name__ == '__main__':
    main()
