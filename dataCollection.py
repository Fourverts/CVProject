import cv2 as cv # OpenCV for computer vision tasks (reading video, manipulating images)
import mediapipe as mp # MediaPipe library for hand detection and landmark estimation
import numpy as np # NumPy for numerical operations, especially image array manipulation
import math # Math functions (used for ceiling/rounding)
import time # Time functions (used for unique file naming)

# --- Configuration Constants ---
offset=20 # Margin/padding around the detected hand bounding box (in pixels)
imgSize = 224 # The target size (width and height) for the normalized square image
folder = "Data/1" # Default folder path to save captured images (for class 1)
counter=0 # Simple counter for saved images (though not strictly used in file naming)

# --- MediaPipe and OpenCV Setup ---
cap = cv.VideoCapture(0) # Initialize video capture from the default camera (index 0)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils # Utility for drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles # Styles for landmark drawing

# Configure the Hands module
hands = mp_hands.Hands(
    static_image_mode=False, # Set to False for continuous video stream processing
    max_num_hands=1, # Only track a single hand
    min_detection_confidence=0.5 # Minimum confidence score for a hand detection
)

# --- Main Video Loop ---
while True:
    succes, img = cap.read() # Read a frame from the webcam
    if not succes:
        break # Exit loop if frame reading failed (e.g., end of video or camera disconnected)
    
    # Convert image color space for MediaPipe
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # OpenCV uses BGR, MediaPipe expects RGB
    results = hands.process(img) # Process the image to detect hand landmarks
    
    # Convert back to BGR to display using OpenCV
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        # Loop through each detected hand (max_num_hands is set to 1)
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 1. Draw Landmarks on the original image
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # 2. Prepare the white background canvas for the normalized image
            # Creates a 224x224x3 white image array (np.uint8 for 0-255 values)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Get frame dimensions
            h, w, _ = img.shape
            
            # 3. Calculate Bounding Box Coordinates
            # Extract all x and y coordinates of the 21 hand landmarks (normalized 0 to 1)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Find min/max and convert to actual pixel values (0 to w or h)
            # Apply the offset padding (offset) to the bounding box
            x_min = max(0, int(min(x_coords) * w)-offset)
            x_max = min(w, int(max(x_coords) * w)+offset)
            y_min = max(0, int(min(y_coords) * h)-offset)
            y_max = min(h, int(max(y_coords) * h)+offset)

            # 4. Crop and Normalize Hand Image
            if x_max > x_min and y_max > y_min:
                # Crop the hand region from the original image
                imgCrop = img[y_min:y_max, x_min:x_max]
                # Check if the cropped region is valid (not empty)
                if imgCrop.size == 0: continue
                # cv.imshow("ImageCrop", imgCrop) # Optional: show the cropped image

                # Get the cropped image dimensions
                h_crop, w_crop, _ = imgCrop.shape

                # Logic to resize and center (preserve aspect ratio)
                if h_crop > w_crop: # Taller hand shape
                    # Calculate scale factor to make height equal to imgSize
                    scale = imgSize / h_crop
                    # Calculate new width, ensuring it's at least 1 pixel wide
                    w_new = min(imgSize, max(1, math.ceil(w_crop * scale)))
                    # Resize the cropped image
                    imgResize = cv.resize(imgCrop, (w_new, imgSize))

                    # Calculate horizontal gap for centering
                    wGap = (imgSize - w_new) // 2
                    # Place the resized image onto the center of the white canvas
                    imgWhite[:, wGap:wGap + w_new] = imgResize

                else: # Wider or square hand shape
                    # Calculate scale factor to make width equal to imgSize
                    scale = imgSize / w_crop
                    # Calculate new height, ensuring it's at least 1 pixel high
                    h_new = min(imgSize, max(1, math.ceil(h_crop * scale)))
                    # Resize the cropped image
                    imgResize = cv.resize(imgCrop, (imgSize, h_new))

                    # Calculate vertical gap for centering
                    hGap = (imgSize - h_new) // 2
                    # Place the resized image onto the center of the white canvas
                    imgWhite[hGap:hGap + h_new, :] = imgResize

                # 5. Display the normalized image
                cv.imshow("WhiteImage", imgWhite)

    # --- Display and User Input ---
    cv.imshow("Image", img) # Display the original frame with landmarks
    key= cv.waitKey(1) # Wait for a key press (1ms delay)
    
    # Save Image Logic
    if key == ord("s"): # If key "s" is pressed, save the image
        # Check if the white image is actually a hand (std dev > 5 means variation/hand is present)
        if np.std(imgWhite) < 5:
            print("No hand detected or image is too uniform (mostly white)")
            continue
        else:
            counter += 1
            print(f"Saving Image {counter} to {folder}")
            # Save the normalized image using a unique timestamp for the filename
            cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            
    # Folder Selection (Class Labeling) Logic
    # Change the target folder path to save images for different gesture classes
    elif key == ord("1"):
        folder = "Data/1"
        print(f"Saving to Folder: {folder}")
    elif key == ord("2"):
        folder = "Data/2"
        print(f"Saving to Folder: {folder}")
    elif key == ord("3"):
        folder = "Data/3"
        print(f"Saving to Folder: {folder}")
    elif key == ord("4"):
        folder = "Data/4"
        print(f"Saving to Folder: {folder}")
    elif key == ord("5"):
        folder = "Data/5"
        print(f"Saving to Folder: {folder}")
        
    # Exit condition
    elif key == ord("q"):
        break

# --- Cleanup ---
cap.release() # Release the webcam resource
cv.destroyAllWindows() # Close all OpenCV display windows

