Steps to Work on the Project in Google Colab
Open a New Google Colab Notebook:
Go to Google Colab and click on New Notebook.
Install Dependencies: In Google Colab, you can install the necessary libraries directly from the notebook using !pip install. Add the following code to your first cell to install the required dependencies:


!pip install tensorflow

!pip install keras

!pip install opencv-python-headless  # OpenCV (headless version for Colab)

!pip install matplotlib  # Optional for visualization

Upload Pre-Trained Emotion Detection Model:
Since Colab doesn't allow direct access to files on your local machine, you need to upload the model file (emotion_model.h5) into your Colab environment. You can upload files directly in Colab with the following code:

from google.colab import files

uploaded = files.upload()

 After running the above code, click on the Choose Files button to select and upload the emotion_model.h5 file.
Access the Camera Feed in Colab:
Unfortunately, Google Colab doesn't directly support real-time access to your computer's camera. However, you can work with images uploaded to the Colab environment. To test the project, students can upload an image with a face, and the model will predict emotions based on the uploaded image.
Modified Code to Handle Image Uploads: Below is an adapted version of the real-time face and emotion detection project using uploaded images instead of live camera feed. This allows the project to be executed fully in Google Colab.

python
Copy
import cv2

import numpy as np

from tensorflow.keras.models import load_model

from keras.preprocessing.image import img_to_array

from google.colab import files

from matplotlib import pyplot as plt


# Upload the image file

uploaded = files.upload()


# Load the pre-trained Haar Cascade Classifier for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Load the emotion detection model (after uploading the model)

emotion_model = load_model('emotion_model.h5')  # The uploaded model


# Define emotion labels

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Load the uploaded image

image_path = list(uploaded.keys())[0]  # Get the file name

image = cv2.imread(image_path)


if image is None:

    print("Error: Image not found!")

else:

    # Convert the image to grayscale

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Detect faces in the grayscale image

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    # For each detected face, perform emotion recognition

    for (x, y, w, h) in faces:

        # Draw rectangle around the face

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # Extract the ROI for emotion detection

        roi_gray = gray_image[y:y + h, x:x + w]

        roi_color = image[y:y + h, x:x + w]


        # Resize the face to match the input shape of the emotion model

        roi_resized = cv2.resize(roi_gray, (48, 48))

        roi_resized = roi_resized.astype('float32') / 255

        roi_resized = img_to_array(roi_resized)

        roi_resized = np.expand_dims(roi_resized, axis=0)


        # Predict emotion

        emotion_pred = emotion_model.predict(roi_resized)

        max_index = np.argmax(emotion_pred[0])

        predicted_emotion = emotion_labels[max_index]


        # Display the predicted emotion on the image

        cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    # Display the resulting image with detected faces and emotions

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.show()

Run the Code:
Run the cell, and once the model and the image are uploaded, the code will detect faces and predict emotions on the uploaded image.
The result will be displayed as an image with rectangles around detected faces and their respective emotions.
