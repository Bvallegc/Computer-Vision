# Vgg16_face_classifier. Used Vggg16 to extract features for the dataset and videos and predicted the features vectors similarity using cosine similarity.
# Roi Coordinates is meant to work with the labels taken from your video.
import cv2
import csv
import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import image_utils
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the model
base_models = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_models.input, outputs=base_models.get_layer('block5_pool').output)

def preprocess_input_image(img):
    img = image_utils.load_img(img)
    size = (224, 224) # Input size required for vgg16
    img = (image_utils.smart_resize(img, size))
    print(img.shape)
    img = image_utils.img_to_array(img)  
    img = img.copy() # add Copy() to ensure it takes the img.
    img = np.expand_dims(img, axis=0)  # Add batch dimension.
    img = preprocess_input(img)
    return img

def preprocess_input_frame(img):
    size = (224, 224)
    img = (image_utils.smart_resize(img, size))
    print(img.shape)
    frame = image_utils.img_to_array(img)
    frame = frame.copy()
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Get vectors
def get_feature_vector(img):
    feature_vector = model.predict(img)
    return feature_vector

# Compare vectors and make predictions from this similarity score
def compare_embeddings(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

dataset_path = 'Path_to_photos_data'  # Directory containing subdirectories for each class
video_path = 'Path_to_video'

labels = sorted(os.listdir(dataset_path))

dataset_features = {}
for label in os.listdir(dataset_path):
    class_dir_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(class_dir_path):
        img_path = os.path.join(class_dir_path, img_name)
        img_path = preprocess_input_image(img_path)
        features = model.predict(img_path)
        dataset_features[img_name] = (label, features)
        # Dictionary with labels and features for the image set.

# Here we get the coordinates of the bounding boxes by looping through roi.csv containing all bounding boxes for each frame.
roi_coordinates = []
with open('roi.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header row
    for row in csvreader:
        index_frame = (row[0], row[2:]) #row[`0`] is the index in this case.
        roi_coordinates.append(list(map(int, row[2:])))
        #print(row)

index = 1 
video_frames = []
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()
while cap.isOpened():
    for roi in roi_coordinates:
        y1, x1, y2, x2 = roi
        ret, frame = cap.read()
        cv2.rectangle(frame, (x1, y1), (x2, y2),  (0, 255, 0), 2) # Create bounding boxes from data frames.
        index += 1
        print(index)
        if not ret or index == 5224:
            cap.release()
            break
        frame = image_utils.array_to_img(frame)
        frame = preprocess_input_frame(frame)
        frame_features = get_feature_vector(frame)
        video_frames.append(frame_features) # Frames processed and features for each frame extracted.

        frame = cv2.cvtColor(frame[0], cv2.COLOR_RGB2BGR)
        cv2.imshow('video:', frame)
        cv2.waitKey(1)
cv2.destroyAllWindows()

matches = []
# Compute best similarity_scores for each frame in the video.
for frame in video_frames:
    best_match = None
    best_similarity = 0.0
    for img_path, features in dataset_features.items():
        #print(features[1])
        similarity = compare_embeddings(np.array(frame.flatten()), np.array(features[1]).flatten()) #Flatten arrays to fit into compare_embeddings.
        print(similarity)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = features[0]
    matches.append(best_match)
    print(matches)


# Print the matches with corresponding labels
for i, match in enumerate(matches):
    if match is not None:
        print(f"Frame {i+1}: Label: {match}")
    else:
        print(f"Frame {i+1}: No match found")
