# translator/consumers.py
import json, base64
import numpy as np
import cv2
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic

# Load model once
from django.conf import settings
import os

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
actions = np.array(['first', 'good', 'goodbye', 'hello', 'I\'m finished', 'it was delicious', 'me', 'morning', 'shower', 'whitespace'])

def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )



model = load_model(os.path.join(settings.BASE_DIR, 'translator', 'action.h5'))
model.summary()

sequence = []
sentence = []
predictions = []
threshold = 0.8
numm = 1
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

class ASLConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    async def disconnect(self, close_code):
        self.holistic.close()

    async def receive(self, text_data):
        data = json.loads(text_data)
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        os.makedirs('output_folder', exist_ok=True)
        global numm


        numm = numm + 1
        image_rgb, results = mediapipe_detection(frame, self.holistic)
        print(results)

        draw_styled_landmarks(image_rgb, results)

        keypoints = extract_keypoints(results)
        global sequence, sentence, predictions
        sequence.append(keypoints)
        sequence = sequence[-30:]

        print(len(sequence))
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            if (actions[np.argmax(res)]!='whitespace'):
                                sentence.append(actions[np.argmax(res)])
                    else:
                        if (actions[np.argmax(res)] != 'whitespace'):
                            sentence.append(actions[np.argmax(res)])


        await self.send(text_data=json.dumps({
            'translation': ' '.join(sentence)
        }))
