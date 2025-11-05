# translator/consumers.py
import json
import base64
import os
import numpy as np
import cv2
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
from tensorflow.keras.models import load_model
from django.conf import settings
import time

# -------- MediaPipe setup --------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------- Lazy-load model --------
model = None
def get_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, 'translator', 'action.h5')
        model = load_model(model_path)
        model.summary()
    return model

# -------- Helper functions --------
def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -------- Define actions --------
actions = ['hello', 'thanks', 'iloveyou', 'yes', 'no', 'whitespace']
threshold = 0.8  # Lowered for live recognition

# -------- ASGI WebSocket Consumer --------
class ASLConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.last_prediction_time = time.time()

    async def disconnect(self, close_code):
        self.holistic.close()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run MediaPipe
            frame, results = mediapipe_detection(frame, self.holistic)
            draw_styled_landmarks(frame, results)

            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:
                model = get_model()
                res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                self.predictions.append(np.argmax(res))

                # Debug prints
                print("Raw predictions:", res)
                print("Predicted action:", actions[np.argmax(res)])

                if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(self.sentence) == 0 or actions[np.argmax(res)] != self.sentence[-1]:
                            if actions[np.argmax(res)] != 'whitespace':
                                self.sentence.append(actions[np.argmax(res)])
                                self.last_prediction_time = time.time()

            # Reset sentence after 5 seconds of no new prediction
            if time.time() - self.last_prediction_time > 5:
                self.sentence = []

            await self.send(text_data=json.dumps({
                'translation': ' '.join(self.sentence)
            }))
        except Exception as e:
            print("Error in ASLConsumer:", e)
            await self.send(text_data=json.dumps({
                'translation': ''
            }))
