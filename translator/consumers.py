# translator/consumers.py
import json
import base64
import numpy as np
import cv2
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
from tensorflow.keras.models import load_model
from django.conf import settings
import os
import asyncio
from google import genai

# Set up Mediapipe holistic model for ASL detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(['first', 'good', 'goodbye', 'hello', 'I\'m finished', 'it was delicious', 'me', 'morning', 'shower', 'whitespace'])

# Load the pre-trained ASL recognition model
model = load_model(os.path.join(settings.BASE_DIR, 'translator', 'action.h5'))

# Utility function to process images and extract landmarks
def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)  # Flip the image horizontally
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    results = model.process(image)  # Get detection results
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

# Function to extract keypoints from detected landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

class ASLConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        await self.accept()

    async def disconnect(self, close_code):
        self.holistic.close()

    async def receive(self, text_data):
        data = json.loads(text_data)
        msg_type = data.get("type")
        payload = data.get("data")
        client = genai.Client(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key
        
        # Sentence forming and translation
        global sentence
        sentence = sentence or []

        if payload == "english":
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents="Translate these ASL words into English: " + " ".join(sentence)
            )
            translated_text = response.text  # Extract the translated text
            await self.send(text_data=json.dumps({
                "type": "translated",
                "text": translated_text
            }))

        elif payload == "tagalog":
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents="Translate these ASL words into Tagalog: " + " ".join(sentence)
            )
            translated_text = response.text
            await self.send(text_data=json.dumps({
                "type": "translated",
                "text": translated_text
            }))

        elif payload == "X":  # Backspace action
            if sentence:
                sentence.pop()
            await self.send(text_data=json.dumps({
                "type": "translation_update",
                "translation": " ".join(sentence)
            }))

        elif msg_type == "image":
            # Handle image data (ASL detection)
            image_data = base64.b64decode(data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process image and detect ASL
            image_rgb, results = mediapipe_detection(frame, self.holistic)
            keypoints = extract_keypoints(results)

            global sequence, predictions
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prediction = actions[np.argmax(res)]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > 0.8:
                    if actions[np.argmax(res)] != 'whitespace':
                        sentence.append(actions[np.argmax(res)])

            await self.send(text_data=json.dumps({
                'translation': ' '.join(sentence)
            }))
