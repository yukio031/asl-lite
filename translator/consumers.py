# translator/consumers.py
import json
import os
import time
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from tensorflow.keras.models import load_model

# ---------- Configuration ----------
# Adjust this to the number of floats per frame you send (e.g., 21*3*2 = 126 for two hands)
FRAME_KEYPOINT_LENGTH = 21 * 3 * 2  # left + right hand (pad with zeros client-side or here)
SEQUENCE_LENGTH = 30

# ---------- Lazy-load model ----------
_model = None
def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(settings.BASE_DIR, 'translator', 'action.h5')
        _model = load_model(model_path)
        print("✅ Loaded model:", model_path)
    return _model

# ---------- Define labels ----------
actions = ['hello', 'thanks', 'iloveyou', 'yes', 'no', 'whitespace']  # replace with your model's labels
THRESHOLD = 0.8

class ASLConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.sequence = []      # sliding window of frames (each frame is an array of floats)
        self.sentence = []      # final words
        self.predictions = []   # raw predicted indices
        self.last_prediction_time = time.time()
        print("🔗 WebSocket connected")

    async def disconnect(self, close_code):
        print("🔌 WebSocket disconnected (code=%s)" % close_code)

    async def receive(self, text_data):
        """
        Expecting JSON: { "keypoints": [x1,y1,z1,...] }
        where each frame sent is flattened keypoints for left+right hand.
        We keep SEQUENCE_LENGTH frames and run predict(seq) when full.
        """
        try:
            data = json.loads(text_data)
            kp = data.get('keypoints', [])

            # Validate and convert to numpy array of fixed length
            kp_arr = np.array(kp, dtype=np.float32)
            if kp_arr.size == 0:
                # nothing to do
                await self._send_translation()
                return

            # If client sent fewer points than FRAME_KEYPOINT_LENGTH (missing hand), pad with zeros
            if kp_arr.size < FRAME_KEYPOINT_LENGTH:
                pad = np.zeros(FRAME_KEYPOINT_LENGTH - kp_arr.size, dtype=np.float32)
                kp_arr = np.concatenate([kp_arr, pad])
            elif kp_arr.size > FRAME_KEYPOINT_LENGTH:
                kp_arr = kp_arr[:FRAME_KEYPOINT_LENGTH]

            self.sequence.append(kp_arr)
            # keep only last SEQUENCE_LENGTH frames
            if len(self.sequence) > SEQUENCE_LENGTH:
                self.sequence = self.sequence[-SEQUENCE_LENGTH:]

            if len(self.sequence) == SEQUENCE_LENGTH:
                model = get_model()
                seq_np = np.expand_dims(np.array(self.sequence), axis=0)  # shape (1, SEQ, FRAME_LEN)
                res = model.predict(seq_np)[0]  # assuming model outputs probabilities for actions
                predicted_idx = int(np.argmax(res))
                self.predictions.append(predicted_idx)

                # require stability in last n predictions
                last_preds = self.predictions[-10:]
                if len(last_preds) > 0 and np.unique(last_preds).shape[0] == 1 and last_preds[-1] == predicted_idx:
                    if res[predicted_idx] > THRESHOLD:
                        predicted_action = actions[predicted_idx]
                        if len(self.sentence) == 0 or predicted_action != self.sentence[-1]:
                            if predicted_action != 'whitespace':
                                self.sentence.append(predicted_action)
                                self.last_prediction_time = time.time()

            # reset sentence after inactivity
            if time.time() - self.last_prediction_time > 5:
                self.sentence = []

            await self._send_translation()

        except Exception as e:
            print("❌ Error in ASLConsumer.receive:", e)
            await self._send_translation()

    async def _send_translation(self):
        await self.send(text_data=json.dumps({
            'translation': ' '.join(self.sentence)
        }))
