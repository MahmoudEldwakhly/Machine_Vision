import os
import cv2
import numpy as np
import json
import time
import sys
import h5py
from collections import deque

sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, ReLU, Add,
    GlobalAveragePooling2D, Dropout, Dense, LSTM, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_PATH = r'D:\University\Senior 2\Fall 2025\Machine Vision\Major Task\Vision'

ACTION_MODEL_PATH  = os.path.join(BASE_PATH, 'action_model.h5')
ACTION_LABELS_PATH = os.path.join(BASE_PATH, 'labels.json')

EMOTION_MODEL_PATH  = os.path.join(BASE_PATH, 'best_adam_Emotions.h5')
EMOTION_LABELS_PATH = os.path.join(BASE_PATH, 'labels_Emotions.json')

# --- ACTION must match training (from h5 file: batch_shape [None, 12, 112, 112, 3]) ---
ACTION_IMG_SIZE = 112
SEQ_LEN = 20 # MUST match h5 file batch_shape
NUM_CLASSES = 4

# display gating
ACTION_THRESHOLD = 0.55
FRAME_SKIP = 0
SMOOTH_WINDOW = 2

# ==========================================
# 2. PREPROCESSING FUNCTIONS
# ==========================================
def preprocess_gray_exact(gray48):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray48)
    g = g.astype(np.float32)
    mean, std = g.mean(), g.std()
    g = (g - mean) / (std + 1e-6)
    g = np.expand_dims(g, axis=-1)
    return g

# ==========================================
# 3. BUILD AND LOAD MODELS
# ==========================================
print("Loading models... (This might take a moment)")

def build_action_model():
    """Build MobileNetV2 + LSTM model matching training architecture"""
    inp = Input(shape=(SEQ_LEN, ACTION_IMG_SIZE, ACTION_IMG_SIZE, 3))
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(ACTION_IMG_SIZE, ACTION_IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    # MobileNetV2 outputs 1280 features directly to LSTM (no Dense 256 in between)
    x = TimeDistributed(base_model, name='time_distributed_8')(inp)
    x = LSTM(128, name='lstm_4')(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='dense_4')(x)
    
    return Model(inp, out)

def load_action_weights(model, h5_path):
    """Load weights from h5 file with proper path handling"""
    print("[INFO] Loading action model weights manually...")
    
    with h5py.File(h5_path, 'r') as f:
        # Load LSTM weights
        lstm_layer = model.get_layer('lstm_4')
        lstm_path = 'model_weights/lstm_4/sequential_4/lstm_4/lstm_cell'
        
        kernel = f[f'{lstm_path}/kernel'][:]
        recurrent_kernel = f[f'{lstm_path}/recurrent_kernel'][:]
        bias = f[f'{lstm_path}/bias'][:]
        lstm_layer.set_weights([kernel, recurrent_kernel, bias])
        print("[INFO] LSTM weights loaded successfully")
        
        # Load Dense weights
        dense_layer = model.get_layer('dense_4')
        dense_path = 'model_weights/dense_4/sequential_4/dense_4'
        
        kernel = f[f'{dense_path}/kernel'][:]
        bias = f[f'{dense_path}/bias'][:]
        dense_layer.set_weights([kernel, bias])
        print("[INFO] Dense weights loaded successfully")
        
        # Load MobileNetV2 weights
        td_layer = model.get_layer('time_distributed_8')
        mobilenet = td_layer.layer
        
        h5_td_path = 'model_weights/time_distributed_8'
        loaded_count = 0
        
        for layer in mobilenet.layers:
            if not layer.weights:
                continue
            
            layer_path = f'{h5_td_path}/{layer.name}'
            if layer_path not in f:
                continue
            
            h5_layer = f[layer_path]
            weight_names = list(h5_layer.keys())
            
            new_weights = []
            for w in layer.weights:
                w_name = w.name.split('/')[-1].replace(':0', '')
                
                # Handle DepthwiseConv2D naming mismatch
                if 'depthwise_kernel' in w_name:
                    h5_name = 'kernel' if 'kernel' in weight_names else 'depthwise_kernel'
                else:
                    h5_name = w_name
                
                if h5_name in weight_names:
                    new_weights.append(h5_layer[h5_name][:])
            
            if len(new_weights) == len(layer.weights):
                layer.set_weights(new_weights)
                loaded_count += 1
        
        print(f"[INFO] Loaded {loaded_count} MobileNetV2 layers")
    
    return model

def load_emotion_model(path):
    """Build Mini-Xception emotion model"""
    weight_decay = 1e-4
    num_classes = 7

    inp = Input(shape=(48, 48, 1))

    x = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(weight_decay))(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3,3), strides=2, padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    def res_block(x, filters):
        res = Conv2D(filters, (1,1), strides=2, padding='same', kernel_regularizer=l2(weight_decay))(x)
        y = SeparableConv2D(filters, (3,3), padding='same',
                            depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = SeparableConv2D(filters, (3,3), strides=2, padding='same',
                            depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(y)
        y = BatchNormalization()(y)
        out = Add()([res, y])
        out = ReLU()(out)
        return out

    x = res_block(x, 128)
    x = res_block(x, 256)
    x = res_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inp, out)

    try:
        model.load_weights(path, by_name=True, skip_mismatch=True)
        print("[INFO] Emotion weights loaded with by_name=True")
    except Exception as e:
        print(f"[WARN] by_name load failed: {e}, trying direct...")
        model.load_weights(path)

    return model

def load_action_labels(labels_path):
    with open(labels_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda x: int(x))
        return [data[k] for k in keys]
    raise ValueError("labels.json must be list or dict")

# Load models
try:
    print("[INFO] Building Action Model...")
    action_model = build_action_model()
    
    print("[INFO] Loading Action Model Weights...")
    action_model = load_action_weights(action_model, ACTION_MODEL_PATH)
    action_labels = load_action_labels(ACTION_LABELS_PATH)
    print("[OK] Action weights loading completed")
    print("[OK] Action Model Loaded.")

    print("[INFO] Building and Loading Emotion Model...")
    emotion_model = load_emotion_model(EMOTION_MODEL_PATH)

    with open(EMOTION_LABELS_PATH, 'r') as f:
        labels_data = json.load(f)
    if isinstance(labels_data, dict):
        sorted_keys = sorted(labels_data.keys(), key=lambda x: int(x))
        emotion_labels = [labels_data[k] for k in sorted_keys]
    else:
        emotion_labels = labels_data

    print("[OK] Emotion Model Loaded.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

except Exception as e:
    print(f"\n[ERROR] Could not load models.\nMake sure the files exist in: {BASE_PATH}\nError details: {e}")
    import traceback
    traceback.print_exc()
    raise

# ==========================================
# 4. MAIN SYSTEM LOOP
# ==========================================
def run_local_system():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    skip_counter = 0
    frame_queue = deque(maxlen=SEQ_LEN)
    prob_history = deque(maxlen=SMOOTH_WINDOW)

    current_action = "Unknown"
    action_conf = 0.0

    prev_time = time.time()
    fps = 0.0

    print(f"[OK] System Started. Skip: {FRAME_SKIP}, Thresh: {ACTION_THRESHOLD}. Press 'q' to quit.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        start_loop = time.time()

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_bgr.shape

        # ---------------------------
        # ACTION RECOGNITION
        # ---------------------------
        if skip_counter % (FRAME_SKIP + 1) == 0:
            resized = cv2.resize(frame_rgb, (ACTION_IMG_SIZE, ACTION_IMG_SIZE))
            normalized = resized.astype(np.float32) / 255.0
            frame_queue.append(normalized)

            if len(frame_queue) == SEQ_LEN:
                input_batch = np.expand_dims(np.array(frame_queue, dtype=np.float32), axis=0)
                probs = action_model.predict(input_batch, verbose=0)[0]

                prob_history.append(probs)
                avg_probs = np.mean(np.array(prob_history), axis=0)

                top_idx = int(np.argmax(avg_probs))
                top_conf = float(avg_probs[top_idx])

                # Debug output
                labels_str = ", ".join([f"{action_labels[i]}={avg_probs[i]:.2f}" for i in range(len(action_labels))])
                print(f"Preds: {labels_str}")

                if top_conf >= ACTION_THRESHOLD:
                    current_action = action_labels[top_idx]
                    action_conf = top_conf
                    print(f">>> DETECTED: {current_action} ({action_conf*100:.1f}%)")
                else:
                    current_action = "Unknown"
                    action_conf = top_conf

        skip_counter += 1

        # ---------------------------
        # EMOTION RECOGNITION
        # ---------------------------
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        is_scanning = False
        if len(faces) == 0:
            is_scanning = True
            faces = [[w//2 - 110, h//2 - 110, 220, 220]]

        for (x, y, fw, fh) in faces:
            color = (255, 0, 0) if is_scanning else (255, 0, 255)
            cv2.rectangle(frame_bgr, (x, y), (x+fw, y+fh), color, 2)

            header_text = "SCANNING" if is_scanning else "Face Detected"
            cv2.putText(frame_bgr, header_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if not is_scanning:
                roi_gray = gray_frame[y:y+fh, x:x+fw]
                if roi_gray.size == 0:
                    break

                roi_processed = preprocess_gray_exact(cv2.resize(roi_gray, (48, 48)))
                emo_probs = emotion_model.predict(np.expand_dims(roi_processed, axis=0), verbose=0)[0]
                emo_idx = int(np.argmax(emo_probs))

                label_text = f"{emotion_labels[emo_idx]} {emo_probs[emo_idx]*100:.0f}%"
                cv2.putText(frame_bgr, label_text, (x, y+fh+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            break

        # ---------------------------
        # OVERLAY & METRICS
        # ---------------------------
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        latency_ms = (curr_time - start_loop) * 1000.0

        cv2.rectangle(frame_bgr, (0, 0), (360, 100), (0,0,0), -1)

        display_color = (0, 255, 0) if current_action != "Unknown" else (0, 165, 255)

        cv2.putText(frame_bgr, f"Action: {current_action}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)

        cv2.putText(frame_bgr, f"Conf:   {action_conf*100:.1f}%", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame_bgr, f"Lat: {latency_ms:.1f}ms", (130, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Real-time Analysis', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local_system()
