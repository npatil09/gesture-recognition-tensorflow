#!/usr/bin/env python3
"""
Hand Gesture Recognition Project

I built this project to explore computer vision and machine learning
using MediaPipe, OpenCV, and TensorFlow. The system can collect hand
gesture samples, train a model on the collected data, and recognize
gestures in real time using a webcam.

Commands:

Collect data:
    python gesture_recognition_system.py --mode collect --gesture peace

Train model:
    python gesture_recognition_system.py --mode train

Run detection:
    python gesture_recognition_system.py --mode detect
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os
import argparse
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns




CONFIG = {
    'data_dir': 'gesture_data',
    'model_path': 'models/gesture_model.h5',
    'scaler_path': 'models/gesture_scaler.pkl',
    'label_map_path': 'models/label_map.pkl',
    'gestures': ['peace', 'ok', 'thumbs_up', 'palm', 'fist'],
    'samples_per_gesture': 200,
    'test_size': 0.2,
    'epochs': 50,
    'batch_size': 32,
}


# 1 :data collection

class DataCollector:
   
    
    def __init__(self, data_dir=CONFIG['data_dir']):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_gesture(self, gesture_name, num_samples=200):
        
        gesture_dir = self.data_dir / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        count = 0
        print(f"\n📸 Collecting {gesture_name}")
        print("Controls: SPACE = capture, Q = quit")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw UI
            cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {count}/{num_samples}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=capture | Q=quit", (20, 480-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Progress bar
            bar_width = 300
            filled = int(bar_width * count / num_samples)
            cv2.rectangle(frame, (20, 140), (20 + bar_width, 160), (0, 0, 255), 2)
            cv2.rectangle(frame, (20, 140), (20 + filled, 160), (0, 255, 0), -1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                filename = gesture_dir / f"{gesture_name}_{count}.jpg"
                cv2.imwrite(str(filename), frame)
                count += 1
                print(f"✓ Saved {count}/{num_samples}")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Completed {gesture_name}: {count} samples")



#  2:feature extraction used mediaPipe

class LandmarkExtractor:
   
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """
        Extracts hand landmarks from image
        
        Returns:
            landmarks: Array of shape (21, 3) or None
            confidence: Detection confidence
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
            return landmarks, 1.0
        
        return None, 0.0
    
    @staticmethod
    def normalize_landmarks(landmarks):
        """Normalize landmarks to be scale and position invariant"""
        if landmarks is None:
            return None
        
        # Center
        center = landmarks.mean(axis=0)
        normalized = landmarks - center
        
        # Scale
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        if max_dist > 0:
            normalized = normalized / max_dist
        
        return normalized.flatten()
    
    def process_dataset(self, data_dir):
        """Process all gesture images"""
        X, y = [], []
        label_map = {}
        
        gestures = sorted([d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))])
        
        for gesture_idx, gesture in enumerate(gestures):
            label_map[gesture_idx] = gesture
            gesture_path = os.path.join(data_dir, gesture)
            
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    landmarks, conf = self.extract_landmarks(image)
                    
                    if landmarks is not None and conf > 0.6:
                        normalized = self.normalize_landmarks(landmarks)
                        if normalized is not None:
                            X.append(normalized)
                            y.append(gesture_idx)
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return np.array(X), np.array(y), label_map



# 3:model trainning


class GestureModel:
   
    
    def __init__(self, num_classes, input_shape=63):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build(self):
        """Build neural network architecture"""
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\n🧠 Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X, y):
        """Evaluate model"""
        X_scaled = self.scaler.transform(X)
        loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
        return loss, accuracy
    
    def predict(self, features):
        """Predict on single sample"""
        if features is None:
            return None, 0.0
        
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        predictions = self.model.predict(X_scaled, verbose=0)
        
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx]
        
        return pred_idx, confidence
    
    def save(self, model_path, scaler_path):
        """Save model and scaler"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
    
    @staticmethod
    def load(model_path, scaler_path):
        """Load model and scaler"""
        model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        gesture_model = GestureModel(model.output_shape[1])
        gesture_model.model = model
        gesture_model.scaler = scaler
        
        return gesture_model



# 4:real-time detection


class RealtimeDetector:
    
    
    def __init__(self, model_path, label_map_path):
        self.model = GestureModel.load(model_path,
                               scaler_path='models/gesture_model_scaler.pkl')
        
        with open(label_map_path, 'rb') as f:
            self.label_map = pickle.load(f)
        
        self.extractor = LandmarkExtractor()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def detect_landmarks(self, frame, hand_idx=0):
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if (results.multi_hand_landmarks and 
            len(results.multi_hand_landmarks) > hand_idx):
            
            lms = results.multi_hand_landmarks[hand_idx]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
            normalized = LandmarkExtractor.normalize_landmarks(landmarks)
            
            return normalized, lms
        
        return None, None
    
    def draw_on_frame(self, frame, landmarks_obj, gesture, confidence):
        
        h, w, _ = frame.shape
        
        # draw landmarks
        if landmarks_obj:
            for idx, lm in enumerate(landmarks_obj.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Draw gesture
        if gesture:
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.putText(frame, f"{gesture.upper()}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def run(self):
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n🎥 Starting gesture detection (Press Q to quit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect both hands
            for hand_idx in range(2):
                features, landmarks_obj = self.detect_landmarks(frame, hand_idx)
                
                if features is not None:
                    pred_idx, confidence = self.model.predict(features)
                    gesture = self.label_map[pred_idx]
                    
                    frame = self.draw_on_frame(frame, landmarks_obj, 
                                              gesture, confidence)
            
            cv2.imshow('Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")



# 5: evaluation and visualization

class Evaluator:
    
    
    @staticmethod
    def evaluate_and_plot(model, X_test, y_test, label_map, history):
        
        X_scaled = model.scaler.transform(X_test)
        
        # predictions
        y_pred = np.argmax(model.model.predict(X_scaled, verbose=0), axis=1)
        
        # Report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(
            y_test, y_pred,
            target_names=[label_map[i] for i in range(len(label_map))],
            digits=4
        )
        print(report)
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # plot 1: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=[label_map[i] for i in range(len(label_map))],
                   yticklabels=[label_map[i] for i in range(len(label_map))])
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # plot 2: Training History
        if history:
            epochs = range(1, len(history.history['accuracy']) + 1)
            axes[1].plot(epochs, history.history['accuracy'], 'b-o', label='Training')
            axes[1].plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation')
            axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation.png', dpi=150, bbox_inches='tight')
        print("\n📊 Evaluation plot saved to evaluation.png")
        plt.show()



# main function


def main():
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition System'
    )
    parser.add_argument('--mode', choices=['collect', 'train', 'detect', 'evaluate'],
                       default='detect', help='Mode to run')
    parser.add_argument('--gesture', type=str, help='Gesture to collect (with collect mode)')
    
    args = parser.parse_args()
    
    
    if args.mode == 'collect':
        if not args.gesture:
            print("❌ Please specify --gesture with collect mode")
            return
        
        collector = DataCollector()
        collector.collect_gesture(args.gesture, CONFIG['samples_per_gesture'])
    
    
    elif args.mode == 'train':
        print("🚀 Starting training pipeline...")
        
        # Extract features
        print("\n📊 Extracting landmarks...")
        extractor = LandmarkExtractor()
        X, y, label_map = extractor.process_dataset(CONFIG['data_dir'])
        
        if len(X) == 0:
            print("❌ No training data found. Collect data first!")
            return
        
        print(f"✅ Extracted {len(X)} samples from {len(label_map)} gestures")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=42, stratify=y
        )
        
        # Build and train
        print(f"\n🧠 Building model...")
        model = GestureModel(len(label_map))
        model.build()
        model.summary = model.model.summary()
        
        history = model.train(
            X_train, y_train,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size']
        )
        
        # Evaluate
        print("\n📈 Evaluating...")
        train_loss, train_acc = model.evaluate(X_train, y_train)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        # Save
        Path(CONFIG['model_path']).parent.mkdir(parents=True, exist_ok=True)
        model.save(CONFIG['model_path'], CONFIG['scaler_path'])
        
        with open(CONFIG['label_map_path'], 'wb') as f:
            pickle.dump(label_map, f)
        
        # Evaluate and plot
        Evaluator.evaluate_and_plot(model, X_test, y_test, label_map, history)
    
    # mode 3:real-time detection
    elif args.mode == 'detect':
        if not os.path.exists(CONFIG['model_path']):
            print(f"❌ Model not found at {CONFIG['model_path']}")
            print("Train a model first with: python gesture_recognition_system.py --mode train")
            return
        
        detector = RealtimeDetector(CONFIG['model_path'], CONFIG['label_map_path'])
        detector.run()
    
    # mode 4: Evaluate
    elif args.mode == 'evaluate':
        print("📊 Evaluating model...")
        
        extractor = LandmarkExtractor()
        X, y, label_map = extractor.process_dataset(CONFIG['data_dir'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], random_state=42, stratify=y
        )
        
        model = GestureModel.load(CONFIG['model_path'], CONFIG['scaler_path'])
        
        with open(CONFIG['label_map_path'], 'rb') as f:
            label_map = pickle.load(f)
        
        Evaluator.evaluate_and_plot(model, X_test, y_test, label_map, None)


if __name__ == '__main__':
    main()
