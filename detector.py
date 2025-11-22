import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import time
from datetime import datetime


class IoTDetector:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_type = None
        self.detection_history = []

    def load_model(self, model_path, model_type='ann'):
        """Load trained model"""
        try:
            if model_type == 'ann':
                self.model = load_model(model_path)
            else:
                self.model = joblib.load(model_path)

            self.model_type = model_type
            print(f"Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_preprocessor(self, preprocessor_path):
        """Load preprocessor"""
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded successfully: {preprocessor_path}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False

    def preprocess_single_sample(self, sample_data):
        """Preprocess a single sample for prediction"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")

        # Convert to DataFrame if needed
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        else:
            sample_df = sample_data

        # Scale features
        scaled_data = self.preprocessor['scaler'].transform(sample_df)
        return scaled_data

    def detect_attack(self, sample_data):
        """Detect if sample data represents an attack"""
        try:
            # Preprocess the sample
            processed_data = self.preprocess_single_sample(sample_data)

            # Make prediction
            if self.model_type == 'ann':
                prediction = self.model.predict(processed_data)
                attack_prob = prediction[0][1]  # Probability of being attack
                predicted_class = np.argmax(prediction, axis=1)[0]
            else:
                prediction = self.model.predict_proba(processed_data)
                attack_prob = prediction[0][1]
                predicted_class = self.model.predict(processed_data)[0]

            # Create detection result
            result = {
                'timestamp': datetime.now(),
                'is_attack': bool(predicted_class),
                'attack_probability': float(attack_prob),
                'predicted_class': int(predicted_class),
                'features': sample_data
            }

            # Add to history
            self.detection_history.append(result)

            return result

        except Exception as e:
            print(f"Error in detection: {e}")
            return None

    def get_detection_stats(self):
        """Get statistics of recent detections"""
        if not self.detection_history:
            return {"total_detections": 0, "attack_count": 0}

        attack_count = sum(1 for detection in self.detection_history if detection['is_attack'])

        return {
            "total_detections": len(self.detection_history),
            "attack_count": attack_count,
            "attack_ratio": attack_count / len(self.detection_history)
        }

    def generate_sample_traffic(self):
        """Generate sample IoT traffic for testing"""
        return {
            'packet_size': np.random.randint(40, 1500),
            'protocol_type': np.random.choice([0, 1, 2]),
            'duration': np.random.exponential(1.0),
            'src_bytes': np.random.poisson(100),
            'dst_bytes': np.random.poisson(100),
            'count': np.random.poisson(10),
            'srv_count': np.random.poisson(5),
            'dst_host_count': np.random.poisson(20),
            'dst_host_srv_count': np.random.poisson(15),
        }