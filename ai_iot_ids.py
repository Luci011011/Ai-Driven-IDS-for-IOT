# ai_iot_ids.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class IoTAIIDS:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detection_history = []
        self.model_dir = "iot_ids_models"
        os.makedirs(self.model_dir, exist_ok=True)

    def generate_sample_data(self, n_samples=10000):
        """Generate synthetic IoT traffic data with attack patterns"""
        print("Generating sample IoT traffic data...")
        np.random.seed(42)

        data = {
            'packet_size': np.random.randint(40, 1500, n_samples),
            'protocol_type': np.random.choice([0, 1, 2], n_samples),  # 0:TCP, 1:UDP, 2:ICMP
            'duration': np.random.exponential(1.0, n_samples),
            'src_bytes': np.random.poisson(100, n_samples),
            'dst_bytes': np.random.poisson(100, n_samples),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(5, n_samples),
            'dst_host_count': np.random.poisson(20, n_samples),
            'dst_host_srv_count': np.random.poisson(15, n_samples),
            'flag': np.random.randint(0, 10, n_samples),
            'service': np.random.randint(0, 5, n_samples),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.01, n_samples),
        }

        # Create labels (0: normal, 1: attack)
        labels = np.zeros(n_samples)

        # Introduce attack patterns (30% of data)
        attack_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
        labels[attack_indices] = 1

        # Modify features for attack samples to create realistic patterns
        for idx in attack_indices:
            attack_type = np.random.choice(['dos', 'probe', 'scan'], p=[0.6, 0.3, 0.1])

            if attack_type == 'dos':  # Denial of Service
                data['packet_size'][idx] = np.random.randint(1400, 1500)
                data['src_bytes'][idx] = np.random.poisson(1000)
                data['count'][idx] = np.random.poisson(100)
                data['duration'][idx] = np.random.exponential(0.1)

            elif attack_type == 'probe':  # Probing/Reconnaissance
                data['duration'][idx] = np.random.exponential(10.0)
                data['dst_bytes'][idx] = np.random.poisson(500)
                data['srv_count'][idx] = np.random.poisson(50)
                data['wrong_fragment'][idx] = np.random.poisson(2)

            else:  # Port scanning
                data['dst_host_count'][idx] = np.random.poisson(100)
                data['dst_host_srv_count'][idx] = np.random.poisson(80)
                data['count'][idx] = np.random.poisson(30)

        df = pd.DataFrame(data)
        df['label'] = labels

        print(f"Generated {len(df)} samples with {labels.sum()} attack instances")
        return df

    def preprocess_data(self, df, target_column='label', test_size=0.2):
        """Preprocess the data for training"""
        print("Preprocessing data...")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Testing set: {X_test_scaled.shape[0]} samples")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def create_ann_model(self, input_dim, num_classes=2):
        """Create Artificial Neural Network model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_cnn_model(self, input_dim, num_classes=2):
        """Create 1D CNN model for sequence data"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(input_dim, 1)),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_ann_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the ANN model"""
        print("Training ANN model...")

        model = self.create_ann_model(X_train.shape[1])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_ann_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        self.models['ann'] = model
        return history

    def train_cnn_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the CNN model"""
        print("Training CNN model...")

        # Reshape data for CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = self.create_cnn_model(X_train.shape[1])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_cnn_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]

        history = model.fit(
            X_train_cnn, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_cnn, y_test),
            callbacks=callbacks,
            verbose=1
        )

        self.models['cnn'] = model
        return history

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest model...")

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # Save the model
        joblib.dump(rf_model, os.path.join(self.model_dir, 'random_forest_model.joblib'))

        return rf_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        results = {}

        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name.upper()}...")

            if model_name in ['ann', 'cnn']:
                if model_name == 'cnn':
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
                else:
                    y_pred = np.argmax(model.predict(X_test), axis=1)
            else:
                y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy

            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)

        return results

    def save_models(self):
        """Save all trained models and preprocessor"""
        print("\nSaving models and preprocessor...")

        # Save ANN model
        if 'ann' in self.models:
            self.models['ann'].save(os.path.join(self.model_dir, 'iot_ids_ann.h5'))

        # Save CNN model
        if 'cnn' in self.models:
            self.models['cnn'].save(os.path.join(self.model_dir, 'iot_ids_cnn.h5'))

        # Save Random Forest model
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'],
                        os.path.join(self.model_dir, 'iot_ids_rf.joblib'))

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

        self.is_trained = True
        print("All models and preprocessor saved successfully!")

    def load_models(self, model_type='ann'):
        """Load trained models and preprocessor"""
        print(f"Loading {model_type.upper()} model...")

        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print("Scaler not found. Please train models first.")
                return False

            # Load specific model
            if model_type == 'ann':
                model_path = os.path.join(self.model_dir, 'iot_ids_ann.h5')
                if os.path.exists(model_path):
                    self.models['ann'] = load_model(model_path)
                else:
                    print("ANN model not found.")
                    return False

            elif model_type == 'cnn':
                model_path = os.path.join(self.model_dir, 'iot_ids_cnn.h5')
                if os.path.exists(model_path):
                    self.models['cnn'] = load_model(model_path)
                else:
                    print("CNN model not found.")
                    return False

            elif model_type == 'random_forest':
                model_path = os.path.join(self.model_dir, 'iot_ids_rf.joblib')
                if os.path.exists(model_path):
                    self.models['random_forest'] = joblib.load(model_path)
                else:
                    print("Random Forest model not found.")
                    return False

            self.is_trained = True
            print(f"{model_type.upper()} model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def generate_live_traffic(self):
        """Generate simulated live IoT traffic"""
        # Base normal traffic
        traffic = {
            'packet_size': np.random.randint(40, 1200),
            'protocol_type': np.random.choice([0, 1, 2]),
            'duration': max(0.1, np.random.exponential(1.0)),
            'src_bytes': np.random.poisson(80),
            'dst_bytes': np.random.poisson(90),
            'count': np.random.poisson(8),
            'srv_count': np.random.poisson(4),
            'dst_host_count': np.random.poisson(18),
            'dst_host_srv_count': np.random.poisson(12),
            'flag': np.random.randint(0, 8),
            'service': np.random.randint(0, 4),
            'wrong_fragment': np.random.poisson(0.05),
            'urgent': np.random.poisson(0.005),
        }

        # Occasionally introduce attacks (10% chance)
        if np.random.random() < 0.1:
            attack_type = np.random.choice(['dos', 'probe', 'scan'], p=[0.7, 0.2, 0.1])

            if attack_type == 'dos':
                traffic.update({
                    'packet_size': np.random.randint(1400, 1500),
                    'src_bytes': np.random.poisson(800),
                    'count': np.random.poisson(80),
                    'duration': np.random.exponential(0.05)
                })
            elif attack_type == 'probe':
                traffic.update({
                    'duration': np.random.exponential(8.0),
                    'dst_bytes': np.random.poisson(400),
                    'srv_count': np.random.poisson(40),
                    'wrong_fragment': np.random.poisson(1.5)
                })
            else:  # scan
                traffic.update({
                    'dst_host_count': np.random.poisson(80),
                    'dst_host_srv_count': np.random.poisson(60),
                    'count': np.random.poisson(25)
                })

        return traffic

    def detect_attack(self, traffic_data, model_type='ann'):
        """Detect attacks in traffic data"""
        if not self.is_trained or model_type not in self.models:
            print("Model not loaded. Please load or train models first.")
            return None

        try:
            # Convert to DataFrame
            if isinstance(traffic_data, dict):
                sample_df = pd.DataFrame([traffic_data])
            else:
                sample_df = traffic_data

            # Scale features
            scaled_data = self.scaler.transform(sample_df)

            # Make prediction
            model = self.models[model_type]

            if model_type in ['ann', 'cnn']:
                if model_type == 'cnn':
                    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
                prediction = model.predict(scaled_data, verbose=0)
                attack_prob = prediction[0][1]
                predicted_class = np.argmax(prediction, axis=1)[0]
            else:  # random forest
                prediction = model.predict_proba(scaled_data)
                attack_prob = prediction[0][1]
                predicted_class = model.predict(scaled_data)[0]

            # Create result
            result = {
                'timestamp': datetime.now(),
                'is_attack': bool(predicted_class),
                'attack_probability': float(attack_prob),
                'confidence': max(float(attack_prob), 1 - float(attack_prob)),
                'predicted_class': int(predicted_class),
                'model_used': model_type,
                'features': traffic_data
            }

            # Add to history
            self.detection_history.append(result)

            return result

        except Exception as e:
            print(f"Error in detection: {e}")
            return None

    def get_detection_stats(self):
        """Get detection statistics"""
        if not self.detection_history:
            return {"total_detections": 0, "attack_count": 0, "attack_ratio": 0.0}

        total = len(self.detection_history)
        attacks = sum(1 for d in self.detection_history if d['is_attack'])
        ratio = attacks / total if total > 0 else 0.0

        # Recent attacks (last 50 detections)
        recent = self.detection_history[-50:]
        recent_attacks = sum(1 for d in recent if d['is_attack'])
        recent_ratio = recent_attacks / len(recent) if recent else 0.0

        return {
            "total_detections": total,
            "attack_count": attacks,
            "attack_ratio": ratio,
            "recent_attack_ratio": recent_ratio,
            "last_10_attacks": recent_attacks
        }

    def print_detection_alert(self, result):
        """Print formatted detection alert"""
        status = "🚨 ATTACK DETECTED" if result['is_attack'] else "✅ Normal Traffic"
        color = "\033[91m" if result['is_attack'] else "\033[92m"  # Red or Green
        reset = "\033[0m"

        print(f"{color}{status}{reset}")
        print(f"   Probability: {result['attack_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model: {result['model_used'].upper()}")
        print(f"   Time: {result['timestamp'].strftime('%H:%M:%S')}")

        if result['is_attack']:
            # Additional attack analysis
            if result['attack_probability'] > 0.8:
                print("   ⚠️  HIGH CONFIDENCE ATTACK")
            elif result['attack_probability'] > 0.6:
                print("   ⚠️  Medium confidence attack")
            else:
                print("   ⚠️  Suspicious activity")

        print("-" * 50)


def main():
    """Main function to run the IoT AI IDS"""
    print("🚀 IoT AI-Driven Intrusion Detection System")
    print("=" * 60)

    ids = IoTAIIDS()

    while True:
        print("\nOptions:")
        print("1. Train New Models")
        print("2. Load Existing Models")
        print("3. Start Real-time Detection")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            # Train models
            print("\n" + "=" * 50)
            print("TRAINING MODELS")
            print("=" * 50)

            # Generate and preprocess data
            df = ids.generate_sample_data(15000)
            X_train, X_test, y_train, y_test = ids.preprocess_data(df)

            # Train models
            ids.train_ann_model(X_train, y_train, X_test, y_test, epochs=30)
            ids.train_random_forest(X_train, y_train)

            # Evaluate models
            results = ids.evaluate_models(X_test, y_test)

            # Save models
            ids.save_models()

            print("\nTraining completed!")
            for model, acc in results.items():
                print(f"{model.upper()}: {acc:.4f}")

        elif choice == '2':
            # Load models
            print("\nAvailable models:")
            print("1. ANN")
            print("2. Random Forest")
            model_choice = input("Select model to load (1-2): ").strip()

            if model_choice == '1':
                success = ids.load_models('ann')
            elif model_choice == '2':
                success = ids.load_models('random_forest')
            else:
                print("Invalid choice")
                continue

            if success:
                print("Model loaded successfully!")

        elif choice == '3':
            # Real-time detection
            if not ids.is_trained:
                print("No model loaded. Please train or load a model first.")
                continue

            print("\n" + "=" * 50)
            print("STARTING REAL-TIME DETECTION")
            print("=" * 50)
            print("Press Ctrl+C to stop detection\n")

            try:
                detection_count = 0
                while True:
                    # Generate simulated traffic
                    traffic = ids.generate_live_traffic()

                    # Detect attacks (using the first available model)
                    model_type = list(ids.models.keys())[0] if ids.models else 'ann'
                    result = ids.detect_attack(traffic, model_type)

                    if result:
                        detection_count += 1
                        ids.print_detection_alert(result)

                        # Print stats every 20 detections
                        if detection_count % 20 == 0:
                            stats = ids.get_detection_stats()
                            print(f"\n📊 Detection Statistics:")
                            print(f"   Total: {stats['total_detections']}")
                            print(f"   Attacks: {stats['attack_count']}")
                            print(f"   Attack Ratio: {stats['attack_ratio']:.2%}")
                            print(f"   Recent Attacks: {stats['last_10_attacks']}/50")
                            print("=" * 50 + "\n")

                    time.sleep(2)  # Simulate real-time processing

            except KeyboardInterrupt:
                print("\n\nStopping detection system...")
                final_stats = ids.get_detection_stats()
                print(f"\nFinal Statistics:")
                print(f"Total Detections: {final_stats['total_detections']}")
                print(f"Attacks Detected: {final_stats['attack_count']}")
                print(f"Overall Attack Ratio: {final_stats['attack_ratio']:.2%}")

        elif choice == '4':
            print("Exiting IoT AI IDS. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()