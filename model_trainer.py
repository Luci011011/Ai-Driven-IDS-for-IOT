import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import joblib
import os
from config import MODELS_DIR


class ModelTrainer:
    def __init__(self):
        self.models = {}

    def create_ann_model(self, input_dim, num_classes=2):
        """Create an Artificial Neural Network model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
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
        """Create a 1D CNN model for sequence data"""
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(input_dim, 1)),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
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

    def train_ann(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the ANN model"""
        model = self.create_ann_model(X_train.shape[1])

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, 'best_ann_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
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

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # Save the model
        joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))

        return rf_model

    def evaluate_model(self, model, X_test, y_test, model_type='ann'):
        """Evaluate the trained model"""
        if model_type == 'ann':
            y_pred = np.argmax(model.predict(X_test), axis=1)
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return y_pred, accuracy

    def save_model(self, model, filename):
        """Save trained model"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        filepath = os.path.join(MODELS_DIR, filename)

        if filename.endswith('.h5'):
            model.save(filepath)
        else:
            joblib.dump(model, filepath)

        print(f"Model saved to {filepath}")