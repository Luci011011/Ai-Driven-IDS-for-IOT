import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import IoTDataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from config import MODEL_CONFIG


def main():
    print("=== IoT AI-Driven IDS Training ===")

    # Step 1: Generate or load data
    print("1. Loading data...")
    loader = IoTDataLoader()
    data_file = loader.save_sample_data()  # Generate sample data
    df = loader.load_data(data_file)
    print(f"Data loaded: {df.shape[0]} samples")

    # Step 2: Preprocess data
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        df, test_size=MODEL_CONFIG['test_size']
    )
    preprocessor.save_preprocessor()

    # Step 3: Train models
    print("3. Training models...")
    trainer = ModelTrainer()

    # Train ANN
    print("Training ANN model...")
    ann_history = trainer.train_ann(
        X_train, y_train, X_test, y_test,
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size']
    )

    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = trainer.train_random_forest(X_train, y_train)

    # Step 4: Evaluate models
    print("4. Evaluating models...")

    print("\nANN Model Evaluation:")
    ann_pred, ann_accuracy = trainer.evaluate_model(
        trainer.models['ann'], X_test, y_test, 'ann'
    )

    print("\nRandom Forest Evaluation:")
    rf_pred, rf_accuracy = trainer.evaluate_model(
        trainer.models['random_forest'], X_test, y_test, 'rf'
    )

    # Save models
    trainer.save_model(trainer.models['ann'], 'iot_ids_ann.h5')
    trainer.save_model(trainer.models['random_forest'], 'iot_ids_rf.joblib')

    print("\n=== Training Complete ===")
    print(f"ANN Accuracy: {ann_accuracy:.4f}")
    print(f"RF Accuracy: {rf_accuracy:.4f}")


if __name__ == "__main__":
    main()