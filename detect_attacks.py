import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import IoTDetector
from config import MODELS_DIR


def main():
    print("=== IoT AI-Driven IDS Detection System ===")

    # Initialize detector
    detector = IoTDetector()

    # Load model and preprocessor
    model_loaded = detector.load_model(
        os.path.join(MODELS_DIR, 'iot_ids_ann.h5'),
        model_type='ann'
    )

    preprocessor_loaded = detector.load_preprocessor(
        os.path.join(MODELS_DIR, 'preprocessor.joblib')
    )

    if not model_loaded or not preprocessor_loaded:
        print("Failed to load required files. Please train the model first.")
        return

    print("Starting real-time detection...")
    print("Press Ctrl+C to stop\n")

    try:
        detection_count = 0
        while True:
            # Generate sample traffic (in real scenario, this would come from IoT devices)
            sample_traffic = detector.generate_sample_traffic()

            # Detect attack
            result = detector.detect_attack(sample_traffic)

            if result:
                detection_count += 1
                status = "ATTACK" if result['is_attack'] else "Normal"
                prob = result['attack_probability']

                print(f"Detection #{detection_count}: {status} (Probability: {prob:.3f})")

                # Print stats every 10 detections
                if detection_count % 10 == 0:
                    stats = detector.get_detection_stats()
                    print(f"\n--- Stats: {stats['attack_count']}/{stats['total_detections']} attacks detected ---\n")

            time.sleep(1)  # Simulate real-time detection

    except KeyboardInterrupt:
        print("\n\nStopping detection system...")
        final_stats = detector.get_detection_stats()
        print(f"Final Statistics: {final_stats}")


if __name__ == "__main__":
    main()