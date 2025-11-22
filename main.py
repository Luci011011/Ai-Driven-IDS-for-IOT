"""
IoT AI-Powered Intrusion Detection System
Main Application Launcher
"""

import os
import sys
import webbrowser
import threading
import time

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    print("🛡️ IoT AI-Powered Intrusion Detection System")
    print("=" * 50)

    while True:
        print("\nPlease choose an option:")
        print("1. 🏗️  Create Sample Dataset")
        print("2. 🤖 Train AI Model")
        print("3. 📊 Launch Dashboard")
        print("4. 🔍 Start Real-time Monitoring")
        print("5. 🚪 Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            print("\nCreating sample dataset...")
            from src.data_processing import create_sample_dataset
            create_sample_dataset()

        elif choice == "2":
            print("\nTraining AI model...")
            from src.model_training import train_iot_ids_model
            train_iot_ids_model()

        elif choice == "3":
            print("\nLaunching dashboard...")
            print("Dashboard will open in your browser at http://localhost:8501")

            # Start dashboard in a thread
            def run_dashboard():
                os.system("streamlit run src/dashboard.py")

            dashboard_thread = threading.Thread(target=run_dashboard)
            dashboard_thread.daemon = True
            dashboard_thread.start()

            # Open browser after short delay
            time.sleep(3)
            webbrowser.open("http://localhost:8501")

        elif choice == "4":
            print("\nStarting real-time monitoring...")
            print("Press Ctrl+C in the monitoring window to stop")
            os.system("python src/real_time_ids.py")

        elif choice == "5":
            print("\nThank you for using IoT AI-IDS! 👋")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()