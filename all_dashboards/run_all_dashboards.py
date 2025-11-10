"""
Launcher script to run both Flask and Streamlit dashboards simultaneously
"""
import subprocess
import sys
import time
import os
from threading import Thread

def run_flask():
    """Run Flask application"""
    print("\n" + "="*80)
    print(" " * 25 + "🌍 FLASK - AIR QUALITY MONITORING")
    print("="*80)
    print("\n🚀 Starting Flask server on http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    subprocess.run([sys.executable, "app.py"])

def run_streamlit():
    """Run Streamlit application"""
    # Wait a bit for Flask to start first
    time.sleep(2)
    
    print("\n" + "="*80)
    print(" " * 25 + "📊 STREAMLIT - INTERACTIVE DASHBOARD")
    print("="*80)
    print("\n🚀 Starting Streamlit on http://localhost:8501")
    print("="*80 + "\n")
    
    subprocess.run([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "streamlit_dashboard.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

def main():
    """Main launcher function"""
    print("\n" + "="*80)
    print(" " * 20 + "🚀 AIR QUALITY DASHBOARD LAUNCHER")
    print("="*80)
    print("\n📋 Starting all dashboards...")
    print("\n✅ Flask App:      http://127.0.0.1:5000")
    print("   - Main Dashboard:    http://127.0.0.1:5000/")
    print("   - Data Explorer:     http://127.0.0.1:5000/data-explorer")
    print("   - Forecast Engine:   http://127.0.0.1:5000/forecast-engine")
    print("   - AQI Dashboard:     http://127.0.0.1:5000/aqi")
    print("\n✅ Streamlit App:  http://localhost:8501")
    print("\n🛑 Press CTRL+C to stop all dashboards")
    print("="*80 + "\n")
    
    # Create threads for both applications
    flask_thread = Thread(target=run_flask, daemon=True)
    streamlit_thread = Thread(target=run_streamlit, daemon=True)
    
    # Start both threads
    flask_thread.start()
    streamlit_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print(" " * 25 + "🛑 SHUTTING DOWN DASHBOARDS")
        print("="*80)
        print("\n✅ All dashboards stopped successfully")
        print("="*80 + "\n")
        sys.exit(0)

if __name__ == "__main__":
    main()