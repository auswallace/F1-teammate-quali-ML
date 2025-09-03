#!/usr/bin/env python3
"""
F1 Teammate Qualifying Dashboard Launcher
Automatically navigates to the right directory and launches the Streamlit app.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the F1 Teammate Qualifying Dashboard."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    print("🚀 Launching F1 Teammate Qualifying Dashboard...")
    print(f"📁 Working directory: {script_dir}")
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if we're in the right place
    if not Path("app.py").exists():
        print("❌ Error: app.py not found in current directory")
        print("Make sure you're running this from the f1_teammate_qual directory")
        return 1
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Error: Virtual environment (.venv) not found")
        print("Please run 'make venv && make install' first")
        return 1
    
    print("✅ Found app.py and virtual environment")
    print("🔧 Activating virtual environment...")
    
    # Determine the activation script path
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print("❌ Error: Python executable not found in virtual environment")
        return 1
    
    print("🚀 Starting Streamlit app...")
    print("🌐 The app will open in your browser at: http://localhost:8501")
    print("📱 If it doesn't open automatically, navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Launch Streamlit using the virtual environment's Python
        subprocess.run([str(python_exe), "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it with 'pip install streamlit'")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
