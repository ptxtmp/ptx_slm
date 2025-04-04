import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def download_vs_buildtools():
    """Download Visual Studio Build Tools installer"""
    print("Downloading Visual Studio Build Tools...")
    temp_dir = tempfile.gettempdir()
    installer_path = os.path.join(temp_dir, "vs_buildtools.exe")
    
    # Use PowerShell to download the file
    download_cmd = [
        "powershell", 
        "-Command", 
        f"Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vs_buildtools.exe -OutFile '{installer_path}'"
    ]
    
    try:
        subprocess.run(download_cmd, check=True)
        print(f"Downloaded installer to {installer_path}")
        return installer_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading VS Build Tools: {e}")
        return None

def install_vs_buildtools(installer_path):
    """Install Visual Studio Build Tools with C++ components"""
    print("Installing Visual Studio Build Tools (this may take a while)...")
    
    # Command to install the C++ build tools silently
    install_cmd = [
        installer_path,
        "--quiet", 
        "--wait", 
        "--norestart",
        "--nocache", 
        "--installPath", "C:\\BuildTools",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--includeRecommended"
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("Visual Studio Build Tools installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing VS Build Tools: {e}")
        return False

def install_llama_cpp_python():
    """Install llama-cpp-python after build tools are installed"""
    print("Installing llama-cpp-python...")
    
    # Set environment variables to use the new build tools
    env = os.environ.copy()
    
    # Try to install with CUDA support if available
    install_cmd = [
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "llama-cpp-python", 
        "--verbose",
        "--extra-index-url", 
        "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        subprocess.run(install_cmd, env=env, check=True)
        print("llama-cpp-python installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install with CUDA support, trying without...")
        
        # Try to install without CUDA support
        install_cmd = [
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "llama-cpp-python", 
            "--verbose"
        ]
        
        try:
            subprocess.run(install_cmd, env=env, check=True)
            print("llama-cpp-python installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing llama-cpp-python: {e}")
            return False

def try_prebuilt_wheels():
    """Try installing pre-built wheels as a fallback"""
    print("Trying to install pre-built wheels...")
    
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "llama-cpp-python",
        "--prefer-binary",
        "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121"
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("Successfully installed llama-cpp-python from pre-built wheels!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing from pre-built wheels: {e}")
        return False

def main():
    print("Setting up environment for llama-cpp-python...")
    
    # First try using pre-built wheels (fastest option)
    if try_prebuilt_wheels():
        return
    
    # If that fails, download and install VS Build Tools
    installer_path = download_vs_buildtools()
    if not installer_path:
        print("Failed to download Visual Studio Build Tools.")
        return
    
    if install_vs_buildtools(installer_path):
        print("Build tools installed. You may need to restart your command prompt.")
        print("After restarting, run: pip install llama-cpp-python")
        
        # Ask if user wants to try installing now
        response = input("Would you like to try installing llama-cpp-python now? (y/n): ")
        if response.lower() == 'y':
            # Wait a moment for installation to complete
            time.sleep(5)
            install_llama_cpp_python()
    else:
        print("Failed to install Visual Studio Build Tools.")
        print("You can try installing them manually from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

if __name__ == "__main__":
    main()