"""
Kaggle Arabic Lip Reading Server with LocalTunnel
Optimized for Kaggle/Colab environments with robust LocalTunnel integration
"""

import subprocess
import os
import time
import threading
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Get backend directory based on current file location
BACKEND_DIR = Path(__file__).parent

def ensure_env_file():
    """Ensure .env file exists, copy from .env.example if missing"""
    env_path = BACKEND_DIR / ".env"
    env_example_path = BACKEND_DIR / ".env.example"
    
    if not env_path.exists():
        if env_example_path.exists():
            print(f"📋 .env file not found, copying from {env_example_path.name}")
            try:
                shutil.copy2(env_example_path, env_path)
                print(f"✅ Created .env file from {env_example_path.name}")
            except Exception as e:
                print(f"❌ Failed to copy {env_example_path.name} to {env_path.name}: {e}")
                print("⚠️  Continuing with default environment variables...")
        else:
            print(f"⚠️  Neither .env nor {env_example_path.name} found. Using default environment variables.")
    else:
        print("✅ .env file found")

# Ensure .env file exists before loading
ensure_env_file()

# Load environment variables
load_dotenv()

def parse_boolean(value: str) -> bool:
    """Parse boolean value from environment variable."""
    if not value:
        return False
    return value.lower() in ("true", "1", "yes", "on")

def parse_port(port_str: str) -> int:
    """Parse port number from environment variable."""
    try:
        port = int(port_str)
        if 1 <= port <= 65535:
            return port
        else:
            print(f"Warning: Invalid port {port}, using default 8000")
            return 8000
    except (ValueError, TypeError):
        print(f"Warning: Could not parse port '{port_str}', using default 8000")
        return 8000

# Configuration
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = parse_port(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
TUNNEL_SUBDOMAIN = os.getenv("TUNNEL_SUBDOMAIN", "arabic-lip-reading")

def run_command(command):
    """Run a shell command and stream output to notebook cell in real time"""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            print(line, end='')
        process.stdout.close()
        process.wait()
        return process
    except Exception as e:
        print(f"❌ Exception while running: {command}")
        print(f"Error: {e}")
        return None

def install_dependencies():
    """Install required packages for Kaggle environment"""
    print("📦 Installing Python dependencies...")
    
    packages = [
        "fastapi",
        "uvicorn[standard]", 
        "python-multipart",
        "google-generativeai",
        "python-dotenv",
        "requests",
        "kornia",
        "editdistance",
        "scikit-image",
        "gdown",
        "av"
    ]
    
    result = run_command(f"pip install {' '.join(packages)} --quiet")
    if result is not None:
        print("✅ Python dependencies installed")
    else:
        print("❌ Failed to install some Python dependencies")

def install_nodejs_and_localtunnel():
    """Install Node.js and LocalTunnel in Kaggle environment"""
    print("🔧 Setting up Node.js and LocalTunnel...")
    
    # Try different methods to install Node.js
    methods = [
        # Method 1: Use NodeSource repository (most reliable)
        [
            "curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -",
            "sudo apt-get install -y nodejs"
        ],
        # Method 2: Use snap (fallback)
        [
            "sudo snap install node --classic"
        ],
        # Method 3: Direct download (last resort)
        [
            "wget https://nodejs.org/dist/v18.17.0/node-v18.17.0-linux-x64.tar.xz",
            "tar -xf node-v18.17.0-linux-x64.tar.xz",
            "export PATH=$PATH:$(pwd)/node-v18.17.0-linux-x64/bin"
        ]
    ]
    
    node_installed = False
    
    # Check if Node.js is already installed
    result = run_command("node --version")
    if result and result.returncode == 0:
        print("✅ Node.js already installed")
        node_installed = True
    
    # Try to install Node.js if not present
    if not node_installed:
        for i, method in enumerate(methods, 1):
            print(f"Trying Node.js installation method {i}...")
            success = True
            for cmd in method:
                result = run_command(cmd)
                if result is None or result.returncode != 0:
                    success = False
                    break
            
            if success:
                # Verify installation
                result = run_command("node --version")
                if result and result.returncode == 0:
                    print(f"✅ Node.js installed successfully with method {i}")
                    node_installed = True
                    break
    
    if not node_installed:
        print("❌ Failed to install Node.js. LocalTunnel will not be available.")
        return False
    
    # Install LocalTunnel
    print("Installing LocalTunnel...")
    result = run_command("npm install -g localtunnel")
    if result and result.returncode == 0:
        print("✅ LocalTunnel installed successfully")
        return True
    else:
        print("❌ Failed to install LocalTunnel")
        return False

def start_localtunnel_thread(port):
    """Start LocalTunnel in a separate thread"""
    subdomain = TUNNEL_SUBDOMAIN
    
    def run_localtunnel():
        try:
            print(f"🌐 Starting LocalTunnel for port {port} with subdomain '{subdomain}'...")
            
            # Build command with optional subdomain
            command = ["lt", "--port", str(port), "--subdomain", subdomain]
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Read output and look for URL
            for line in process.stdout:
                # Look for the public URL
                if "your url is:" in line.lower():
                    url = line.split("your url is:")[-1].strip()
                    print(f"\n🌍 PUBLIC URL: {url}")
                    print(f"🔗 Use this URL to access your API from anywhere!")
                    
        except Exception as e:
            print(f"LocalTunnel error: {e}")
    
    thread = threading.Thread(target=run_localtunnel, daemon=True)
    thread.start()
    return thread

def start_server():
    try:
        import uvicorn
        uvicorn.run("main:app", host=API_HOST, port=API_PORT, log_level=LOG_LEVEL)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("\n💡 Try running manually:")
        print(f"   cd {BACKEND_DIR}")
        print(f"   python -m uvicorn main:app --host {API_HOST} --port {API_PORT}")

def start_server_and_tunnel():
    """Start both the FastAPI server and LocalTunnel"""
    print("🚀 Starting Arabic Lip Reading Server...")
    
    # Install everything
    install_dependencies()
    localtunnel_available = install_nodejs_and_localtunnel()
    
    if not BACKEND_DIR.exists():
        print("❌ Failed to set up repository")
        return

    # Change to backend directory
    os.chdir(str(BACKEND_DIR))
    
    print(f"\n✅ Setup complete!")
    print(f"📁 Backend directory: {BACKEND_DIR}")
    print(f"🌐 Local URL: http://{API_HOST}:{API_PORT}")
    
    # Start the FastAPI server
    print(f"🚀 Starting FastAPI server on {API_HOST}:{API_PORT} ...")
    threading.Thread(target=start_server).start()
    time.sleep(2)

    # Start LocalTunnel if available
    if localtunnel_available:
        print("🔄 Starting LocalTunnel...")
        start_localtunnel_thread(API_PORT)


def main():
    """Main entry point"""
    print("🇸🇦 Arabic Lip Reading Server - Kaggle Edition with LocalTunnel")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Host: {API_HOST}")
    print(f"   Port: {API_PORT}")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Tunnel Subdomain: {TUNNEL_SUBDOMAIN}")
    print(f"   Environment File: {'.env found' if (BACKEND_DIR / '.env').exists() else '.env not found'}")
    print("=" * 70)
    start_server_and_tunnel()
