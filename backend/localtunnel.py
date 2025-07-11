import subprocess
import os
import sys
import time
import threading
import shutil
from pathlib import Path

# Get backend directory based on current file location
BACKEND_DIR = Path(__file__).parent

# ============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ============================================================================

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

# ============================================================================
# SYSTEM COMMAND EXECUTION
# ============================================================================

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

# ============================================================================
# LOCALTUNNEL SETUP AND MANAGEMENT
# ============================================================================

def ensure_localtunnel_installed(project_dir):
    """Ensure localtunnel is installed in the local project node_modules"""
    node_modules_path = os.path.join(project_dir, "node_modules", "localtunnel")
    
    if not os.path.exists(node_modules_path):
        print("📦 Installing LocalTunnel locally...")
        result = subprocess.run(
            ["npm", "install", "localtunnel", "--silent", "--no-progress", "--no-audit", "--no-fund"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=os.name == "nt"  # Use shell on Windows
        )
        if result.returncode != 0:
            print("❌ LocalTunnel installation failed.")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
        else:
            print("✅ LocalTunnel installed successfully")
    else:
        print("✅ LocalTunnel already installed locally")
    
    return True

def get_lt_command(project_dir, port, subdomain):
    """Get the local LocalTunnel client path and build command"""
    lt_script = os.path.join(project_dir, "node_modules", "localtunnel", "bin", "client")
    
    if os.path.exists(lt_script):
        node_exe = shutil.which("node")
        if not node_exe:
            print("❌ Node.js executable not found in PATH.")
            return None
        return [node_exe, lt_script, "--port", str(port), "--subdomain", subdomain, "--host", "https://loca.lt"]
    else:
        print("❌ LocalTunnel client script not found after installation.")
        return None

def run_localtunnel_local(project_dir, port, subdomain):
    """Run LocalTunnel using local installation (cross-platform)"""
    cmd = get_lt_command(project_dir, port, subdomain)
    if not cmd:
        return None

    print(f"🚀 Starting LocalTunnel: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in proc.stdout:
            print(line, end='')
            if "your url is" in line.lower() or "url:" in line.lower():
                url = line.strip().split()[-1]
                print(f"🌍 PUBLIC URL: {url}")
                return url

        proc.wait()
        print(f"❌ LocalTunnel exited with code {proc.returncode}")
        
    except Exception as e:
        print(f"❌ Error running LocalTunnel: {e}")
    
    return None

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================

def install_dependencies():
    """Install required Python packages"""
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
        "av",
        "dotenv"
    ]
    
    # Use sys.executable to ensure we're using the correct Python interpreter
    pip_command = f'"{sys.executable}" -m pip install {" ".join(packages)} --quiet'
    result = run_command(pip_command)
    if result is not None:
        print("✅ Python dependencies installed")
    else:
        print("❌ Failed to install some Python dependencies")

def install_nodejs():
    """Install Node.js if not already available"""
    print("🔧 Checking Node.js installation...")
    
    # Check if Node.js is already installed
    result = run_command("node --version")
    if result and result.returncode == 0:
        print("✅ Node.js already installed")
        return True

    print("🤔 Node.js not found. Attempting to install...")

    # Platform-specific installation
    if os.name == 'nt':  # Windows
        print("💻 Running on Windows. Trying to install with winget...")
        try:
            # Using winget to install Node.js LTS
            result = run_command("winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements --silent")
            if result and result.returncode == 0:
                print("✅ Node.js installed successfully via winget.")
                # NOTE: A restart of the script/shell might be needed for the PATH to update.
                return True
            else:
                print("❌ Winget installation failed or was cancelled.")
        except FileNotFoundError:
            print("❌ winget command not found. Please install Node.js manually from https://nodejs.org/")
            return False

    elif sys.platform.startswith('linux') or sys.platform == 'darwin': # Linux or macOS
        # The original script's Linux installation methods
        methods = [
            ["curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -", "sudo apt-get install -y nodejs"],
            ["sudo snap install node --classic"],
        ]
        for i, method in enumerate(methods, 1):
            print(f"🐧 Trying Node.js installation method {i}...")
            success = True
            for cmd in method:
                result = run_command(cmd)
                if result is None or result.returncode != 0:
                    success = False
                    break
            if success and shutil.which("node"):
                print(f"✅ Node.js installed successfully with method {i}")
                return True
    
    print("❌ Failed to automatically install Node.js.")
    print("   Please install it manually from https://nodejs.org/ and try again.")
    return False

# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

def start_localtunnel_thread(port):
    """Start LocalTunnel in a separate thread"""
    subdomain = TUNNEL_SUBDOMAIN
    
    def run_localtunnel():
        try:
            print(f"🌐 Starting LocalTunnel for port {port} with subdomain '{subdomain}'...")
            url = run_localtunnel_local(str(BACKEND_DIR), port, subdomain)
            if url:
                print(f"🔗 Use this URL to access your API from anywhere!")
            else:
                print("❌ Failed to start LocalTunnel")
        except Exception as e:
            print(f"❌ LocalTunnel error: {e}")
    
    thread = threading.Thread(target=run_localtunnel, daemon=True)
    thread.start()
    return thread

# ============================================================================
# MAIN APPLICATION STARTUP
# ============================================================================

def start_server():
    """Start the FastAPI server"""
    try:
        # Ensure we're in the correct directory
        original_cwd = os.getcwd()
        backend_dir_str = str(BACKEND_DIR)
        
        print(f"🔍 Server starting in directory: {backend_dir_str}")
        os.chdir(backend_dir_str)
        
        # Check if main.py exists
        if not os.path.exists("main.py"):
            print("❌ main.py not found in backend directory")
            return
            
        # Import and run uvicorn
        import uvicorn
        print(f"🚀 Running uvicorn with: main:app on {API_HOST}:{API_PORT}")
        uvicorn.run("main:app", host=API_HOST, port=API_PORT, log_level=LOG_LEVEL)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("❌ uvicorn not found. Please install it with: pip install uvicorn[standard]")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Current directory: {os.getcwd()}")
        print(f"❌ Backend directory: {BACKEND_DIR}")
        print("\n💡 Try running manually:")
        print(f"   cd {BACKEND_DIR}")
        print(f"   python -m uvicorn main:app --host {API_HOST} --port {API_PORT}")
    finally:
        # Restore original directory if needed
        try:
            os.chdir(original_cwd)
        except:
            pass

def start_server_and_tunnel():
    """Start both the FastAPI server and LocalTunnel"""
    print("🚀 Starting Arabic Lip Reading Server...")
    
    # Debug: Print current directory and backend directory
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Backend directory: {BACKEND_DIR}")
    print(f"🔍 Backend directory exists: {BACKEND_DIR.exists()}")
    
    # Install dependencies and setup environment
    install_dependencies()
    ensure_env_file()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Setup Node.js for LocalTunnel
    nodejs_available = install_nodejs()
    
    if not nodejs_available:
        print("❌ Cannot start server because Node.js is not available. Please install it and try again.")
        return

    if not BACKEND_DIR.exists():
        print("❌ Backend directory not found")
        return

    # Change to backend directory
    os.chdir(str(BACKEND_DIR))
    print(f"🔍 Changed to directory: {os.getcwd()}")
    
    # Check if main.py exists
    main_py_path = BACKEND_DIR / "main.py"
    print(f"🔍 main.py exists: {main_py_path.exists()}")
    
    # Install LocalTunnel after Node.js is available
    if nodejs_available:
        print("📦 Setting up LocalTunnel...")
        localtunnel_installed = ensure_localtunnel_installed(str(BACKEND_DIR))
        if not localtunnel_installed:
            print("⚠️  LocalTunnel installation failed, tunnel will not be available")
            nodejs_available = False
    
    print(f"\n✅ Setup complete!")
    print(f"📁 Backend directory: {BACKEND_DIR}")
    print(f"🌐 Local URL: http://{API_HOST}:{API_PORT}")
    
    # Start the FastAPI server in background
    print(f"🚀 Starting FastAPI server on {API_HOST}:{API_PORT}...")
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(2)  # Give server time to start

    # Start LocalTunnel if Node.js is available and LocalTunnel is installed
    if nodejs_available:
        print("🔄 Starting LocalTunnel...")
        start_localtunnel_thread(API_PORT)
    else:
        print("⚠️  LocalTunnel not available (Node.js or LocalTunnel installation required)")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


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

if __name__ == '__main__':
    main()