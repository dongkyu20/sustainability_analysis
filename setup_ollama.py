#!/usr/bin/env python3
"""
Ollama 설치 및 설정 도우미 스크립트
"""

import subprocess
import requests
import time
import platform
import sys
import config

def check_ollama_installed():
    """Ollama가 설치되어 있는지 확인"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def install_ollama():
    """운영체제별 Ollama 설치 안내"""
    system = platform.system().lower()
    
    print("📥 Ollama is not installed. Please install it:")
    print()
    
    if system == "darwin":  # macOS
        print("🍎 macOS:")
        print("   Method 1: Download from https://ollama.ai/")
        print("   Method 2: brew install ollama")
        
    elif system == "linux":
        print("🐧 Linux:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        
    elif system == "windows":
        print("🪟 Windows:")
        print("   Download from https://ollama.ai/")
        
    print()
    print("After installation, run: ollama serve")
    return False

def check_ollama_running():
    """Ollama 서버가 실행 중인지 확인"""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running")
        print("💡 Please run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama server: {str(e)}")
        return False

def start_ollama_server():
    """Ollama 서버 시작 시도"""
    print("🚀 Attempting to start Ollama server...")
    try:
        # 백그라운드에서 ollama serve 실행
        process = subprocess.Popen(['ollama', 'serve'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        
        # 서버가 시작될 때까지 잠시 대기
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        if check_ollama_running():
            print(f"✅ Ollama server started with PID: {process.pid}")
            return True
        else:
            print("❌ Failed to start Ollama server")
            return False
            
    except FileNotFoundError:
        print("❌ 'ollama serve' command not found")
        return False
    except Exception as e:
        print(f"❌ Error starting Ollama server: {str(e)}")
        return False

def pull_embedding_model():
    """임베딩 모델 다운로드"""
    model = config.OLLAMA_EMBEDDING_MODEL
    print(f"📥 Pulling embedding model: {model}")
    
    try:
        # ollama pull 명령 실행
        process = subprocess.Popen(['ollama', 'pull', model], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)
        
        print("⏳ Downloading model... This may take several minutes.")
        
        # 실시간 출력
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
        
        if process.returncode == 0:
            print(f"✅ Model {model} downloaded successfully")
            return True
        else:
            stderr = process.stderr.read()
            print(f"❌ Failed to download model: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return False

def check_model_available():
    """모델이 사용 가능한지 확인"""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'].split(':')[0] for model in models]
            
            if config.OLLAMA_EMBEDDING_MODEL in model_names:
                print(f"✅ Model {config.OLLAMA_EMBEDDING_MODEL} is available")
                return True
            else:
                print(f"❌ Model {config.OLLAMA_EMBEDDING_MODEL} not found")
                return False
        else:
            print(f"❌ Failed to check models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")
        return False

def setup_ollama():
    """Ollama 전체 설정"""
    print("🔧 Setting up Ollama for code embedding...")
    print()
    
    # 1. Ollama 설치 확인
    if not check_ollama_installed():
        install_ollama()
        return False
    
    # 2. 서버 실행 확인
    if not check_ollama_running():
        if not start_ollama_server():
            print("💡 Please manually run: ollama serve")
            return False
    
    # 3. 모델 확인 및 다운로드
    if not check_model_available():
        if not pull_embedding_model():
            return False
    
    print()
    print("🎉 Ollama setup completed successfully!")
    print("🚀 You can now run the code embedding tool.")
    
    return True

def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        # 설정 상태만 확인
        print("🔍 Checking Ollama setup...")
        installed = check_ollama_installed()
        running = check_ollama_running() if installed else False
        model_ready = check_model_available() if running else False
        
        print(f"   Installed: {'✅' if installed else '❌'}")
        print(f"   Running: {'✅' if running else '❌'}")
        print(f"   Model Ready: {'✅' if model_ready else '❌'}")
        
        if installed and running and model_ready:
            print("🎉 Everything is ready!")
        else:
            print("⚠️ Setup required. Run without --check flag to setup.")
    else:
        # 전체 설정 실행
        setup_ollama()

if __name__ == "__main__":
    main()