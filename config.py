import os
from pathlib import Path

# 프로젝트 설정
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / "code_embeddings.db"

# 임베딩 설정
EMBEDDING_METHOD = "ollama"  # "openai", "sentence_transformer", 또는 "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI 사용시 필요

# Ollama 설정
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"  # 또는 "nomic-embed-text"

# 청킹 설정
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 30  # 최소 청크 길이 (이보다 짧으면 제외)

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = [
    '*.html', '*.css', '*.js', '*.jsx', '*.ts', '*.tsx',
    '*.vue', '*.php', '*.py', '*.java', '*.json', '*.xml',
    '*.md', '*.txt'
]

# 제외할 디렉토리
EXCLUDE_DIRS = [
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.next', '.nuxt'
]

# 제외할 파일
EXCLUDE_FILES = [
    '*.min.js', '*.min.css', '*.map', '*.lock', '*.log'
]