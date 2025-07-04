import os
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import re
from dataclasses import dataclass
import argparse
import time

@dataclass
class CodeChunk:
    file_path: str
    chunk_id: str
    content: str
    file_type: str
    start_line: int
    end_line: int
    embedding: List[float] = None

class CodeChunker:
    def __init__(self, min_chunk_length: int = 30):
        self.min_chunk_length = min_chunk_length
        
    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """파일을 적절한 크기로 chunking"""
        file_type = self._get_file_type(file_path)
        chunks = []
        
        if file_type in ['html', 'xml']:
            chunks = self._chunk_html(file_path, content)
        elif file_type in ['js', 'javascript']:
            chunks = self._chunk_javascript(file_path, content)
        elif file_type in ['css']:
            chunks = self._chunk_css(file_path, content)
        elif file_type in ['py', 'python']:
            chunks = self._chunk_python(file_path, content)
        else:
            chunks = self._chunk_generic(file_path, content)
        
        # 최소 길이 필터링
        filtered_chunks = [chunk for chunk in chunks if len(chunk.content.strip()) >= self.min_chunk_length]
        print(f"  - {len(chunks)}개 청크 생성, {len(filtered_chunks)}개 청크 유효 (최소 길이: {self.min_chunk_length}자)")
        
        return filtered_chunks
    
    def _get_file_type(self, file_path: str) -> str:
        """파일 확장자로 타입 결정"""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.html': 'html', '.htm': 'html',
            '.js': 'js', '.jsx': 'js', '.ts': 'js', '.tsx': 'js',
            '.css': 'css', '.scss': 'css', '.sass': 'css',
            '.py': 'python',
            '.xml': 'xml'
        }
        return type_map.get(ext, 'text')
    
    def _chunk_html(self, file_path: str, content: str) -> List[CodeChunk]:
        """HTML 파일을 태그 단위로 chunking"""
        chunks = []
        lines = content.split('\n')
        
        # 스크립트 태그 추출
        script_pattern = r'<script[^>]*>(.*?)</script>'
        scripts = re.finditer(script_pattern, content, re.DOTALL)
        for match in scripts:
            script_content = match.group(1).strip()
            if len(script_content) >= self.min_chunk_length:
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=script_content,
                    file_type='js',
                    start_line=start_line,
                    end_line=end_line
                ))
        
        # 스타일 태그 추출
        style_pattern = r'<style[^>]*>(.*?)</style>'
        styles = re.finditer(style_pattern, content, re.DOTALL)
        for match in styles:
            style_content = match.group(1).strip()
            if len(style_content) >= self.min_chunk_length:
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=style_content,
                    file_type='css',
                    start_line=start_line,
                    end_line=end_line
                ))
        
        # HTML 컨텐츠를 div, section 등 블록 단위로 분할
        current_chunk = []
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # 블록 태그 종료 시 청크 생성
            if re.search(r'</(?:div|section|article|main|header|footer|nav|aside)>', line):
                chunk_content = '\n'.join(current_chunk)
                if len(chunk_content.strip()) >= self.min_chunk_length:
                    chunk_id = hashlib.md5(f"{file_path}:{start_line}:{i}".encode()).hexdigest()
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        chunk_id=chunk_id,
                        content=chunk_content,
                        file_type='html',
                        start_line=start_line,
                        end_line=i
                    ))
                current_chunk = []
                start_line = i + 1
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content.strip()) >= self.min_chunk_length:
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{len(lines)}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=chunk_content,
                    file_type='html',
                    start_line=start_line,
                    end_line=len(lines)
                ))
        
        return chunks
    
    def _chunk_javascript(self, file_path: str, content: str) -> List[CodeChunk]:
        """JavaScript 파일을 함수/클래스 단위로 chunking"""
        chunks = []
        lines = content.split('\n')
        
        # 함수 패턴 매칭
        function_patterns = [
            r'function\s+\w+\s*\([^)]*\)\s*{',
            r'const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*{',
            r'let\s+\w+\s*=\s*\([^)]*\)\s*=>\s*{',
            r'var\s+\w+\s*=\s*function\s*\([^)]*\)\s*{',
            r'class\s+\w+\s*{',
            r'\w+\s*:\s*function\s*\([^)]*\)\s*{',
        ]
        
        current_chunk = []
        brace_count = 0
        in_function = False
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            # 함수 시작 감지
            if not in_function:
                for pattern in function_patterns:
                    if re.search(pattern, line):
                        in_function = True
                        start_line = i
                        break
            
            if in_function:
                current_chunk.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # 함수 종료 감지
                if brace_count <= 0:
                    chunk_content = '\n'.join(current_chunk)
                    if len(chunk_content.strip()) >= self.min_chunk_length:
                        chunk_id = hashlib.md5(f"{file_path}:{start_line}:{i}".encode()).hexdigest()
                        chunks.append(CodeChunk(
                            file_path=file_path,
                            chunk_id=chunk_id,
                            content=chunk_content,
                            file_type='js',
                            start_line=start_line,
                            end_line=i
                        ))
                    current_chunk = []
                    brace_count = 0
                    in_function = False
        
        # 함수 외부의 코드도 청크로 만들기
        if not chunks:
            return self._chunk_generic(file_path, content)
        
        return chunks
    
    def _chunk_css(self, file_path: str, content: str) -> List[CodeChunk]:
        """CSS 파일을 규칙 단위로 chunking"""
        chunks = []
        
        # CSS 규칙 패턴 매칭
        css_rules = re.finditer(r'([^{}]+){([^{}]*)}', content, re.DOTALL)
        
        for match in css_rules:
            selector = match.group(1).strip()
            declarations = match.group(2).strip()
            rule_content = f"{selector} {{\n{declarations}\n}}"
            
            if len(rule_content.strip()) >= self.min_chunk_length:
                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=rule_content,
                    file_type='css',
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return chunks
    
    def _chunk_python(self, file_path: str, content: str) -> List[CodeChunk]:
        """Python 파일을 함수/클래스 단위로 chunking.

        - 함수/클래스 정의 직전에 연속으로 배치된 주석(`# ...`)과 빈 줄을
          해당 함수/클래스의 청크에 포함해 설명과 코드가 분리되지 않도록 한다.
        """
        chunks: List[CodeChunk] = []
        lines: List[str] = content.split('\n')

        current_chunk: List[str] = []
        start_line: int = 1
        current_indent: int = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            is_def_line = re.match(r'^(def|class)\s+\w+', stripped) is not None

            if is_def_line:
                # ----- 이전 청크 저장 (trailing comment/blank lines 제외) -----
                if current_chunk:
                    trailing_comments: List[str] = []
                    while current_chunk and (current_chunk[-1].strip() == '' or current_chunk[-1].lstrip().startswith('#')):
                        trailing_comments.insert(0, current_chunk.pop())

                    if current_chunk:  # 실제 코드가 존재할 때만 저장
                        chunk_content = '\n'.join(current_chunk)
                        if len(chunk_content.strip()) >= self.min_chunk_length:
                            end_line = i - len(trailing_comments) - 1
                            chunk_id = hashlib.md5(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                            chunks.append(CodeChunk(
                                file_path=file_path,
                                chunk_id=chunk_id,
                                content=chunk_content,
                                file_type='python',
                                start_line=start_line,
                                end_line=end_line
                            ))
                else:
                    trailing_comments = []

                # ----- 새 청크 시작 : 분리한 주석 + 현재 def/class 줄 -----
                current_chunk = trailing_comments + [line]
                start_line = i - len(trailing_comments)
                current_indent = len(line) - len(line.lstrip())
                continue

            # def/class 라인이 아닌 경우
            if current_chunk:
                # 들여쓰기 레벨을 통해 함수/클래스 블록 종료 감지
                if stripped and (len(line) - len(line.lstrip())) <= current_indent and not stripped.startswith('#'):
                    # 함수/클래스 블록 종료: 현재 줄은 다음 블록의 첫 줄일 수 있음
                    chunk_content = '\n'.join(current_chunk)
                    if len(chunk_content.strip()) >= self.min_chunk_length:
                        chunk_id = hashlib.md5(f"{file_path}:{start_line}:{i-1}".encode()).hexdigest()
                        chunks.append(CodeChunk(
                            file_path=file_path,
                            chunk_id=chunk_id,
                            content=chunk_content,
                            file_type='python',
                            start_line=start_line,
                            end_line=i-1
                        ))

                    # 다음 블록 준비 (주석/빈 줄로 시작할 수 있음)
                    current_chunk = [line]
                    start_line = i
                    current_indent = len(line) - len(line.lstrip())
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)

        # ----- 마지막 청크 저장 -----
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content.strip()) >= self.min_chunk_length:
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{len(lines)}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=chunk_content,
                    file_type='python',
                    start_line=start_line,
                    end_line=len(lines)
                ))

        return chunks
    
    def _chunk_generic(self, file_path: str, content: str) -> List[CodeChunk]:
        """일반적인 파일을 라인 단위로 chunking"""
        chunks = []
        lines = content.split('\n')
        
        chunk_size = 50  # 라인 수
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            if len(chunk_content.strip()) >= self.min_chunk_length:
                start_line = i + 1
                end_line = min(i + chunk_size, len(lines))
                chunk_id = hashlib.md5(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=chunk_id,
                    content=chunk_content,
                    file_type=self._get_file_type(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return chunks

class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-coder-v2:16b"):
        self.base_url = base_url
        self.model = model
        self._test_connection()
    
    def _test_connection(self):
        """Ollama 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"Ollama 연결 성공: {self.base_url}")
            
            # 모델 존재 확인
            tags = response.json()
            models = [model["name"] for model in tags.get("models", [])]
            if self.model not in models:
                print(f"경고: {self.model} 모델이 설치되지 않았습니다.")
                print(f"사용 가능한 모델: {models}")
            else:
                print(f"모델 확인 완료: {self.model}")
                
        except requests.RequestException as e:
            print(f"Ollama 연결 실패: {e}")
            print("Ollama가 실행 중인지 확인하세요: ollama serve")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Ollama API를 사용하여 임베딩 생성"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [])
        
        except requests.RequestException as e:
            print(f"임베딩 생성 실패: {e}")
            return []
    
    def batch_generate_embeddings(self, chunks: List[CodeChunk], batch_size: int = 10) -> List[CodeChunk]:
        """여러 청크에 대해 배치로 임베딩 생성"""
        embedded_chunks = []
        total_chunks = len(chunks)
        
        print(f"총 {total_chunks}개 청크에 대해 임베딩 생성 시작...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_embedded = []
            
            for j, chunk in enumerate(batch):
                current_idx = i + j + 1
                print(f"임베딩 생성 중... ({current_idx}/{total_chunks}) - {chunk.file_path}:{chunk.start_line}")
                
                embedding = self.generate_embedding(chunk.content)
                
                if embedding:
                    chunk.embedding = embedding
                    batch_embedded.append(chunk)
                    print(f"  ✓ 성공 (임베딩 차원: {len(embedding)})")
                else:
                    print(f"  ✗ 실패")
                
                # API 과부하 방지를 위한 대기
                time.sleep(0.02)
            
            embedded_chunks.extend(batch_embedded)
            
            # 배치 간 대기
            if i + batch_size < total_chunks:
                print(f"배치 완료 ({len(batch_embedded)}/{len(batch)}개 성공). 잠시 대기...")
                time.sleep(0.2)
        
        print(f"임베딩 생성 완료: {len(embedded_chunks)}/{total_chunks}개 성공")
        return embedded_chunks

class CodeEmbeddingDB:
    def __init__(self, db_path: str = "code_embeddings.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화 및 마이그레이션"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기본 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                file_path TEXT,
                content TEXT,
                start_line INTEGER,
                end_line INTEGER,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 스키마 마이그레이션: file_type 컬럼 추가
        try:
            cursor.execute('SELECT file_type FROM code_chunks LIMIT 1')
        except sqlite3.OperationalError:
            print("file_type 컬럼이 없습니다. 스키마를 업데이트합니다...")
            cursor.execute('ALTER TABLE code_chunks ADD COLUMN file_type TEXT DEFAULT "text"')
            print("file_type 컬럼이 추가되었습니다.")
        
        # 인덱스 생성
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_path ON code_chunks(file_path);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_type ON code_chunks(file_type);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_id ON code_chunks(chunk_id);
        ''')
        
        conn.commit()
        conn.close()
        print(f"데이터베이스 초기화 완료: {self.db_path}")
    
    def save_chunks(self, chunks: List[CodeChunk]):
        """청크들을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        updated_count = 0
        
        for chunk in chunks:
            if chunk.embedding:
                # 기존 청크 확인
                cursor.execute('SELECT id FROM code_chunks WHERE chunk_id = ?', (chunk.chunk_id,))
                existing = cursor.fetchone()
                
                if existing:
                    cursor.execute('''
                        UPDATE code_chunks 
                        SET file_path=?, content=?, file_type=?, start_line=?, end_line=?, embedding=?
                        WHERE chunk_id=?
                    ''', (
                        chunk.file_path,
                        chunk.content,
                        chunk.file_type,
                        chunk.start_line,
                        chunk.end_line,
                        json.dumps(chunk.embedding).encode('utf-8'),
                        chunk.chunk_id
                    ))
                    updated_count += 1
                else:
                    cursor.execute('''
                        INSERT INTO code_chunks 
                        (chunk_id, file_path, content, file_type, start_line, end_line, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.content,
                        chunk.file_type,
                        chunk.start_line,
                        chunk.end_line,
                        json.dumps(chunk.embedding).encode('utf-8')
                    ))
                    saved_count += 1
        
        # 기존 데이터의 file_type이 NULL인 경우 업데이트
        cursor.execute('''
            UPDATE code_chunks 
            SET file_type = CASE 
                WHEN file_path LIKE '%.html' OR file_path LIKE '%.htm' THEN 'html'
                WHEN file_path LIKE '%.js' OR file_path LIKE '%.jsx' THEN 'js'
                WHEN file_path LIKE '%.ts' OR file_path LIKE '%.tsx' THEN 'js'
                WHEN file_path LIKE '%.css' OR file_path LIKE '%.scss' OR file_path LIKE '%.sass' THEN 'css'
                WHEN file_path LIKE '%.py' THEN 'python'
                WHEN file_path LIKE '%.json' THEN 'json'
                WHEN file_path LIKE '%.xml' THEN 'xml'
                ELSE 'text'
            END
            WHERE file_type IS NULL OR file_type = ''
        ''')
        
        null_updated = cursor.rowcount
        if null_updated > 0:
            print(f"기존 데이터 {null_updated}개의 file_type이 업데이트되었습니다.")
        
        conn.commit()
        conn.close()
        print(f"데이터베이스 저장 완료: {saved_count}개 신규 저장, {updated_count}개 업데이트")
    
    def get_stats(self) -> Dict:
        """데이터베이스 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM code_chunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('SELECT file_type, COUNT(*) FROM code_chunks GROUP BY file_type')
        type_counts = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(DISTINCT file_path) FROM code_chunks')
        total_files = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_chunks': total_chunks,
            'total_files': total_files,
            'type_counts': type_counts
        }

class WebsiteCodeProcessor:
    def __init__(self, min_chunk_length: int = 30, ollama_url: str = "http://localhost:11434", model: str = "deepseek-coder-v2:16b"):
        self.chunker = CodeChunker(min_chunk_length)
        self.embedder = OllamaEmbedder(ollama_url, model)
        self.db = CodeEmbeddingDB()
    
    def process_website_code(self, website_path: str, file_extensions: List[str] = None, exclude_keywords: List[str] = None, exclude_dirs: List[str] = None):
        """웹사이트 코드 전체 처리"""
        if not os.path.exists(website_path):
            print(f"오류: 경로가 존재하지 않습니다: {website_path}")
            return []
        
        # 기본 제외 키워드 설정
        if exclude_keywords is None:
            exclude_keywords = []
        
        # 기본 제외 디렉토리 설정
        if exclude_dirs is None:
            exclude_dirs = ['node_modules', '__pycache__', 'dist', 'build', '.git', '.vscode', '.idea']
        
        all_chunks = []
        processed_files = 0
        skipped_files = 0
        excluded_files = 0
        
        print(f"웹사이트 코드 스캔 시작: {website_path}")
        if exclude_keywords:
            print(f"제외 키워드: {exclude_keywords}")
        if exclude_dirs:
            print(f"제외 디렉토리: {exclude_dirs}")
        
        # 웹사이트 디렉토리 스캔
        for root, dirs, files in os.walk(website_path):
            # 제외 디렉토리 필터링
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in exclude_dirs]
            
            for file in files:
                # 파일 확장자 확인
                if not self._is_code_file(file, file_extensions):
                    continue
                
                # 제외 키워드 확인
                if self._should_exclude_file(file, exclude_keywords):
                    excluded_files += 1
                    print(f"제외됨 (키워드 매칭): {file}")
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, website_path)
                
                # 경로에도 제외 키워드가 있는지 확인
                if self._should_exclude_file(relative_path, exclude_keywords):
                    excluded_files += 1
                    print(f"제외됨 (경로 키워드 매칭): {relative_path}")
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if len(content.strip()) < self.chunker.min_chunk_length:
                        print(f"건너뛰기 (파일이 너무 작음): {relative_path}")
                        skipped_files += 1
                        continue
                    
                    print(f"처리 중: {relative_path}")
                    chunks = self.chunker.chunk_file(relative_path, content)
                    all_chunks.extend(chunks)
                    processed_files += 1
                    
                except Exception as e:
                    print(f"파일 처리 실패 {relative_path}: {e}")
                    skipped_files += 1
        
        print(f"\n파일 처리 완료:")
        print(f"  - 처리된 파일: {processed_files}개")
        print(f"  - 건너뛴 파일: {skipped_files}개")
        print(f"  - 제외된 파일: {excluded_files}개")
        print(f"  - 총 청크: {len(all_chunks)}개")
        
        if not all_chunks:
            print("처리할 청크가 없습니다.")
            return []
        
        # 임베딩 생성
        print(f"\n임베딩 생성 시작...")
        embedded_chunks = self.embedder.batch_generate_embeddings(all_chunks)
        
        if not embedded_chunks:
            print("임베딩 생성에 실패했습니다.")
            return []
        
        # 데이터베이스에 저장
        print(f"\n데이터베이스 저장 중...")
        self.db.save_chunks(embedded_chunks)
        
        # 통계 출력
        stats = self.db.get_stats()
        print(f"\n=== 처리 완료 ===")
        print(f"총 청크 수: {stats['total_chunks']}")
        print(f"총 파일 수: {stats['total_files']}")
        print("파일 타입별 청크 수:")
        for file_type, count in stats['type_counts'].items():
            print(f"  - {file_type}: {count}개")
        
        return embedded_chunks
    
    def _should_exclude_file(self, file_path: str, exclude_keywords: List[str]) -> bool:
        """파일이 제외 키워드에 해당하는지 확인"""
        if not exclude_keywords:
            return False
        
        file_path_lower = file_path.lower()
        
        for keyword in exclude_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in file_path_lower:
                return True
        
        return False
    
    def _is_code_file(self, filename: str, allowed_extensions: List[str] = None) -> bool:
        """코드 파일인지 확인"""
        if allowed_extensions:
            return Path(filename).suffix.lower() in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in allowed_extensions]
        
        code_extensions = {
            '.html', '.htm', '.js', '.jsx', '.ts', '.tsx',
            '.css', '.scss', '.sass', '.py', '.java', '.cpp',
            '.c', '.h', '.php', '.rb', '.go', '.rs', '.vue',
            '.json', '.xml', '.yaml', '.yml'
        }
        
        return Path(filename).suffix.lower() in code_extensions

def main():
    parser = argparse.ArgumentParser(description='웹사이트 코드를 청킹하고 임베딩합니다.')
    parser.add_argument('website_path', help='웹사이트 코드 디렉토리 경로')
    parser.add_argument('--min-length', type=int, default=30, help='최소 청크 길이 (기본값: 30)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama 서버 URL')
    parser.add_argument('--model', default='deepseek-coder-v2:16b', help='사용할 모델명')
    parser.add_argument('--db-path', default='code_embeddings.db', help='데이터베이스 파일 경로')
    parser.add_argument('--extensions', nargs='+', help='처리할 파일 확장자 (예: .py .js .html)')
    parser.add_argument('--exclude-keywords', nargs='+', help='제외할 파일명 키워드 (예: test spec mock)')
    parser.add_argument('--exclude-dirs', nargs='+', help='제외할 디렉토리명 (기본: node_modules __pycache__ dist build .git .vscode .idea)')
    
    args = parser.parse_args()
    
    # 데이터베이스 경로 설정
    if args.db_path != 'code_embeddings.db':
        import builtins
        original_init = CodeEmbeddingDB.__init__
        def new_init(self, db_path=args.db_path):
            original_init(self, db_path)
        CodeEmbeddingDB.__init__ = new_init
    
    try:
        processor = WebsiteCodeProcessor(
            min_chunk_length=args.min_length,
            ollama_url=args.ollama_url,
            model=args.model
        )
        
        processor.process_website_code(
            website_path=args.website_path, 
            file_extensions=args.extensions,
            exclude_keywords=args.exclude_keywords,
            exclude_dirs=args.exclude_dirs
        )
        
    except KeyboardInterrupt:
        print("\n처리가 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()