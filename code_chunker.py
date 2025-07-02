import ast
import re
from typing import List, Dict
from bs4 import BeautifulSoup
import config

class CodeChunker:
    def __init__(self, max_chunk_size=1000):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_length = config.MIN_CHUNK_LENGTH
    
    def _filter_chunks_by_length(self, chunks: List[Dict]) -> List[Dict]:
        """길이가 최소 기준 이상인 청크만 필터링"""
        filtered_chunks = []
        excluded_count = 0
        
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if len(content) >= self.min_chunk_length:
                filtered_chunks.append(chunk)
            else:
                excluded_count += 1
        
        if excluded_count > 0:
            print(f"      📏 Excluded {excluded_count} chunks (length < {self.min_chunk_length})")
        
        return filtered_chunks
    
    def chunk_javascript(self, code: str) -> List[Dict]:
        """JavaScript 코드를 함수/클래스 단위로 청킹"""
        chunks = []
        
        # 함수 선언 패턴들
        patterns = [
            # 일반 함수
            r'function\s+(\w+)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            # 화살표 함수 (const/let/var)
            r'(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            # 화살표 함수 (단일 표현식)
            r'(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>[^;]+;?',
            # 객체 메서드
            r'(\w+)\s*:\s*function\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            # 클래스
            r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        ]
        
        lines = code.split('\n')
        
        for pattern in patterns:
            for match in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
                start_pos = match.start()
                start_line = code[:start_pos].count('\n') + 1
                
                chunk = {
                    'content': match.group(0).strip(),
                    'type': 'function',
                    'name': match.group(1) if match.groups() else 'anonymous',
                    'start_line': start_line
                }
                chunks.append(chunk)
        
        # 함수로 분류되지 않은 나머지 코드도 처리
        if not chunks:
            chunks = self.chunk_by_lines(code)
        
        return self._filter_chunks_by_length(chunks)
    
    def chunk_python(self, code: str) -> List[Dict]:
        """Python 코드를 함수/클래스 단위로 청킹"""
        try:
            tree = ast.parse(code)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    lines = code.split('\n')
                    content = '\n'.join(lines[start_line-1:end_line]).strip()
                    
                    if content:  # 빈 내용 제외
                        chunk = {
                            'content': content,
                            'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                            'name': node.name,
                            'start_line': start_line,
                            'end_line': end_line
                        }
                        chunks.append(chunk)
            
            # AST 파싱으로 찾지 못한 경우 라인 기반으로 처리
            if not chunks:
                chunks = self.chunk_by_lines(code)
            
            return self._filter_chunks_by_length(chunks)
            
        except SyntaxError as e:
            print(f"      ⚠️ Python syntax error, falling back to line-based chunking: {str(e)}")
            return self._filter_chunks_by_length(self.chunk_by_lines(code))
        except Exception as e:
            print(f"      ⚠️ Error parsing Python code: {str(e)}")
            return self._filter_chunks_by_length(self.chunk_by_lines(code))
    
    def chunk_html(self, code: str) -> List[Dict]:
        """HTML을 태그/섹션 단위로 청킹"""
        try:
            soup = BeautifulSoup(code, 'html.parser')
            chunks = []
            
            # 주요 섹션들을 찾아서 청킹
            important_tags = ['div', 'section', 'article', 'main', 'aside', 'header', 'footer', 'nav']
            
            for tag_name in important_tags:
                for tag in soup.find_all(tag_name):
                    # ID나 클래스가 있는 의미있는 태그만
                    if tag.get('class') or tag.get('id'):
                        content = str(tag).strip()
                        if content:
                            chunk = {
                                'content': content,
                                'type': 'html_section',
                                'tag': tag.name,
                                'classes': ' '.join(tag.get('class', [])),
                                'id': tag.get('id', ''),
                                'name': f"{tag.name}#{tag.get('id', '')}" if tag.get('id') else f"{tag.name}.{' '.join(tag.get('class', []))}"
                            }
                            chunks.append(chunk)
            
            # script와 style 태그도 별도로 처리
            for script in soup.find_all('script'):
                if script.string and script.string.strip():
                    chunk = {
                        'content': script.string.strip(),
                        'type': 'javascript',
                        'name': 'inline_script'
                    }
                    chunks.append(chunk)
            
            for style in soup.find_all('style'):
                if style.string and style.string.strip():
                    chunk = {
                        'content': style.string.strip(),
                        'type': 'css',
                        'name': 'inline_style'
                    }
                    chunks.append(chunk)
            
            # 의미있는 청크가 없으면 라인 기반으로 처리
            if not chunks:
                chunks = self.chunk_by_lines(code)
            
            return self._filter_chunks_by_length(chunks)
            
        except Exception as e:
            print(f"      ⚠️ Error parsing HTML: {str(e)}")
            return self._filter_chunks_by_length(self.chunk_by_lines(code))
    
    def chunk_css(self, code: str) -> List[Dict]:
        """CSS를 규칙/선택자 단위로 청킹"""
        chunks = []
        
        # CSS 규칙 패턴 (선택자 { ... })
        css_rule_pattern = r'([^{}]+)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        
        for match in re.finditer(css_rule_pattern, code, re.MULTILINE | re.DOTALL):
            selector = match.group(1).strip()
            rules = match.group(2).strip()
            
            if selector and rules:
                content = f"{selector} {{\n{rules}\n}}".strip()
                chunk = {
                    'content': content,
                    'type': 'css_rule',
                    'name': selector.split(',')[0].strip(),  # 첫 번째 선택자만 이름으로
                    'selector': selector
                }
                chunks.append(chunk)
        
        if not chunks:
            chunks = self.chunk_by_lines(code)
        
        return self._filter_chunks_by_length(chunks)
    
    def chunk_json(self, code: str) -> List[Dict]:
        """JSON을 의미있는 객체 단위로 청킹"""
        try:
            import json
            data = json.loads(code)
            chunks = []
            
            if isinstance(data, dict):
                # 최상위 키들을 기준으로 청킹
                for key, value in data.items():
                    content = json.dumps({key: value}, indent=2).strip()
                    chunk = {
                        'content': content,
                        'type': 'json_object',
                        'name': str(key),
                        'key': key
                    }
                    chunks.append(chunk)
            
            elif isinstance(data, list) and len(data) > 0:
                # 배열의 각 항목을 청킹
                for i, item in enumerate(data):
                    content = json.dumps(item, indent=2).strip()
                    chunk = {
                        'content': content,
                        'type': 'json_array_item',
                        'name': f'item_{i}',
                        'index': i
                    }
                    chunks.append(chunk)
            
            if not chunks:
                chunks = [{
                    'content': code.strip(),
                    'type': 'json_document',
                    'name': 'root'
                }]
            
            return self._filter_chunks_by_length(chunks)
            
        except json.JSONDecodeError:
            print("      ⚠️ Invalid JSON, treating as plain text")
            return self._filter_chunks_by_length(self.chunk_by_lines(code))
    
    def chunk_by_lines(self, code: str, overlap=50) -> List[Dict]:
        """라인 기반 청킹 (fallback)"""
        lines = code.split('\n')
        chunks = []
        
        # 빈 줄이 많은 경우 제거
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < len(lines) * 0.3:  # 70% 이상이 빈 줄이면
            lines = non_empty_lines
        
        for i in range(0, len(lines), self.max_chunk_size - overlap):
            chunk_lines = lines[i:i + self.max_chunk_size]
            content = '\n'.join(chunk_lines).strip()
            
            if content:  # 빈 내용 제외
                chunk = {
                    'content': content,
                    'type': 'code_block',
                    'start_line': i + 1,
                    'end_line': min(i + self.max_chunk_size, len(lines)),
                    'name': f'block_{i//self.max_chunk_size + 1}'
                }
                chunks.append(chunk)
        
        return chunks  # 이미 빈 내용은 제외했으므로 길이 필터링만 적용
    
    def chunk_code(self, code: str, file_extension: str) -> List[Dict]:
        """파일 확장자에 따른 코드 청킹"""
        if not code.strip():
            print("      ⚠️ Empty file content")
            return []
        
        ext = file_extension.lower()
        
        print(f"      📦 Chunking {ext} file...")
        
        if ext == '.py':
            chunks = self.chunk_python(code)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            chunks = self.chunk_javascript(code)
        elif ext in ['.html', '.htm']:
            chunks = self.chunk_html(code)
        elif ext in ['.css', '.scss', '.sass']:
            chunks = self.chunk_css(code)
        elif ext == '.json':
            chunks = self.chunk_json(code)
        else:
            chunks = self.chunk_by_lines(code)
        
        # 최종 길이 필터링
        final_chunks = self._filter_chunks_by_length(chunks)
        
        print(f"      ✅ Created {len(final_chunks)} valid chunks")
        
        return final_chunks