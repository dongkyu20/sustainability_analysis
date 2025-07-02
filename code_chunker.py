import ast
import re
import config
from typing import List, Dict

class CodeChunker:
    def __init__(self, max_chunk_size=1000):
        self.max_chunk_size = max_chunk_size
    
    def chunk_javascript(self, code: str) -> List[Dict]:
        """JavaScript 코드를 함수/클래스 단위로 청킹"""
        chunks = []
        
        # 함수 선언 패턴
        function_pattern = r'(function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}|const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\})'
        
        functions = re.finditer(function_pattern, code, re.MULTILINE | re.DOTALL)
        
        for match in functions:
            content = match.group(0)
            if len(content.strip()) <= 30:
                continue
            chunk = {
                'content': content,
                'type': 'function',
                'start_line': code[:match.start()].count('\n') + 1
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_python(self, code: str) -> List[Dict]:
        """Python 코드를 함수/클래스 단위로 청킹"""
        try:
            tree = ast.parse(code)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    
                    lines = code.split('\n')
                    content = '\n'.join(lines[start_line-1:end_line])

                    if len(content.strip()) <= 30:
                        continue

                    chunk = {
                        'content': content,
                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                        'name': node.name,
                        'start_line': start_line
                    }
                    chunks.append(chunk)
            
            return chunks
        except:
            return self.chunk_by_lines(code)
    
    def chunk_html(self, code: str) -> List[Dict]:
        """HTML을 태그/섹션 단위로 청킹"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(code, 'html.parser')
        chunks = []
        
        # 주요 섹션들을 찾아서 청킹
        for tag in soup.find_all(['div', 'section', 'article', 'main', 'aside']):
            content = str(tag)
            if len(content.strip()) <= 30:
                continue
            if tag.get('class') or tag.get('id'):
                chunk = {
                    'content': content,
                    'type': 'html_section',
                    'tag': tag.name,
                    'classes': tag.get('class', []),
                    'id': tag.get('id', '')
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_lines(self, code: str, overlap=50) -> List[Dict]:
        """라인 기반 청킹 (fallback)"""
        lines = code.split('\n')
        chunks = []
        
        for i in range(0, len(lines), self.max_chunk_size - overlap):
            chunk_lines = lines[i:i + self.max_chunk_size]
            content = '\n'.join(chunk_lines)
            if len(content.strip()) <= 30:
                continue
            chunk = {
                'content': content,
                'type': 'code_block',
                'start_line': i + 1,
                'end_line': min(i + self.max_chunk_size, len(lines))
            }
            chunks.append(chunk)
        
        return chunks