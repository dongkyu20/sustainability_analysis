import sqlite3
import json
from typing import List, Dict, Optional
import requests
import argparse
import os
from dataclasses import dataclass
import math

@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    content: str
    file_type: str
    start_line: int
    end_line: int
    similarity: float
    
    def __str__(self):
        return f"[{self.file_type}] {self.file_path}:{self.start_line}-{self.end_line} (유사도: {self.similarity:.4f})"

class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:32b"):
        self.base_url = base_url
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """Ollama API를 사용하여 임베딩 생성"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [])
        
        except requests.RequestException as e:
            print(f"임베딩 생성 실패: {e}")
            return []

class CodeSearchEngine:
    def __init__(self, db_path: str = "code_embeddings.db", ollama_url: str = "http://localhost:11434", model: str = "deepseek-r1:32b"):
        self.db_path = db_path
        self.embedder = OllamaEmbedder(ollama_url, model)
        self._check_database()
    
    def _check_database(self):
        """데이터베이스 존재 및 데이터 확인"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"데이터베이스 파일이 존재하지 않습니다: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM code_chunks')
            count = cursor.fetchone()[0]
            if count == 0:
                print("경고: 데이터베이스에 저장된 청크가 없습니다.")
            else:
                print(f"데이터베이스 연결 성공: {count}개 청크 확인")
        except sqlite3.Error as e:
            raise RuntimeError(f"데이터베이스 오류: {e}")
        finally:
            conn.close()
    
    def search(self, query: str, limit: int = 10, file_type: str = None, file_path_filter: str = None) -> List[SearchResult]:
        """코드 검색"""
        print(f"검색 쿼리: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.generate_embedding(query)
        if not query_embedding:
            print("쿼리 임베딩 생성에 실패했습니다.")
            return []
        
        print(f"쿼리 임베딩 생성 완료 (차원: {len(query_embedding)})")
        
        # 데이터베이스에서 검색
        results = self._search_similar_chunks(query_embedding, limit, file_type, file_path_filter)
        
        print(f"검색 완료: {len(results)}개 결과")
        return results
    
    def _search_similar_chunks(self, query_embedding: List[float], limit: int, file_type: str = None, file_path_filter: str = None) -> List[SearchResult]:
        """유사한 청크 검색 (코사인 유사도 기반)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 필터 조건 구성
        where_conditions = ["embedding IS NOT NULL"]
        params = []
        
        if file_type:
            where_conditions.append("file_type = ?")
            params.append(file_type)
        
        if file_path_filter:
            where_conditions.append("file_path LIKE ?")
            params.append(f"%{file_path_filter}%")
        
        where_clause = " AND ".join(where_conditions)
        
        cursor.execute(f'''
            SELECT chunk_id, file_path, content, file_type, start_line, end_line, embedding
            FROM code_chunks 
            WHERE {where_clause}
        ''', params)
        
        results = []
        processed_count = 0
        
        for row in cursor.fetchall():
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  처리 중... {processed_count}개")
            
            try:
                chunk_embedding = json.loads(row[6].decode('utf-8'))
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                results.append(SearchResult(
                    chunk_id=row[0],
                    file_path=row[1],
                    content=row[2],
                    file_type=row[3],
                    start_line=row[4],
                    end_line=row[5],
                    similarity=similarity
                ))
            except (json.JSONDecodeError, TypeError) as e:
                continue
        
        conn.close()
        
        # 유사도 순으로 정렬하여 상위 결과 반환
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """특정 청크 ID로 청크 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, file_path, content, file_type, start_line, end_line
            FROM code_chunks WHERE chunk_id = ?
        ''', (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return SearchResult(
                chunk_id=row[0],
                file_path=row[1],
                content=row[2],
                file_type=row[3],
                start_line=row[4],
                end_line=row[5],
                similarity=1.0
            )
        return None
    
    def get_chunks_by_file(self, file_path: str) -> List[SearchResult]:
        """특정 파일의 모든 청크 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, file_path, content, file_type, start_line, end_line
            FROM code_chunks WHERE file_path = ?
            ORDER BY start_line
        ''', (file_path,))
        
        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                chunk_id=row[0],
                file_path=row[1],
                content=row[2],
                file_type=row[3],
                start_line=row[4],
                end_line=row[5],
                similarity=1.0
            ))
        
        conn.close()
        return results
    
    def get_file_list(self) -> List[Dict]:
        """데이터베이스에 있는 파일 목록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path, file_type, COUNT(*) as chunk_count, 
                   MIN(start_line) as first_line, MAX(end_line) as last_line
            FROM code_chunks 
            GROUP BY file_path, file_type
            ORDER BY file_path
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'file_path': row[0],
                'file_type': row[1],
                'chunk_count': row[2],
                'line_range': f"{row[3]}-{row[4]}"
            })
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        """검색 데이터베이스 통계"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM code_chunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('SELECT file_type, COUNT(*) FROM code_chunks GROUP BY file_type ORDER BY COUNT(*) DESC')
        type_counts = cursor.fetchall()
        
        cursor.execute('SELECT COUNT(DISTINCT file_path) FROM code_chunks')
        total_files = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(LENGTH(content)) FROM code_chunks')
        avg_chunk_size = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_chunks': total_chunks,
            'total_files': total_files,
            'avg_chunk_size': int(avg_chunk_size) if avg_chunk_size else 0,
            'type_counts': type_counts
        }

class InteractiveSearchCLI:
    def __init__(self, search_engine: CodeSearchEngine):
        self.search_engine = search_engine
    
    def run(self):
        """대화형 검색 CLI 실행"""
        print("=== 코드 검색 엔진 ===")
        print("명령어:")
        print("  search <쿼리> [--limit N] [--type TYPE] [--file FILE]")
        print("  files - 파일 목록 보기")
        print("  stats - 통계 보기")
        print("  chunk <chunk_id> - 특정 청크 보기")
        print("  file <file_path> - 파일의 모든 청크 보기")
        print("  quit - 종료")
        print()
        
        # 초기 통계 표시
        stats = self.search_engine.get_stats()
        print(f"데이터베이스 정보: {stats['total_files']}개 파일, {stats['total_chunks']}개 청크")
        print()
        
        while True:
            try:
                command = input("검색> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("검색 엔진을 종료합니다.")
                    break
                
                self._process_command(command)
                
            except KeyboardInterrupt:
                print("\n검색 엔진을 종료합니다.")
                break
            except Exception as e:
                print(f"오류: {e}")
    
    def _process_command(self, command: str):
        """명령어 처리"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == 'search' and len(parts) > 1:
            self._handle_search(parts[1:])
        elif cmd == 'files':
            self._handle_files()
        elif cmd == 'stats':
            self._handle_stats()
        elif cmd == 'chunk' and len(parts) > 1:
            self._handle_chunk(parts[1])
        elif cmd == 'file' and len(parts) > 1:
            self._handle_file(' '.join(parts[1:]))
        else:
            print("알 수 없는 명령어입니다. 도움말을 참조하세요.")
    
    def _handle_search(self, args: List[str]):
        """검색 명령어 처리"""
        # 인자 파싱
        query_parts = []
        limit = 10
        file_type = None
        file_filter = None
        
        i = 0
        while i < len(args):
            if args[i] == '--limit' and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            elif args[i] == '--type' and i + 1 < len(args):
                file_type = args[i + 1]
                i += 2
            elif args[i] == '--file' and i + 1 < len(args):
                file_filter = args[i + 1]
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        
        if not query_parts:
            print("검색 쿼리를 입력해주세요.")
            return
        
        query = ' '.join(query_parts)
        results = self.search_engine.search(query, limit, file_type, file_filter)
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        print(f"\n검색 결과 ({len(results)}개):")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            
            # 코드 미리보기 (처음 3줄)
            preview_lines = result.content.split('\n')[:3]
            for line in preview_lines:
                print(f"   {line}")
            
            if len(result.content.split('\n')) > 3:
                print("   ...")
            print()
    
    def _handle_files(self):
        """파일 목록 명령어 처리"""
        files = self.search_engine.get_file_list()
        
        if not files:
            print("파일이 없습니다.")
            return
        
        print(f"\n파일 목록 ({len(files)}개):")
        print("-" * 80)
        print(f"{'파일 경로':<50} {'타입':<10} {'청크 수':<8} {'라인 범위'}")
        print("-" * 80)
        
        for file_info in files:
            print(f"{file_info['file_path']:<50} {file_info['file_type']:<10} {file_info['chunk_count']:<8} {file_info['line_range']}")
    
    def _handle_stats(self):
        """통계 명령어 처리"""
        stats = self.search_engine.get_stats()
        
        print(f"\n=== 데이터베이스 통계 ===")
        print(f"총 파일 수: {stats['total_files']:,}")
        print(f"총 청크 수: {stats['total_chunks']:,}")
        print(f"평균 청크 크기: {stats['avg_chunk_size']:,} 문자")
        print(f"\n파일 타입별 청크 수:")
        
        for file_type, count in stats['type_counts']:
            percentage = (count / stats['total_chunks']) * 100
            print(f"  {file_type:<10}: {count:>6,}개 ({percentage:>5.1f}%)")
    
    def _handle_chunk(self, chunk_id: str):
        """청크 조회 명령어 처리"""
        result = self.search_engine.get_chunk_by_id(chunk_id)
        
        if not result:
            print(f"청크를 찾을 수 없습니다: {chunk_id}")
            return
        
        print(f"\n=== 청크 정보 ===")
        print(f"ID: {result.chunk_id}")
        print(f"파일: {result.file_path}")
        print(f"타입: {result.file_type}")
        print(f"라인: {result.start_line}-{result.end_line}")
        print(f"\n내용:")
        print("-" * 40)
        print(result.content)
        print("-" * 40)
    
    def _handle_file(self, file_path: str):
        """파일 청크 조회 명령어 처리"""
        results = self.search_engine.get_chunks_by_file(file_path)
        
        if not results:
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return
        
        print(f"\n파일: {file_path}")
        print(f"청크 수: {len(results)}개")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 라인 {result.start_line}-{result.end_line} ({result.file_type})")
            
            # 코드 미리보기 (처음 2줄)
            preview_lines = result.content.split('\n')[:2]
            for line in preview_lines:
                print(f"   {line}")
            
            if len(result.content.split('\n')) > 2:
                print("   ...")
            print()

def main():
    parser = argparse.ArgumentParser(description='코드 검색 엔진')
    parser.add_argument('--db-path', default='code_embeddings.db', help='데이터베이스 파일 경로')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama 서버 URL')
    parser.add_argument('--model', default='deepseek-r1:32b', help='사용할 모델명')
    parser.add_argument('--interactive', '-i', action='store_true', help='대화형 모드 실행')
    parser.add_argument('--query', '-q', help='검색 쿼리')
    parser.add_argument('--limit', '-l', type=int, default=10, help='결과 개수 제한')
    parser.add_argument('--type', '-t', help='파일 타입 필터')
    parser.add_argument('--file', '-f', help='파일 경로 필터')
    parser.add_argument('--stats', action='store_true', help='통계 출력')
    parser.add_argument('--files', action='store_true', help='파일 목록 출력')
    
    args = parser.parse_args()
    
    try:
        search_engine = CodeSearchEngine(args.db_path, args.ollama_url, args.model)
        
        if args.interactive:
            cli = InteractiveSearchCLI(search_engine)
            cli.run()
        elif args.query:
            results = search_engine.search(args.query, args.limit, args.type, args.file)
            
            if results:
                print(f"\n검색 결과 ({len(results)}개):")
                print("=" * 80)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result}")
                    print("-" * 40)
                    print(result.content)
                    print("-" * 40)
            else:
                print("검색 결과가 없습니다.")
        elif args.stats:
            stats = search_engine.get_stats()
            print(f"총 파일 수: {stats['total_files']:,}")
            print(f"총 청크 수: {stats['total_chunks']:,}")
            print(f"평균 청크 크기: {stats['avg_chunk_size']:,} 문자")
            print("\n파일 타입별 청크 수:")
            for file_type, count in stats['type_counts']:
                print(f"  {file_type}: {count:,}개")
        elif args.files:
            files = search_engine.get_file_list()
            print(f"파일 목록 ({len(files)}개):")
            for file_info in files:
                print(f"{file_info['file_path']} ({file_info['file_type']}) - {file_info['chunk_count']}개 청크")
        else:
            print("검색 쿼리나 옵션을 지정해주세요. --help를 참조하세요.")
    
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()