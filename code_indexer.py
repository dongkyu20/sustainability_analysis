import json
import sqlite3
from datetime import datetime

class CodeIndexer:
    def __init__(self, db_path='code_embeddings.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            chunk_type TEXT,
            content TEXT,
            embedding BLOB,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_chunks(self, chunks: List[Dict], file_path: str):
        """청크들을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk in chunks:
            embedding_blob = json.dumps(chunk['embedding']).encode()
            metadata = json.dumps({
                'start_line': chunk.get('start_line'),
                'end_line': chunk.get('end_line'),
                'name': chunk.get('name'),
                'classes': chunk.get('classes'),
                'id': chunk.get('id')
            })
            
            cursor.execute('''
            INSERT INTO code_chunks (file_path, chunk_type, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (file_path, chunk['type'], chunk['content'], embedding_blob, metadata))
        
        conn.commit()
        conn.close()

    def reset_database(self):
        """데이터베이스 초기화 (모든 데이터 삭제)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS code_chunks')
        conn.commit()
        conn.close()
        self.init_database()
        print("✅ Database reset completed")
    
    def get_database_stats(self):
        """데이터베이스 통계 정보 반환"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 총 청크 수
        cursor.execute('SELECT COUNT(*) FROM code_chunks')
        total_chunks = cursor.fetchone()[0]
        
        # 파일 수
        cursor.execute('SELECT COUNT(DISTINCT file_path) FROM code_chunks')
        total_files = cursor.fetchone()[0]
        
        # 청크 타입별 분포
        cursor.execute('SELECT chunk_type, COUNT(*) FROM code_chunks GROUP BY chunk_type')
        chunk_types = dict(cursor.fetchall())
        
        # 데이터베이스 크기
        import os
        db_size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        
        conn.close()
        
        return {
            'total_chunks': total_chunks,
            'total_files': total_files,
            'chunk_types': chunk_types,
            'db_size': round(db_size, 2)
        }