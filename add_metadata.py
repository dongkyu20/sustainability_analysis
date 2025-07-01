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