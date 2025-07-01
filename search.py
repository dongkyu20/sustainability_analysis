import sqlite3
import json
import numpy as np
from typing import List, Tuple, Any
from pathlib import Path
from code_embedder import CodeEmbedder
import config

class CodeSearcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedder = CodeEmbedder(method=config.EMBEDDING_METHOD)
    
    def search_similar_code(self, query: str, top_k: int = 5) -> List[Tuple[float, Any]]:
        """유사한 코드 청크 검색"""
        
        # 쿼리 임베딩 생성
        print(f"🔍 Generating embedding for query: '{query[:50]}...'")
        try:
            query_embedding = self.embedder.create_embedding(query)
            if not query_embedding or all(x == 0 for x in query_embedding):
                print("❌ Failed to generate query embedding")
                return []
        except Exception as e:
            print(f"❌ Error generating query embedding: {str(e)}")
            return []
        
        # 데이터베이스에서 모든 청크 가져오기
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM code_chunks WHERE embedding IS NOT NULL')
            results = cursor.fetchall()
            
            if not results:
                print("❌ No embeddings found in database")
                return []
            
            print(f"📊 Comparing with {len(results)} stored embeddings...")
            
        except sqlite3.Error as e:
            print(f"❌ Database error: {str(e)}")
            return []
        finally:
            conn.close()
        
        # 유사도 계산
        similarities = []
        valid_comparisons = 0
        
        for i, row in enumerate(results):
            try:
                # 저장된 임베딩 파싱
                stored_embedding_blob = row[4]  # embedding 컬럼
                if stored_embedding_blob:
                    stored_embedding = json.loads(stored_embedding_blob.decode())
                    
                    # 유효한 임베딩인지 확인
                    if stored_embedding and len(stored_embedding) > 0:
                        if not all(x == 0 for x in stored_embedding):
                            similarity = self.cosine_similarity(query_embedding, stored_embedding)
                            similarities.append((similarity, row))
                            valid_comparisons += 1
                
            except Exception as e:
                print(f"⚠️ Error processing embedding {i}: {str(e)}")
                continue
        
        print(f"✅ Completed {valid_comparisons} valid comparisons")
        
        if not similarities:
            print("❌ No valid similarities calculated")
            return []
        
        # 유사도 순으로 정렬하고 상위 k개 반환
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"🎯 Top similarity scores: {[f'{s[0]:.3f}' for s in top_results[:3]]}")
        
        return top_results
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        try:
            # numpy 배열로 변환
            a = np.array(vec1)
            b = np.array(vec2)
            
            # 벡터 길이가 다른 경우 처리
            if len(a) != len(b):
                min_len = min(len(a), len(b))
                a = a[:min_len]
                b = b[:min_len]
            
            # 영벡터 체크
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # 코사인 유사도 계산
            similarity = np.dot(a, b) / (norm_a * norm_b)
            
            # NaN이나 무한대 값 처리
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️ Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def search_by_file_type(self, file_extension: str, top_k: int = 10) -> List[Any]:
        """특정 파일 타입의 코드 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM code_chunks 
            WHERE file_path LIKE ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (f'%.{file_extension}', top_k))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def search_by_chunk_type(self, chunk_type: str, top_k: int = 10) -> List[Any]:
        """특정 청크 타입으로 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM code_chunks 
            WHERE chunk_type = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (chunk_type, top_k))
        
        results = cursor.fetchall()
        conn.close()
        
        return results