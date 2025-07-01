#!/usr/bin/env python3
"""
웹사이트 코드 임베딩 메인 실행 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 모듈 import
from code_chunker import CodeChunker
from code_embedder import CodeEmbedder
from code_indexer import CodeIndexer
from search import CodeSearcher
import config


def collect_code_files(directory: str):
    """웹사이트 디렉토리에서 코드 파일들을 수집"""
    directory = Path(directory)
    files = []
    
    for ext in config.SUPPORTED_EXTENSIONS:
        for file_path in directory.rglob(ext):
            # 제외할 디렉토리 체크
            if any(exclude_dir in str(file_path) for exclude_dir in config.EXCLUDE_DIRS):
                continue
            
            # 제외할 파일 체크
            if any(file_path.match(exclude_pattern) for exclude_pattern in config.EXCLUDE_FILES):
                continue
            
            files.append(str(file_path))
    
    return files


def process_website_code(website_directory: str, reset_db: bool = False):
    """웹사이트 코드 전체 처리 파이프라인"""
    
    print(f"🚀 Starting code processing for: {website_directory}")
    print(f"🔧 Using embedding method: {config.EMBEDDING_METHOD}")
    
    # 1. 디렉토리 존재 확인
    if not os.path.exists(website_directory):
        print(f"❌ Directory not found: {website_directory}")
        return False
    
    # 2. 파일 수집
    print("📁 Collecting code files...")
    files = collect_code_files(website_directory)
    print(f"✅ Found {len(files)} code files")
    
    if not files:
        print("❌ No code files found!")
        return False
    
    # 3. 객체 초기화
    print("🔧 Initializing components...")
    chunker = CodeChunker(max_chunk_size=config.MAX_CHUNK_SIZE)
    embedder = CodeEmbedder(method=config.EMBEDDING_METHOD)
    indexer = CodeIndexer(db_path=str(config.DATABASE_PATH))
    
    if reset_db:
        print("🗑️ Resetting database...")
        indexer.reset_database()
    
    # 4. 각 파일 처리
    total_chunks = 0
    processed_files = 0
    
    for i, file_path in enumerate(files, 1):
        try:
            print(f"📝 Processing ({i}/{len(files)}): {Path(file_path).name}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            if not code.strip():
                print(f"⚠️ Empty file, skipping: {file_path}")
                continue
            
            # 파일 확장자에 따른 청킹
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.py':
                chunks = chunker.chunk_python(code)
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                chunks = chunker.chunk_javascript(code)
            elif file_ext == '.html':
                chunks = chunker.chunk_html(code)
            else:
                chunks = chunker.chunk_by_lines(code)
            
            if not chunks:
                print(f"⚠️ No chunks created for: {file_path}")
                continue
            
            print(f"   📦 Created {len(chunks)} chunks")
            
            # 임베딩 생성
            print(f"   🧠 Generating embeddings...")
            chunks_with_embeddings = embedder.embed_chunks(chunks)
            
            # 저장
            indexer.save_chunks(chunks_with_embeddings, file_path)
            
            total_chunks += len(chunks)
            processed_files += 1
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            continue
    
    print(f"\n🎉 Processing completed!")
    print(f"   📊 Processed files: {processed_files}/{len(files)}")
    print(f"   📦 Total chunks: {total_chunks}")
    print(f"   💾 Database: {config.DATABASE_PATH}")
    
    return True


def interactive_search():
    """대화형 검색 모드"""
    print("\n🔍 Interactive Search Mode")
    print("Type 'quit' to exit")
    
    searcher = CodeSearcher(db_path=str(config.DATABASE_PATH))
    
    while True:
        try:
            query = input("\n📝 Enter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🔍 Searching...")
            results = searcher.search_similar_code(query, top_k=5)
            
            if not results:
                print("❌ No results found")
                continue
            
            print(f"\n📋 Found {len(results)} similar code chunks:")
            print("-" * 60)
            
            for i, (similarity, chunk_data) in enumerate(results, 1):
                file_path = chunk_data[1]
                chunk_type = chunk_data[2]
                content = chunk_data[3][:200] + "..." if len(chunk_data[3]) > 200 else chunk_data[3]
                
                print(f"\n{i}. Similarity: {similarity:.3f}")
                print(f"   File: {Path(file_path).name}")
                print(f"   Type: {chunk_type}")
                print(f"   Content: {content}")
                print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Website Code Embedding Tool")
    parser.add_argument("command", choices=["process", "search", "info"], 
                       help="Command to execute")
    parser.add_argument("--directory", "-d", type=str, 
                       help="Website directory to process")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset database before processing")
    parser.add_argument("--query", "-q", type=str, 
                       help="Search query (for search command)")
    
    args = parser.parse_args()
    
    if args.command == "process":
        if not args.directory:
            print("❌ --directory is required for process command")
            sys.exit(1)
        
        success = process_website_code(args.directory, args.reset)
        sys.exit(0 if success else 1)
    
    elif args.command == "search":
        if args.query:
            # 단일 검색
            searcher = CodeSearcher(db_path=str(config.DATABASE_PATH))
            results = searcher.search_similar_code(args.query, top_k=5)
            
            for i, (similarity, chunk_data) in enumerate(results, 1):
                print(f"{i}. Similarity: {similarity:.3f} - {Path(chunk_data[1]).name}")
        else:
            # 대화형 검색
            interactive_search()
    
    elif args.command == "info":
        # 데이터베이스 정보 출력
        indexer = CodeIndexer(db_path=str(config.DATABASE_PATH))
        stats = indexer.get_database_stats()
        
        print(f"📊 Database Statistics:")
        print(f"   📁 Total files: {stats['total_files']}")
        print(f"   📦 Total chunks: {stats['total_chunks']}")
        print(f"   📈 Chunk types: {stats['chunk_types']}")
        print(f"   💾 Database size: {stats['db_size']} MB")


if __name__ == "__main__":
    main()