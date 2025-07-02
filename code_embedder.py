import requests
import json
import time
from typing import List, Dict
import config

class CodeEmbedder:
    def __init__(self, method='ollama'):
        self.method = method
        
        if method == 'ollama':
            self.base_url = config.OLLAMA_BASE_URL
            self.model = config.OLLAMA_EMBEDDING_MODEL
            self._check_ollama_connection()
            self._ensure_model_available()
        
        elif method == 'sentence_transformer':
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence Transformer model loaded")
            except ImportError:
                print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
                
        elif method == 'openai':
            try:
                import openai
                if config.OPENAI_API_KEY:
                    openai.api_key = config.OPENAI_API_KEY
                    print("✅ OpenAI API key configured")
                else:
                    print("❌ OpenAI API key not found")
                    raise ValueError("OpenAI API key required")
            except ImportError:
                print("❌ openai package not installed. Run: pip install openai")
                raise

    def _check_ollama_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server connected")
                return True
            else:
                print(f"❌ Ollama server responded with status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Ollama server not found. Please make sure Ollama is running:")
            print("   1. Install Ollama: https://ollama.ai/")
            print("   2. Run: ollama serve")
            print(f"   3. Pull embedding model: ollama pull {self.model}")
            raise ConnectionError("Ollama server not available")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {str(e)}")
            raise

    def _ensure_model_available(self):
        """임베딩 모델이 사용 가능한지 확인하고 필요시 다운로드"""
        try:
            # 설치된 모델 목록 확인
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'].split(':')[0] for model in models]
                
                if self.model in model_names:
                    print(f"✅ Model {self.model} is available")
                    return
                else:
                    print(f"📥 Model {self.model} not found. Attempting to pull...")
                    self._pull_model()
            else:
                print(f"❌ Failed to check available models: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error checking model availability: {str(e)}")
            raise

    def _pull_model(self):
        """Ollama 모델 다운로드"""
        try:
            print(f"📥 Pulling {self.model} model... This may take a while.")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=300  # 5분 타임아웃
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'status' in data:
                                print(f"   📥 {data['status']}")
                            if data.get('status') == 'success':
                                print(f"✅ Model {self.model} pulled successfully")
                                return
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"❌ Failed to pull model: {response.status_code}")
                raise Exception(f"Model pull failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error pulling model: {str(e)}")
            print(f"💡 Try manually: ollama pull {self.model}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        """텍스트에 대한 임베딩 생성"""
        
        if self.method == 'ollama':
            return self._create_ollama_embedding(text)
        
        elif self.method == 'sentence_transformer':
            if len(text) > 5000:
                text = text[:5000]
            embedding = self.model.encode(text)
            return embedding.tolist()
            
        elif self.method == 'openai':
            import openai
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']

    def _create_ollama_embedding(self, text: str, max_retries=3) -> List[float]:
        """Ollama를 통한 임베딩 생성 (재시도 로직 포함)"""
        
        # CodeLlama는 더 긴 컨텍스트를 처리할 수 있음 (최대 16K 토큰)
        if len(text) > 12000:  # CodeLlama를 위한 더 큰 제한
            text = text[:12000] + "..."
        
        for attempt in range(max_retries):
            try:
                # CodeLlama의 경우 embedding API 대신 generate API 사용할 수도 있음
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=60  # CodeLlama는 더 오래 걸릴 수 있음
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'embedding' in result:
                        return result['embedding']
                    else:
                        print(f"⚠️ No embedding in response: {result}")
                        
                elif response.status_code == 404:
                    print(f"❌ Model {self.model} not found. Please run: ollama pull {self.model}")
                    raise Exception(f"Model {self.model} not available")
                    
                else:
                    print(f"⚠️ Ollama API error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout on attempt {attempt + 1}/{max_retries} (CodeLlama can be slow)")
                if attempt < max_retries - 1:
                    time.sleep(5 + (2 ** attempt))  # 더 긴 대기시간
                    continue
                    
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Connection error on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                print(f"⚠️ Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        # 모든 재시도 실패시 기본값 반환
        print("❌ All embedding attempts failed, using zero vector")
        return [0.0] * 4096  # CodeLlama의 일반적인 임베딩 차원

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """청크들에 대한 임베딩 생성"""
        print(f"   🧠 Generating embeddings using {self.method}...")
        
        successful_embeddings = 0
        failed_embeddings = 0
        
        for i, chunk in enumerate(chunks):
            if i % 5 == 0:  # 진행상황 표시 (5개마다)
                print(f"      Progress: {i+1}/{len(chunks)} (Success: {successful_embeddings}, Failed: {failed_embeddings})")
            
            # 코드에 메타데이터 추가해서 임베딩
            text_to_embed = self._prepare_text_for_embedding(chunk)
            
            try:
                embedding = self.create_embedding(text_to_embed)
                
                # 임베딩이 유효한지 확인
                if embedding and len(embedding) > 0 and not all(x == 0 for x in embedding):
                    chunk['embedding'] = embedding
                    chunk['embedding_text'] = text_to_embed
                    successful_embeddings += 1
                else:
                    print(f"      ⚠️ Invalid embedding for chunk {i+1}")
                    chunk['embedding'] = [0.0] * 1024
                    chunk['embedding_text'] = text_to_embed
                    failed_embeddings += 1
                    
            except Exception as e:
                print(f"      ❌ Error embedding chunk {i+1}: {str(e)}")
                chunk['embedding'] = [0.0] * 1024
                chunk['embedding_text'] = text_to_embed
                failed_embeddings += 1
            
            # Ollama 서버 부하 방지를 위한 짧은 지연
            if self.method == 'ollama' and i % 10 == 0:
                time.sleep(0.1)
        
        print(f"   ✅ Embedding completed: {successful_embeddings} success, {failed_embeddings} failed")
        return chunks

    def _prepare_text_for_embedding(self, chunk: Dict) -> str:
        """CodeLlama를 위한 텍스트 준비 (코드 컨텍스트 강화)"""
        context_info = []
        
        # CodeLlama는 코드 구조를 더 잘 이해하므로 더 상세한 컨텍스트 제공
        if chunk.get('type'):
            context_info.append(f"Code Type: {chunk['type']}")
        
        if chunk.get('name'):
            context_info.append(f"Function/Class Name: {chunk['name']}")
        
        if chunk.get('file_ext'):
            language_map = {
                '.py': 'Python', '.js': 'JavaScript', '.jsx': 'React JSX',
                '.ts': 'TypeScript', '.tsx': 'React TypeScript',
                '.html': 'HTML', '.css': 'CSS', '.json': 'JSON'
            }
            lang = language_map.get(chunk['file_ext'], chunk['file_ext'][1:].upper())
            context_info.append(f"Language: {lang}")
        
        # 추가 메타데이터
        if chunk.get('start_line'):
            context_info.append(f"Line: {chunk['start_line']}")
        
        context = " | ".join(context_info)
        content = chunk['content']
        
        # CodeLlama는 더 긴 컨텍스트를 처리할 수 있음
        if len(content) > 10000:
            content = content[:10000] + "..."
        
        # 코드 블록 형태로 포맷팅 (CodeLlama가 선호하는 형태)
        if context:
            return f"# {context}\n```\n{content}\n```"
        else:
            return f"```\n{content}\n```"