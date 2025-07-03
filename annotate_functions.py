#!/usr/bin/env python3
"""annotate_functions.py
웹사이트 코드베이스의 각 함수(또는 메서드)에 대해, 함수 본문을 LLM에 보내 기능과 역할을 요약한 한글 주석을 자동으로 삽입한다.

현재 지원 언어: Python(.py), JavaScript(.js, .jsx, .ts, .tsx)
- Python: 함수/클래스 메서드 정의 바로 위에 `# Description:` 형태의 주석 추가(이미 존재 시 건너뜀)
- JavaScript: 함수 정의 앞에 `/** Description: ... */` 블록 주석 추가

LLM 백엔드로 Ollama API를 사용한다. (POST /api/generate, stream=False)
사용 모델은 기본 deepseek-coder-v2:16b (CLI 옵션으로 변경 가능)

주의: 매우 큰 코드베이스에서 호출이 많아질 수 있으니 적절한 `--max-tokens`과 sleep 값을 조정할 것.
"""
from __future__ import annotations

import os
import re
import ast
import argparse
import textwrap
from pathlib import Path
from typing import List, Tuple
import requests
import time

SUPPORTED_PY_EXT = {".py"}
SUPPORTED_JS_EXT = {".js", ".jsx", ".ts", ".tsx"}
SUPPORTED_EXT = SUPPORTED_PY_EXT | SUPPORTED_JS_EXT

class OllamaLLM:
    """간단한 Ollama generate 래퍼"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-coder-v2:16b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._test_connection()

    def _test_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            print("✅ Ollama 서버 연결 확인 완료")
        except Exception as e:
            raise RuntimeError(f"Ollama 서버에 연결할 수 없습니다: {e}")

    def generate_comment(self, code_snippet: str, language: str, max_retries: int = 3) -> str:
        prompt = textwrap.dedent(
            f"""
            Write a concise 1–3 line English comment that clearly explains the purpose and behavior of the following {language} function.
            Do not start the comment with 'Description:' or any other label; output only the explanation sentences.
            Function code:
            ```{language}\n{code_snippet}\n```
            Return the comment in English.
            """
        ).strip()

        payload = {"model": self.model, "prompt": prompt, "stream": False}

        for attempt in range(max_retries):
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    if "response" in data and data["response"].strip():
                        return data["response"].strip()
                print(f"⚠️ Ollama 응답 오류 (status={r.status_code}), 재시도...")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 네트워크 오류: {e}, 재시도...")
            time.sleep(2 ** attempt)
        print("❌ 주석 생성 실패, 기본 빈 문자열 반환")
        return ""

class FunctionAnnotator:
    def __init__(self, llm: OllamaLLM, sleep: float = 0.1, output_dir: Path | None = None):
        self.llm = llm
        self.sleep = sleep  # API 부하 방지용
        self.output_dir = output_dir.resolve() if output_dir else None
        self.root_path: Path | None = None  # annotate_path 호출 시 설정

    # --------------------- 파이썬 ---------------------
    def _annotate_python(self, path: Path) -> None:
        code = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(code)
        except SyntaxError:
            print(f"⚠️ 파싱 실패(SyntaxError): {path}")
            return

        lines = code.splitlines()
        insertions: List[Tuple[int, List[str]]] = []  # (line_idx, comment_lines)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 이미 docstring/comment가 있는지 검사(한 줄 위에 '# Description:' 존재 여부)
                fn_start = node.lineno - 1  # 0-index
                if fn_start > 0 and lines[fn_start - 1].lstrip().startswith("# Description:"):
                    continue  # 이미 주석 존재

                # 함수 소스 추출
                end_line = getattr(node, "end_lineno", fn_start + 1) - 1
                snippet = "\n".join(lines[fn_start:end_line + 1])

                comment_text = self.llm.generate_comment(snippet, "python")
                if not comment_text:
                    continue

                indent = len(lines[fn_start]) - len(lines[fn_start].lstrip())
                indent_spaces = " " * indent
                comment_lines = [f"{indent_spaces}# Description: {comment_text}"]
                insertions.append((fn_start, comment_lines))
                time.sleep(self.sleep)

        # 역순으로 삽입하여 라인 번호 무효화 방지
        if insertions:
            for idx, comment_lines in sorted(insertions, key=lambda x: x[0], reverse=True):
                lines[idx:idx] = comment_lines  # 앞에 삽입
            self._save_annotated_file(path, "\n".join(lines))
            print(f"📝 주석 추가 완료: {path} ({len(insertions)}개 함수)")

    # --------------------- 자바스크립트 ---------------------
    _JS_FUNC_PATTERN = re.compile(
        r"""(
            (?:function\s+([\w$]+)\s*\([^)]*\)\s*\{) |                       # function foo(){}
            (?:const|let|var)\s+([\w$]+)\s*=\s*\([^)]*\)\s*=>\s*\{          # const foo = () => {
        )""",
        re.VERBOSE,
    )

    def _annotate_js(self, path: Path) -> None:
        code = path.read_text(encoding="utf-8", errors="ignore")
        lines = code.splitlines()
        insertions: List[Tuple[int, List[str]]] = []

        for match in self._JS_FUNC_PATTERN.finditer(code):
            start_pos = match.start()
            start_line = code.count("\n", 0, start_pos)
            # 이미 /** Description: */ 블록이 바로 위에 있으면 건너뜀
            if start_line > 0 and lines[start_line - 1].lstrip().startswith("/** Description:"):
                continue

            # 함수 블록 추출(여기서는 시작 라인만 활용, 간단하게 앞뒤 40줄 한정)
            snippet = "\n".join(lines[max(0, start_line - 1): min(len(lines), start_line + 40)])
            comment_text = self.llm.generate_comment(snippet, "javascript")
            if not comment_text:
                continue

            indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            indent_spaces = " " * indent
            comment_block = textwrap.indent(f"/** Description: {comment_text} */", indent_spaces)
            insertions.append((start_line, [comment_block]))
            time.sleep(self.sleep)

        if insertions:
            for idx, comment_lines in sorted(insertions, key=lambda x: x[0], reverse=True):
                lines[idx:idx] = comment_lines
            self._save_annotated_file(path, "\n".join(lines))
            print(f"📝 주석 추가 완료: {path} ({len(insertions)}개 함수)")

    # --------------------- 진입점 ---------------------
    def annotate_path(self, target_path: Path, exts: List[str] | None = None) -> None:
        target_path = target_path.resolve()
        # 주석 삽입 대상 루트 디렉토리 기록(상대 경로 계산용)
        self.root_path = target_path if target_path.is_dir() else target_path.parent

        if target_path.is_file():
            self._annotate_file(target_path, exts)
        else:
            for root, _, files in os.walk(target_path):
                for fname in files:
                    self._annotate_file(Path(root) / fname, exts)

    def _save_annotated_file(self, original_path: Path, new_content: str) -> None:
        """주석이 삽입된 코드를 저장. output_dir 지정 시 동일한 상대 경로로 복사 저장한다."""
        if self.output_dir and self.root_path:
            rel = original_path.relative_to(self.root_path)
            dest_path = self.output_dir / rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(new_content, encoding="utf-8")
        else:
            # in-place 저장(기존 동작)
            original_path.write_text(new_content, encoding="utf-8")

    def _annotate_file(self, path: Path, exts: List[str] | None) -> None:
        if exts and path.suffix not in exts:
            return
        if path.suffix not in SUPPORTED_EXT:
            return
        try:
            if path.suffix in SUPPORTED_PY_EXT:
                self._annotate_python(path)
            elif path.suffix in SUPPORTED_JS_EXT:
                self._annotate_js(path)
        except Exception as e:
            print(f"⚠️ {path} 처리 중 오류: {e}")

# ------------------------- CLI -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="웹사이트 코드 함수 주석 자동화 스크립트")
    parser.add_argument("path", help="대상 디렉토리 또는 파일 경로")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 서버 URL")
    parser.add_argument("--model", default="deepseek-coder-v2:16b", help="사용할 모델명")
    parser.add_argument("--extensions", nargs="*", help="처리할 파일 확장자 목록 예) .py .js")
    parser.add_argument("--sleep", type=float, default=0.1, help="API 호출 간 대기시간(초)")
    parser.add_argument("--output-dir", help="주석이 삽입된 파일을 저장할 디렉토리(미지정 시 원본 덮어쓰기)")
    args = parser.parse_args()

    exts = None
    if args.extensions:
        exts = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions]

    llm = OllamaLLM(args.ollama_url, args.model)
    annotator = FunctionAnnotator(llm, sleep=args.sleep, output_dir=Path(args.output_dir) if args.output_dir else None)
    annotator.annotate_path(Path(args.path), exts)

if __name__ == "__main__":
    main()
