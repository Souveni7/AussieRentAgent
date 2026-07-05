"""
离线入库脚本:把 FAQ.md 切成 chunk → 调 Gemini embedding → 存成 NumPy 向量表。
只需跑一次;之后检索时直接加载 faq_index.npz,不再调 embedding。

运行前需在环境变量里设置:
  GEMINI_API_KEY   —— 你的 Google AI Studio 免费 key
运行:python ingest.py
"""

import os
import numpy as np
from google import genai
from google.genai import types

FAQ_PATH = "FAQ.md"
INDEX_PATH = "faq_index.npz"
MODEL = "gemini-embedding-001"


def load_chunks(path: str) -> list[str]:
    """把 FAQ.md 按 '## 标题' 切成若干 chunk,每个 chunk = 标题 + 正文。
    FAQ.md 已手工保持干净,这里不再做清洗,只按标题切分、跳过空行。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    chunks: list[str] = []
    current: list[str] | None = None   # None 表示还没遇到第一个 ## 标题
    for line in lines:
        if line.startswith("## "):          # 新 chunk 开始
            if current:
                chunks.append("\n".join(current))
            current = [line[3:].strip()]
        elif current is not None:
            text = line.strip()
            if text:
                current.append(text)

    if current:
        chunks.append("\n".join(current))
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """一次性批量把所有 chunk 送去 embedding,返回 (N, 维度) 的矩阵。
    task_type=RETRIEVAL_DOCUMENT 告诉模型"这是被检索的文档",
    检索时对用户问题要用 RETRIEVAL_QUERY,两者配对能提升检索质量。"""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = [e.values for e in resp.embeddings]
    return np.array(vectors, dtype=np.float32)


def main() -> None:
    chunks = load_chunks(FAQ_PATH)
    print(f"切出 {len(chunks)} 个 chunk,开始 embedding……")
    embeddings = embed(chunks)
    # 存成单文件:向量矩阵 + 对应原文,检索时一起加载
    np.savez(INDEX_PATH, embeddings=embeddings, texts=np.array(chunks))
    print(f"完成 → {INDEX_PATH}  向量矩阵 shape={embeddings.shape}")


if __name__ == "__main__":
    main()
