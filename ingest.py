"""
离线入库脚本:把 FAQ.md 切成 chunk → 调 embedding API → 存成 NumPy 向量表。
只需跑一次;之后检索时直接加载 faq_index.npz,不再调 embedding。

运行前需在环境变量里设置:
  OPENAI_API_KEY   —— 你的 key
  OPENAI_BASE_URL  —— 可选,用中转/兼容网关时才填
运行:python ingest.py
"""

import os
import re
import numpy as np
from openai import OpenAI

FAQ_PATH = "FAQ.md"
INDEX_PATH = "faq_index.npz"
MODEL = "text-embedding-3-small"


def load_chunks(path: str) -> list[str]:
    """把 FAQ.md 按 '## 标题' 切成若干 chunk,每个 chunk = 标题 + 正文。
    清洗:去掉 🆕 标记、TODO 行、<!-- --> 注释,这些是给人看的元信息,不该进向量。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    chunks: list[str] = []
    current: list[str] | None = None   # None 表示还没遇到第一个 ## 标题
    in_comment = False

    for line in lines:
        # 跳过 <!-- ... --> 注释(可能跨多行)
        if "<!--" in line:
            in_comment = True
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue

        if line.startswith("## "):          # 新 chunk 开始
            if current:
                chunks.append("\n".join(current))
            current = [line[3:].replace("🆕", "").strip()]   # 标题去掉 🆕
        elif current is not None:
            if "TODO" in line:               # 丢掉 TODO 行
                continue
            text = line.replace("🆕", "").strip()
            if text:
                current.append(text)

    if current:
        chunks.append("\n".join(current))
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """一次性批量把所有 chunk 送去 embedding,返回 (N, 维度) 的矩阵。"""
    client = OpenAI()   # 自动读 OPENAI_API_KEY / OPENAI_BASE_URL
    resp = client.embeddings.create(model=MODEL, input=texts)
    vectors = [item.embedding for item in resp.data]
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
