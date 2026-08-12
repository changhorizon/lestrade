import re
from abc import ABC, abstractmethod


class ChunkingPlugin(ABC):
    """文本分块策略的抽象基类。

    垂直领域可继承此类实现自定义分块逻辑：
    - 法律文档按条款分块
    - 产品手册按功能/章节分块
    - 病历按诊断流程分块
    """

    @abstractmethod
    def chunk(self, text: str, max_chars: int = 5000) -> list[str]:
        ...


class MarkdownChunking(ChunkingPlugin):
    """默认 Markdown 感知分块策略。

    按标题分节，尊重代码块边界，合并过小的碎片。
    """

    def chunk(self, text: str, max_chars: int = 5000) -> list[str]:
        text = self._strip_frontmatter(text)
        lines = text.split("\n")
        sections = []
        current_lines = []

        def flush():
            if not current_lines:
                return
            block = "\n".join(current_lines).strip()
            current_lines.clear()
            if not block or len(block) < 20:
                return
            sections.append(block)

        for line in lines:
            stripped = line.strip()
            if stripped in ("---", "___", "***") or stripped.startswith("<!--"):
                continue
            if re.match(r"^#{1,5} ", line):
                flush()
                current_lines.append(line)
            else:
                current_lines.append(line)

        flush()

        chunks = []
        for sec in sections:
            if len(sec) <= max_chars:
                chunks.append(sec)
            else:
                parts = re.split(r"(```[\s\S]*?```)", sec)
                for part in parts:
                    if not part.strip():
                        continue
                    if part.startswith("```") and part.endswith("```"):
                        chunks.append(part[:max_chars])
                    elif len(part) > max_chars:
                        for para in part.split("\n\n"):
                            para = para.strip()
                            if para and len(para) >= 20:
                                chunks.append(para[:max_chars])
                    else:
                        chunks.append(part)

        merged = []
        for c in chunks:
            if merged and len(c) < 30:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)

        if not merged:
            text_clean = text[:max_chars]
            if text_clean.strip():
                merged = [text_clean]

        return merged

    def _strip_frontmatter(self, text: str) -> str:
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                return text[end + 3:].strip()
        return text
