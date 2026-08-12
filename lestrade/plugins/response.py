import re
from abc import ABC, abstractmethod

from ..llm.base import ChatMessage

_HTML_FORMAT_ENABLED = True


def set_html_format(enabled: bool):
    global _HTML_FORMAT_ENABLED
    _HTML_FORMAT_ENABLED = enabled


class ResponsePlugin(ABC):
    """响应处理策略的抽象基类。

    垂直领域可继承此类实现自定义逻辑：
    - 法律：回复必须附法条出处，末尾强制免责声明
    - 医疗：症状描述→诊断建议不构成医疗意见的声明
    - 电商：回复插入商品链接、优惠券
    """

    @abstractmethod
    def build_messages(
        self,
        contexts: list,
        user_message: str,
        lang: str,
    ) -> list[ChatMessage]:
        """根据检索到的上下文和用户消息构建发送给 LLM 的消息列表。"""
        ...

    @abstractmethod
    def format_response(self, text: str) -> str:
        """对 LLM 返回的原始文本进行后处理（格式化、添加免责声明等）。"""
        ...


class DefaultResponse(ResponsePlugin):
    """默认响应策略：中英双语检测、上下文 Prompt 注入、HTML 格式化。"""

    def build_messages(
        self,
        contexts: list,
        user_message: str,
        lang: str,
    ) -> list[ChatMessage]:
        context_text, _ = self._prepare_contexts(contexts, user_message)
        if context_text:
            if lang == 'en':
                augmented = (
                    f"Answer the question using only the provided information. "
                    f"Be concise and factual. Do not mention sources.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {user_message}\n"
                )
            else:
                augmented = (
                    f"仅根据提供的资料回答问题，简洁、实事求是。"
                    f"不要提及来源，不要使用 markdown 符号。\n\n"
                    f"资料：\n{context_text}\n\n"
                    f"问题：{user_message}\n"
                )
        else:
            from .. import config
            fallback = config.FALLBACK_MESSAGE_ZH if lang == 'zh' else config.FALLBACK_MESSAGE_EN
            augmented = (
                f"{fallback}\n\nUser: {user_message}" if lang == 'en'
                else f"{fallback}\n\n用户：{user_message}"
            )
        return [ChatMessage(role="user", content=augmented)]

    def format_response(self, text: str) -> str:
        if not _HTML_FORMAT_ENABLED:
            return text
        return self._format_html(text)

    def _prepare_contexts(self, contexts, user_message) -> tuple[str, list]:
        if not contexts:
            return '', []

        merged = {}
        for c in contexts:
            merged.setdefault(c[1], []).append(c)
        src_limit = 2
        filtered = []
        for src, chunks in list(merged.items())[:src_limit]:
            filtered.extend(chunks)

        items = []
        for c in filtered:
            item = re.sub(r'^\[\S+?\]\s*', '', c[0])
            item = re.sub(r'^#{2,4}\s*.*\n?', '', item, count=1)
            items.append(item.strip())
        return "\n\n".join(items), filtered

    def _format_html(self, text: str) -> str:
        _re_bullet = re.compile(r'^[-*]\s')
        _re_ordered = re.compile(r'^\d+[.)]\s')

        if re.search(r'<(li|p|ul|ol)\b', text):
            return text
        if not re.search(r'^[-*\d]', text, re.MULTILINE):
            return text
        lines = text.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if _re_bullet.match(stripped):
                out.append("<ul>")
                while i < len(lines) and _re_bullet.match(lines[i].strip()):
                    c = _re_bullet.sub('', lines[i].strip())
                    out.append(f"<li>{c}</li>")
                    i += 1
                out.append("</ul>")
            elif _re_ordered.match(stripped):
                out.append("<ol>")
                while i < len(lines) and _re_ordered.match(lines[i].strip()):
                    c = _re_ordered.sub('', lines[i].strip())
                    out.append(f"<li>{c}</li>")
                    i += 1
                out.append("</ol>")
            elif stripped == "":
                out.append("")
                i += 1
            else:
                out.append(f"<p>{line}</p>")
                i += 1
        return "\n".join(out)
