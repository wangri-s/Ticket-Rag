"""
文档加载器 — 多格式文件 → LangChain Document 列表

支持: TXT / PDF / Markdown
每个文档携带 metadata（来源文件路径、文件名、文档类型）。
"""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


class DocumentLoader:
    """统一文档加载器，根据文件后缀自动选择加载策略"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else None

    def load(self, file_path: str) -> list[Document]:
        """加载单个文件"""
        path = Path(file_path)
        if not path.is_absolute() and self.base_dir:
            path = self.base_dir / path
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._load_txt(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in (".md", ".markdown"):
            return self._load_txt(path)  # Markdown 按纯文本处理

        raise ValueError(f"不支持的文件格式: {suffix}")

    def load_directory(
        self, dir_path: str, recursive: bool = True
    ) -> list[Document]:
        """加载目录下所有支持的文档"""
        path = Path(dir_path)
        if not path.is_absolute() and self.base_dir:
            path = self.base_dir / path
        if not path.is_dir():
            raise NotADirectoryError(f"目录不存在: {path}")

        supported = {".txt", ".pdf", ".md", ".markdown"}
        all_docs: list[Document] = []

        # 只遍历文件，跳过 .gitkeep 等非文档文件
        pattern = "**/*" if recursive else "*"
        for f in sorted(path.glob(pattern)):
            if f.is_file() and f.suffix.lower() in supported:
                all_docs.extend(self._load_single(f))

        return all_docs

    # ── 内部加载器 ──────────────────────────────

    def _load_txt(self, path: Path) -> list[Document]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["file_type"] = "txt"
        return docs

    def _load_pdf(self, path: Path) -> list[Document]:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["file_type"] = "pdf"
        return docs

    def _load_single(self, path: Path) -> list[Document]:
        try:
            return self.load(str(path))
        except Exception as e:
            print(f"  [警告] 跳过 {path.name}: {e}")
            return []
