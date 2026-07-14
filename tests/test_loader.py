"""
测试文档加载器 src/ingestion/loader.py
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.ingestion.loader import DocumentLoader


class TestLoader:
    """加载器基础功能"""

    @property
    def loader(self):
        return DocumentLoader()

    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("CT 扫描偶发图像伪影\n排查步骤：检查探测器阵列")
            tmp_path = f.name

        try:
            docs = self.loader.load(tmp_path)
            assert len(docs) == 1
            assert "CT 扫描" in docs[0].page_content
            assert docs[0].metadata["file_type"] == "txt"
        finally:
            Path(tmp_path).unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load("/nonexistent/file.txt")

    def test_load_unsupported_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("test")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持"):
                self.loader.load(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建两个 txt 文件
            (Path(tmpdir) / "a.txt").write_text("文档A内容", encoding="utf-8")
            (Path(tmpdir) / "b.txt").write_text("文档B内容", encoding="utf-8")
            # 创建一个非文档文件
            (Path(tmpdir) / "readme.gitkeep").write_text("")

            docs = self.loader.load_directory(tmpdir)
            assert len(docs) == 2

    def test_load_directory_skips_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.txt").write_text("有效文件", encoding="utf-8")
            (Path(tmpdir) / "image.png").write_text("假装是图片")

            docs = self.loader.load_directory(tmpdir)
            assert len(docs) == 1

    def test_metadata_present(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("测试内容")
            tmp_path = f.name

        try:
            docs = self.loader.load(tmp_path)
            assert "source" in docs[0].metadata
            assert "file_type" in docs[0].metadata
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

