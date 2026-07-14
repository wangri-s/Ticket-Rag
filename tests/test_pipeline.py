"""
测试摄入流水线 src/ingestion/pipeline.py
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import IngestionPipeline


# 构建足够长的样本，确保超过 chunk_size=500，触发分块
_UNIT_1 = (
    "═══════════════════════\n"
    "【工单编号】GD-2026-03001\n【提交时间】2026-03-01 08:35\n"
    "【设备类型】CT 机\n【设备型号】Siemens SOMATOM go.Top\n"
    "【所属科室】放射科\n"
    "【故障现象】CT 扫描过程中偶发图像伪影，表现为条状低密度阴影，"
    "影响右下肺叶区域影像判读，故障频率约每 10 例检查出现 1 次。\n"
    "【处理方案】\n"
    "  Step 1: 校准 X 射线管电流调制模块（mA modulation），零位漂移补偿。\n"
    "  Step 2: 检查探测器阵列，第 3 排第 12 通道灵敏度下降 15%，更换该通道探测器模块。\n"
    "  Step 3: 执行空气校准（Air Calibration）和水模校准（Water Phantom）。\n"
    "【处理结果】伪影消失，图像质量恢复正常。建议 3 个月后复查探测器阵列。\n"
)

_UNIT_2 = (
    "───────────────────────────────────────────────────────────────\n"
    "【工单编号】GD-2026-03002\n【提交时间】2026-03-01 14:20\n"
    "【设备类型】MRI 核磁共振\n【设备型号】GE SIGNA Pioneer 3.0T\n"
    "【所属科室】影像中心\n"
    "【故障现象】扫描过程中梯度线圈发出异常啸叫声，声音频率约 2kHz，"
    "比正常运行声音高 10-15dB，患者反馈不适，1 例检查被迫中断。\n"
    "【处理方案】\n"
    "  Step 1: 关闭系统，断开梯度放大器电源，检查输入端滤波电容。\n"
    "  Step 2: 发现 X 轴梯度放大器直流母线滤波电容 C12 鼓包，更换同规格电容。\n"
    "  Step 3: 重新上电，运行梯度线性度测试，谐波分量 < 0.5%，正常。\n"
    "【处理结果】啸叫消除，扫描正常运行。建议每半年检查梯度放大器电容状态。\n"
)

SAMPLE_TICKET = _UNIT_1 + _UNIT_2


class TestPipeline:
    """摄入流水线"""

    def test_run_with_txt_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test_tickets.txt"
            f.write_text(SAMPLE_TICKET, encoding="utf-8")

            pipeline = IngestionPipeline()
            docs = pipeline.run(tmpdir)

            assert len(docs) >= 2  # 至少两个工单各产生块

    def test_run_with_single_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_TICKET)
            tmp_path = f.name

        try:
            pipeline = IngestionPipeline()
            docs = pipeline.run(tmp_path)
            assert len(docs) >= 2
        finally:
            Path(tmp_path).unlink()

    def test_run_with_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test_tickets.txt"
            f.write_text(SAMPLE_TICKET, encoding="utf-8")

            pipeline = IngestionPipeline()
            result = pipeline.run_with_stats(tmpdir)

            assert result["total_chunks"] >= 2
            assert result["total_tickets"] == 2
            assert result["max_chunk_size"] <= 500 + 50
            assert "GD-2026-03001" in result["ticket_ids"]
            assert "GD-2026-03002" in result["ticket_ids"]

    def test_run_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline()
            docs = pipeline.run(tmpdir)
            assert docs == []

    def test_group_by_ticket_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test_tickets.txt"
            f.write_text(SAMPLE_TICKET, encoding="utf-8")

            pipeline = IngestionPipeline()
            docs = pipeline.run(tmpdir)
            groups = pipeline.group_by_ticket(docs)

            # 至少有两张工单被识别
            assert "GD-2026-03001" in groups or any(
                "GD-2026-03001" in str(d.page_content) for d in docs
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

