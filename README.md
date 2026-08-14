# Temporal OCR Engine

独立、可 benchmark 的通用视频 OCR 引擎。项目目标是在尽量保证文字事件完整性与
识别准确率的前提下，利用视频的时序冗余最大化处理吞吐。

它不依赖 Bilibili、MCP、浏览器或网络下载逻辑。未来这些系统只能通过适配器调用本引擎。

## 当前里程碑

首版先建立可以独立验证的架构内核：

- 几何轨迹与文字内容轨迹分离。
- 全局运动补偿和运动补偿后的分块变化检测。
- 四边形透视矫正与规范化内容指纹。
- 质量与信息互补性兼顾的候选帧选择。
- 快速、局部高分辨率、完整审计三级检测请求。
- FAST 已覆盖区域会跳过 LOCAL；相邻变化瓦片先合并，避免逐瓦片调用模型。
- 基于运行时统计的规则式策略调度器。
- Event Recall、Text Accuracy、Duplicate Rate、时空 IoU、吞吐和延迟指标。
- 可替换的视频源、检测器和 OCR 后端接口。

架构细节见 [docs/architecture.md](docs/architecture.md)。

## 开发安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[video,ocr,dev]"
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

查看 CLI：

```powershell
.\.venv\Scripts\python.exe -m temporal_ocr --help
```

使用 RapidOCR 基线处理本地视频：

```powershell
.\.venv\Scripts\python.exe -m temporal_ocr run video.mp4 --output artifacts\run-001
```

高帧率或超高清录屏可以显式限制输入成本；例如视觉小说通常可先用 5 FPS、1280 宽做
基准，再根据漏检情况提高采样率或尺寸：

```powershell
.\.venv\Scripts\python.exe -m temporal_ocr run video.mp4 `
  --output artifacts\run-001 --sample-fps 5 --max-width 1280
```

这些输入参数会写入 `run.json`，保证 benchmark 可复现。对远高于采样率的 H.264 视频，
输入源会自动跳过双向预测帧，减少无效解码；如果需要逐帧保真分析，可以不设置
`--sample-fps`。

使用 `--end-sec` 做短段 benchmark 时，读取器会在达到终点后立即停止，不会继续扫描整个源视频。

该基线已经将检测与识别分开：检测器输出四边形，文字框经过透视规范化后，
recognizer 使用 RapidOCR 内部的批量文字识别接口。低置信度任务才会使用互补候选帧重试。

当前版本已经接入真实 RapidOCR CPU 检测与批量识别，并通过本地视频端到端运行。
下一阶段重点是建立带标注的多场景 benchmark、自动参数搜索和 GPU 后端对照；这些扩展
都通过现有接口接入，不会让特定视频类型侵入核心逻辑。

## 真实视频速度基准

对一段 850 秒、2560×1600、约 120 FPS 的视觉小说视频，使用 `--sample-fps 1 --max-width 1280`
完整处理耗时约 300 秒，即 `2.83× realtime`。当前画像中检测耗时约 120 秒，OCR 推理约
17 秒；其余时间主要是视频解码、候选管理和模型运行时开销。结果保存在
`benchmarks/output/BV1hGGV6REWk-full-final/`。
