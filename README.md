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
- 已有高置信度文字轨迹覆盖的 LOCAL 区域直接复用投影四边形，独立运动超过阈值时
  自动退回检测；周期性 AUDIT/FAST 仍负责发现新文字。
- 基于运行时统计的规则式策略调度器。
- Event Recall、Text Accuracy、Duplicate Rate、时空 IoU、吞吐和延迟指标。匹配只依赖
  时空对应关系：文本识别错误计入 Text Accuracy（未匹配 reference 计为删除错误），
  另有 matched_text_accuracy 与 event_precision 两个补充指标。
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
输入源会自动跳过非参考帧，减少无效解码；如果需要逐帧保真分析，可以不设置
`--sample-fps`。

使用 `--end-sec` 做短段 benchmark 时，读取器会在达到终点后立即停止，不会继续扫描整个源视频。

如果目标是视觉小说这类文字尺寸较大、画面布局相对稳定的视频，可以使用仓库附带的高速
配置。它把 LOCAL/AUDIT 的输入上限分别设为 960/1600，并把运动不可靠时的审计限制在较低
频率；同时默认给 ONNX Runtime 使用 12 个 intra-op 线程、1 个 inter-op 线程。示例：

```powershell
.venv\Scripts\python.exe -m temporal_ocr run `
  benchmarks\input\BV1hGGV6REWk\BV1hGGV6REWk.mp4 `
  --output benchmarks\output\BV1hGGV6REWk-60s-fast `
  --config benchmarks\config-fast.json --end-sec 60 `
  --sample-fps 1 --max-width 1280
```

在当前 i9-13980HX 上，这条命令对同一视频前 60 秒实测约 10.3 秒，即 `5.74× realtime`；
14 秒对白的三行（“我慢慢走过去，”“在床边坐下。”“希罗也跟着”）仍全部识别到。该结果是
工程回归，不等同于带人工标注集的 Event Recall/Text Accuracy；密集小字、滚动字幕或短暂
弹窗应优先使用默认高分辨率配置，再根据标注结果逐步放宽高速参数。

该基线已经将检测与识别分开：检测器输出四边形，文字框经过透视规范化后，
recognizer 使用 RapidOCR 内部的批量文字识别接口。低置信度任务才会使用互补候选帧重试。

当前版本已经接入真实 RapidOCR CPU 检测与批量识别，并通过本地视频端到端运行。
下一阶段重点是建立带标注的多场景 benchmark、自动参数搜索和 GPU 后端对照；这些扩展
都通过现有接口接入，不会让特定视频类型侵入核心逻辑。

## 真实视频速度基准

对一段 850 秒、2560×1600、约 120 FPS 的视觉小说视频，使用 `--sample-fps 1 --max-width 1280`
完整处理的旧基线约 300 秒（`2.83× realtime`），检测约 120 秒。启用轨迹引导 LOCAL 后，
完整运行约 185 秒（`4.59× realtime`），检测约 41 秒；模型 LOCAL 调用从 334 次降到 23 次，
输出事件从 299 增至 367（仍需带标注集确认 Event Recall/Text Accuracy）。结果分别保存在
`benchmarks/output/BV1hGGV6REWk-full-final/` 和 `benchmarks/output/BV1hGGV6REWk-full-guided/`。
如需做消融对照，可在配置文件中设置 `detection.track_guided_local=false`。

轨迹复用不会把“上一帧的一行框”永久当成完整对话框：当补偿后的变化像素溢出高置信度
轨迹，LOCAL 会刷新其几何范围；刷新后新建或同范围内的轨迹进入短暂冷却，先完成下一次
稳定观察再允许再次刷新，避免多行对白刚检测出来就被下一帧提前结束。探测冷却期间若已有
文字轨迹仍发生小范围变化，调度器会发出 `urgent_local_change`，但仍限制在 LOCAL scope，
不会退回全画面 FAST。

以同一视频前 60 秒、`--sample-fps 1 --max-width 1280` 的保护版历史回归约 28.0 秒，
即 `2.12× realtime`；它主要用于检测刷新完整性对照。当前推荐的高速配置见上文，正式结果
保存在 `benchmarks/output/BV1hGGV6REWk-60s-fast40-w1280-nonref/`。

## 本地 OCR MCP

项目还提供一个独立的 stdio MCP 适配层。它只处理已经存在于本地的视频文件；Bilibili
视频获取、AI 字幕、评论和弹幕仍由独立的 Bilibili MCP 负责，两个服务不互相耦合。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[video,ocr,mcp]"
.\.venv\Scripts\python.exe -m temporal_ocr.mcp_server
```

MCP 的详细工具、异步任务和结果文件说明见 `docs/mcp.md`。默认 OCR 任务以 1 FPS、1280 px
上限运行，并把完整事件写入 `events.jsonl`，对话中只返回摘要和文件路径。
长视频可使用 `ocr_video_chunked`：引擎会自动选择约 180 秒的分段间距、保留重叠区并行识别，
再按时间、文本相似度和空间位置合并边界重复事件；300 秒以内的视频自动走连续引擎。
默认的 `ocr_video` 也使用这套自动策略；分段算法位于 OCR 核心，MCP 只负责任务适配。

所有视频 OCR 工具还支持可选的 `exclude_regions`。调用方可以先根据
视频画面判断水印位置，再传入归一化矩形 `[left, top, right, bottom]`，例如
`[[0.78, 0.88, 1.0, 1.0]]`。这些区域会被排除在运动/变化分析和检测结果之外，
并自动沿用到长视频的每个分段。
