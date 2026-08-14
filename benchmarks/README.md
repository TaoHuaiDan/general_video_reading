# Benchmark dataset contract

Each case should contain:

```text
case-name/
├── video.mp4
├── reference.jsonl
└── case.json
```

`reference.jsonl` uses the same event schema as engine output. At minimum every event must have:

```json
{
  "event_id": 1,
  "start": 1.2,
  "end": 3.8,
  "text_raw": "Example",
  "text_normalized": "Example",
  "polygon_history": [
    [1.2, [[10, 10], [110, 10], [110, 40], [10, 40]]],
    [3.8, [[30, 10], [130, 10], [130, 40], [30, 40]]]
  ]
}
```

The first benchmark suite should deliberately include:

- Fixed layout with long-lived text.
- Camera pan and zoom.
- Moving, rotating and perspective text.
- Typewriter animation.
- Scrolling credits or chat.
- Dense small text.
- Short-lived transient text.
- Repeated UI and numeric changes.

Results must compare quality and speed together. A throughput-only table is not sufficient.

## 已验证的真实输入

`BV1hGGV6REWk-full-final` 是一段 850 秒、2560×1600、约 120 FPS 的视觉小说视频，使用
1 FPS 采样和 1280 最大宽度。最终运行画像：约 300 秒墙钟时间、`2.83× realtime`、845
个采样帧、299 个输出事件、309 个 OCR 任务。该结果是工程基线，不是带标注数据集的
准确率结论；建立 reference.jsonl 后才能计算 Event Recall 和 Text Accuracy。
