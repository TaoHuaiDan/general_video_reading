# Temporal OCR Engine architecture

## Scope

Temporal OCR Engine is a local, platform-neutral library. Its input is a decoded frame stream;
its output is a sequence of spatially and temporally bounded text events. Bilibili download,
MCP transport, browser automation and LLM-based correction are outside the core package.

The engine optimizes for, in order:

1. Event completeness.
2. Text accuracy.
3. Duplicate suppression.
4. Throughput.
5. Streaming latency where applicable.

## Core boundary

The engine deliberately separates three identities:

```text
GeometryTrack: where the text is and how it moves
ContentTrack: whether the normalized text appearance changed
TextEvent: the final recognized content plus its time and geometry history
```

A fixed subtitle box can produce many content tracks. A moving or rotating title can keep one
content track while its geometry changes continuously.

## Processing graph

```text
FrameSource
  -> luma pyramid
  -> global motion estimation
  -> compensated tile change map
  -> policy scheduler
  -> hierarchical detection requests
  -> geometry association
  -> perspective normalization
  -> content tracking
  -> complementary candidate selection
  -> exact cache
  -> OCR micro-batches
  -> confidence fallback
  -> TextEvent output
```

## Global motion

The initial implementation estimates an affine transform from sparse optical flow and RANSAC.
Known text polygons are masked so that moving captions do not become camera-motion evidence.
The transform is enabled only when its inlier ratio and residual pass configured gates. An invalid
estimate only becomes a risk signal when widespread residual change is also present, and
motion-triggered audits are rate-limited. This avoids audit storms on flat or synthetic scenes
where sparse feature matching is impossible but the image is otherwise stable.

Dense flow, SLAM and mesh warping remain deferred until a benchmark set demonstrates that affine
motion is insufficient often enough to justify their cost.

## Detection tiers

- `FAST`: low-resolution full-frame recall-oriented proposal pass.
- `LOCAL`: high-resolution passes over motion-compensated changed tiles.
- `AUDIT`: high-accuracy full-frame pass after scene cuts, at a bounded interval, or after motion
  compensation becomes unreliable.

All tiers emit the same `DetectionObservation`. Tracking and OCR do not depend on detector type.
FAST runs before LOCAL. Changed tiles are merged into connected scopes, and scopes already covered
by FAST observations are not reprocessed locally. Cross-tier boxes are fused by IoU and containment
before they can create geometry or content tracks.

For offline high-FPS sources, the frame source can sample timestamps, skip H.264 B-frames when
safe, and stop decoding immediately at `end_sec`. The scheduler then becomes event-driven: idle
frames update active event extent without invoking detection, small changes use LOCAL, and broad
changes or audits use FAST/AUDIT. Typewriter intermediate states can be deferred from OCR until a
stable or final state is available.

When a LOCAL scope is already covered by a high-confidence geometry track, the engine projects the
track polygon with the current global motion estimate and feeds the current crop directly to
content tracking. This removes a redundant detector invocation while preserving periodic
full-frame discovery. Tracks with excessive independent velocity are excluded from this shortcut
and fall back to LOCAL detection, so scrolling or independently moving text does not silently
inherit a stale box. `detection.track_guided_local` is an explicit ablation switch.

## Policy scheduler

The scheduler consumes continuous runtime signals rather than a video-type label. Initial rules
adjust probe interval, audit interval, detector width, stability wait, batch size and batch delay.
Every decision is logged, bounded and deterministic so it can be evaluated with ablation tests.

No reinforcement learning is planned before the deterministic controller has a strong benchmark
baseline.

## Accuracy safeguards

- Geometric normalization happens before content hashing and OCR.
- New text events cannot be discarded because of queue pressure; only redundant probes may be
  delayed.
- Geometry disappearance forces the best content candidate to be submitted.
- Exact cache reuse is the default. Perceptual cache reuse will require a separate accuracy study,
  especially for numbers and short strings.
- Low-confidence OCR candidates remain observable; language correction must never overwrite raw
  OCR evidence.
- Full-frame audits have a maximum interval even when the layout appears static.

## Benchmark contract

Every benchmark run records:

- Event Recall.
- Text Accuracy (`1 - CER`).
- Duplicate Rate.
- Mean Temporal IoU.
- Mean Spatial IoU.
- Processed video seconds per wall second.
- P50/P95/P99 latency when streaming timestamps are available.
- Per-stage time and detector/OCR request counts.
- Policy changes and their reasons.

Configuration ranking is completeness-first. A faster configuration is rejected when it reduces
event recall or text accuracy beyond the benchmark acceptance gate.

## Iteration plan

1. Core abstractions, synthetic tests and benchmark schema. (implemented)
2. Real detector and batched CPU recognizer adapters. (implemented)
3. Small annotated video suite covering static, moving, rotating, perspective, scrolling,
   typewriter and transient text.
4. GPU batch recognition and CPU/GPU comparison.
5. Rule-controller ablation tests.
6. Only then consider dense motion, multi-frame image fusion or learned scheduling.
