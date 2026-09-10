# YOLO26 vs YOLO11 on VisDrone: how to run the notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommergo/yolo26-vs-yolo11-visdrone/blob/main/yolo26_vs_yolo11_visdrone.ipynb)

**Deep Learning - Final Project**

**Paper:** *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, Ultralytics, [arXiv:2606.03748](https://arxiv.org/abs/2606.03748), June 2026

**Group members**

| Name | ID |
|---|---|
| David Goldzweig | 342745767 |
| Tomer Goldstein | 207797234 |
| Neta Barnoor | 322852146 |

**Links**

- Repository: https://github.com/Tommergo/yolo26-vs-yolo11-visdrone
- Video walkthrough (YouTube): https://www.youtube.com/watch?v=ydTMnXCKuyo

**At a glance**

| | |
|---|---|
| Notebook | [`yolo26_vs_yolo11_visdrone.ipynb`](https://colab.research.google.com/github/Tommergo/yolo26-vs-yolo11-visdrone/blob/main/yolo26_vs_yolo11_visdrone.ipynb) |
| Weights | `weights/` in this repo, no training needed |
| Dataset | VisDrone, downloaded automatically (~2 GB) on first run |
| Total runtime | ~30 minutes end to end |

---

## What is being compared

Two models were trained, YOLO26n and YOLO11n, and the comparison is drawn between **three
configurations** of them:

| # | Configuration | Head | Post-processing |
|---|---|---|---|
| 1 | YOLO26n + NMS | one-to-many | NMS |
| 2 | YOLO26n end-to-end | one-to-one | none |
| 3 | YOLO11n + NMS | (YOLO11's only head) | NMS |

Not every experiment uses all three.

**Why the two YOLO26n rows are one model.** Training produced one set of weights per model.
YOLO26n's weights carry **two detection heads**, and which one runs is chosen at inference
time rather than at training time, so configurations 1 and 2 have the same trained weights
but are different: identical parameters, different behaviour. In the notebook that choice is
a single argument: `nms=None` selects the one-to-many head, `nms=False` selects the
one-to-one head. YOLO11n has only one head, so it has only one configuration.

### The terms used throughout

| Term | What it means |
|---|---|
| **NMS** (non-maximum suppression) | The clean-up step after the network: a detector proposes several overlapping boxes per object, NMS keeps the best one and deletes the rest. Costs time that grows with how crowded the image is, and has a threshold to tune. |
| **One-to-many head** | The classic design, emits duplicates by construction, so it needs NMS. |
| **One-to-one head** (end-to-end, NMS-free) | One prediction per object, so NMS can be skipped entirely: a single graph, nothing to tune, latency that no longer depends on how crowded the scene is. What it costs in accuracy is the question this project asks. |
| **mAP50-95** | The standard detection score, averaged over ten overlap thresholds from 0.50 to 0.95 (`mAP50` is the loose single-threshold version). |
| **Latency** | Reported as `preprocess` / `inference` / `postprocess`. Claim D is about the network alone, so it is measured on `inference`, with NMS excluded from both sides. |
| **Paired bootstrap** | The 548 validation images resampled 1000 times, all configurations scored on the *same* resample each round, so variation between images cancels. |

---

## 1. Before you press anything

### Runtime: L4, not T4

In Colab: **Runtime > Change runtime type > L4 GPU**.

This is not a preference. The L4 runtime comes with **12 vCPUs and ~57 GB RAM**; the T4
runtime comes with **2 vCPUs and ~14 GB**. Two consequences:

- The dataloader starves on 2 vCPUs, and GPU latency reads 40-55 % higher than it should.
- Part 7 times the models **on the CPU**. On 2 vCPUs the model becomes compute-bound, the
  memory-bandwidth saving that Claim D rests on disappears, and the measured figure
  collapses from ~33 % to ~19 %. The claim looks refuted when in fact the machine is wrong.

Cell **0** prints the GPU, vCPU count and RAM. If it does not say `NVIDIA L4` and
`vCPUs: 12`, change the runtime before going further.

### Ultralytics version: pinned to 8.4.145

Cell **0.1** installs `ultralytics==8.4.145`, the version the original run used. Two things
moved between releases and both matter here:

| version | what it reports |
|---|---|
| 8.4.138 | one-to-many head: **146 layers, 2,498,204 params** |
| 8.4.145 | one-to-many head: **120 layers, 2,376,786 params** |
| 8.4.146 | CPU postprocess ~14 ms instead of ~41 ms (different NMS implementation) |

If the cell stops with an error saying the version does not match, Ultralytics was already
loaded in this session and the old version is still in memory: **Runtime > Restart session**,
then run from the top.

To run on whatever pip resolves today, set `ULTRA_PIN = None` in that cell, but expect
Part 8 to report differences.

---

## 2. Quick start

1. Open the notebook in Colab.
2. **Runtime > Change runtime type > L4 GPU.**
3. Run cell **0** and confirm `NVIDIA L4`, `vCPUs: 12`.
4. Run **0.1** (installs the pinned Ultralytics). Restart the session if it tells you to.
5. Run **0.2** and approve the Google Drive mount. Everything is written under
   `MyDrive/dl_final_project_yolo26/`.
6. Run the rest of Part 0 in order: **0.3** (constants), **0.4** (rounding and display
   helpers), **0.5** (downloads the weights). Each prints what it set up, so you can see
   it worked before moving on.
7. In cell **0.6**, choose `DATA_SOURCE` (see section 3) and run it, then run **0.7**
   (the frozen original run, which Part 8 compares against).
8. **Runtime > Run all.** Skip Part 1 (training); the weights came from 0.5.

If you would rather not mount Drive, set `SAVE_TO_DRIVE = False` in 0.2. Outputs then go to
`/content/runs` and are lost when the session ends.

---

## 3. `DATA_SOURCE`, the one switch

**Where:** cell **0.6**, first line.

```python
DATA_SOURCE = 'original'     # or 'new'
```

| value | what the tables show |
|---|---|
| `'original'` | the latencies the slideshow was built from (2026-09-02, L4, 12 vCPU, ultralytics 8.4.145) |
| `'new'` | the latencies measured in this session |

**Either way every cell still measures.** The switch does not turn measurement off, it
decides which numbers get *displayed*. Only latency is affected: mAP, mAP50, precision,
recall, per-class AP, box counts and all 13 bootstrap intervals are deterministic and come
out the same under both settings.

Why the switch exists: latency is the one quantity here that does not reproduce. It moves up
to 10 % between sessions on shared cloud hardware, and the *ratio* between two models moves
more than either measurement, because they drift independently. Freezing the original
latencies keeps the notebook's tables matching the slideshow while every figure is still
recomputed underneath.

**Which to use**

- `'original'` to reproduce the slideshow exactly. Part 7a can be skipped (it is an
  11-minute CPU timing run that 7b will not read).
- `'new'` to see what this machine produces. 7a is then **required**; 7b raises
  `FileNotFoundError` without it.

Either way, **Part 8 compares the two**, value by value, with the difference in milliseconds
and in percent. That comparison is the point of the switch: a reader can see how far today's
hardware sits from the run behind the slides.

### Is my run correct?

The expected result is the original run, the numbers the slideshow reports. You do not have
to check them by hand: **Part 8 does it**, putting every figure of the original run beside
today's with the difference between them, including all four claims.

Read it this way: latency is allowed to move (hardware and session load), while mAP,
per-class AP, box counts, the intervals and the model size should agree to the last digit or
close to it. `MODEL SIZE 9/9 values match the original run` means the checkpoint, the head
selection and the Ultralytics version are all what they were; a `DIFFERS` there is the first
thing to look at.

The frozen numbers themselves live in cell **0.7** (`SNAP_GPU`, `SNAP_CPU`, `SNAP_SIZE`,
`SNAP_CLAIMS`, `SNAP_ENV`). Nothing measures against them; only Part 8 reads them.

There is a second switch beside it:

```python
VERBOSITY = 'compact'        # or 'full'
DRIFT_TOL = 5.0              # percent
```

`'compact'` is the submission view, where Part 8 lists only what moved. `'full'` prints every
row and both runs' tables in full. `DRIFT_TOL` is the threshold below which a latency row is
summarised instead of listed.

---

## 4. No training needed

Cell **0.5** downloads both trained checkpoints from this repo's `weights/` folder
(~5.4 MB each) and verifies their byte size and parameter counts:

| file | fused params |
|---|---|
| `yolo26n_visdrone_musgd.pt` | 2,376,786 |
| `yolo11n_visdrone_sgd.pt` | 2,584,102 |

They are downloaded fresh every run, so the notebook never depends on what happens to be in
your Drive.

**Part 1 is the training code, kept for reference. Do not run it.** It is ~4 GPU-hours: 60
epochs per model, YOLO26n under MuSGD and YOLO11n under SGD with momentum, each under the
optimizer its own authors specify. The optimizer is the only intentional difference between
the two runs; dataset, image size (640), batch (16), seed (0) and patience are identical.

---

## 5. What runs, and for how long

| Part | What it does | Time | Needed for |
|---|---|---|---|
| 0 | Setup, weights, the switch | ~2 min | everything |
| 1 | Training | ~4 h | **skip** |
| 2a | Three `val()` runs, the only cell that runs the models | ~2 min | all slides |
| 2b | VisDrone labels rebuilt in COCO format | ~1 min | 2c |
| 2c | Paired bootstrap, 1000 resamples | ~15 min | every interval and p-value |
| 3 | Slide 9, Claim A | seconds | |
| 4 | Slide 10, Claim B, per class by object size | seconds | |
| 5 | Slide 11, Claim C test 1: the pictures and the box counts | ~1 min | |
| 6 | Slide 12, Claim C test 2: NMS-free vs NMS | seconds | |
| 7a | CPU timing, ONNX, 5 interleaved repeats | ~11 min | only `DATA_SOURCE='new'` |
| 7b | Slide 13, Claim D | seconds | |
| 8 | Today vs the original run, claim by claim | seconds | |

Run them in order. Parts 3-8 all read what Part 2 produced; 2c is the long one and only
needs to run once per session.

---

## 6. What lands in Drive

Under `MyDrive/dl_final_project_yolo26/`:

```
val/<config>/predictions.json     one per configuration, read by 2b
visdrone_val_gt.json              VisDrone labels in COCO format
bootstrap_all.csv                 all 13 intervals
slide09_result1.csv               Result 1
slide10_per_class.csv             per-class breakdown
slide11_box_counts.csv            box counts
slide13_cpu_raw.csv               per-repeat CPU timings (7a writes, 7b reads)
slide13_cpu_onnx.csv              Result 3
qualitative/                      the eight annotated images + the 2x4 grid
```

`slide13_cpu_raw.csv` is the one worth knowing about: 7b reads it instead of re-timing, so
7b can be re-run freely at no cost. If it is stale from an earlier session, delete it before
running 7a.

---

## 7. The four claims, and where each is tested

| Claim | Paper says | Tested in |
|---|---|---|
| **A** mAP gain over YOLO11 | +1.6 to +2.8 at matched scales (nano is not tabulated; +1.4 derived from the published COCO figures) | Part 3 |
| **B** gains concentrate on small, dense objects (STAL) | | Part 4 |
| **C** the NMS-free head costs only 0.6-0.8 mAP | | Parts 5 and 6 |
| **D** up to 43 % faster CPU inference | ceiling, NMS excluded from the measurement | Part 7 |

---

## 8. If something goes wrong

**`pin failed: got 8.4.x, wanted 8.4.145`** Ultralytics was already loaded. Runtime >
Restart session, then run from the top.

**`DATA_SOURCE = 'new' but slide13_cpu_raw.csv does not exist`** run 7a, or switch to
`'original'`.

**`both rows report N layers -- the nms flag was ignored`** (2a) the Ultralytics version in
the session does not honour `nms=`, so the same head ran twice. Check that 0.1 installed
8.4.145 and that the session was restarted.

**`Results do not correspond to current coco set`** (2b/2c) Ultralytics writes `image_id` as
the filename stem when the stem is not numeric, and VisDrone's stems are not. 2b remaps them;
if this appears, the remap did not run, so re-run 2b before 2c.

**`No "summary (fused)" line was captured`** (2a) Ultralytics changed its logging, so the
layer/parameter/GFLOPs columns cannot be filled. Check the pin.

**Part 8 says `DIFFERS` on model size** the checkpoint, the head selection or the Ultralytics
version is not what the original run used. This is the check working, not a bug; start with
the version.

**Claim D comes out near 19 % instead of ~33 %** you are on a 2-vCPU runtime. See section 1.
