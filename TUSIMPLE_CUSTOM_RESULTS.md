# TuSimple Custom Target Results

This file records the internal validation results for the custom TuSimple target setup used in this branch.

These numbers are **not** the official TuSimple benchmark metrics from the original LaneAF README table.
They come from the custom supervision defined in [TARGET_DEFINITION.md](/home/chan/LaneAF/TARGET_DEFINITION.md:1):

- `lane_hm`
  - `ego_lanes_left`
  - `ego_lanes`
  - `ego_lanes_right`
- `line_hm`
  - `ego_lines_left`
  - `ego_lines`
  - `ego_lines_right`
- `vaf`
- `haf`

## ERFNet

- Experiment directory: [experiments/tusimple/ego-triplet-erfnet-lines](/home/chan/LaneAF/experiments/tusimple/ego-triplet-erfnet-lines)
- Log file: [logs.txt](/home/chan/LaneAF/experiments/tusimple/ego-triplet-erfnet-lines/logs.txt)
- Completed epochs: `10`
- Best validation epoch: `10`

Validation metrics at epoch `10`:

| Metric | Value |
|:--|--:|
| total loss | 4.16 |
| area loss | 1.73 |
| line loss | 2.39 |
| VAF loss | 0.02 |
| HAF loss | 0.02 |
| accuracy | 0.9736 |
| lane F1 | 0.8782 |

Notes:

- This run was logged before the per-channel `lane_f1` and `line_f1` TensorBoard scalars were added.
- The logged channel names use the earlier naming:
  - `ego_left`
  - `ego`
  - `ego_right`
  - `ego_left_lines`
  - `ego_lines`
  - `ego_right_lines`

## DLA-34

- Experiment directory: [experiments/tusimple/ego-triplet-dla34-lines](/home/chan/LaneAF/experiments/tusimple/ego-triplet-dla34-lines)
- Log file: [logs.txt](/home/chan/LaneAF/experiments/tusimple/ego-triplet-dla34-lines/logs.txt)
- Training stopped after validation at epoch `22`
- Best checkpoint: [net_0022.pth](/home/chan/LaneAF/experiments/tusimple/ego-triplet-dla34-lines/net_0022.pth)

Validation metrics at epoch `22`:

| Metric | Value |
|:--|--:|
| total loss | 3.83 |
| area loss | 1.59 |
| line loss | 2.21 |
| VAF loss | 0.02 |
| HAF loss | 0.01 |
| lane accuracy | 0.9832 |
| lane F1 | 0.9182 |
| line F1 | 0.5647 |

Per-channel validation F1 at epoch `22`:

| Head | Channel | F1 |
|:--|:--|--:|
| lane_hm | ego_lanes_left | 0.8060 |
| lane_hm | ego_lanes | 0.9362 |
| lane_hm | ego_lanes_right | 0.7769 |
| line_hm | ego_lines_left | 0.5781 |
| line_hm | ego_lines | 0.5521 |
| line_hm | ego_lines_right | 0.5665 |

Training metrics at epoch `22`:

| Metric | Value |
|:--|--:|
| total loss | 2.60 |
| area loss | 0.53 |
| line loss | 2.03 |
| VAF loss | 0.02 |
| HAF loss | 0.02 |
| lane accuracy | 0.9901 |
| lane F1 | 0.9603 |
| line F1 | 0.5942 |

## Summary

- `DLA-34` outperformed `ERFNet` on validation lane-area quality in this custom setup.
- `DLA-34` also reduced validation total loss and line loss relative to `ERFNet`.
- The center `ego_lanes` channel is consistently easier than the left and right lane-area channels.
- Line prediction remains materially harder than lane-area prediction for both backbones.
