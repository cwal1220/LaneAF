# Target Definition

This document defines the target semantics used by the custom lane-area and lane-line supervision in this repository.

## Terms

### Lane

In this repository, a `lane` means a drivable lane region between two neighboring lane markings.

Examples:

- the region between lane markings `1` and `2`
- the region between lane markings `2` and `3`
- the region between lane markings `3` and `4`

This is an **area target**, not a thin curve.

### Line

In this repository, a `line` means a lane-marking boundary curve.

Examples:

- the left edge of a lane marking
- the right edge of a lane marking

This is a **thin boundary target**, not a filled region.

## Head Semantics

The current custom TuSimple setup uses two segmentation-style heads.

- `lane_hm`: predicts lane areas
- `line_hm`: predicts lane boundary lines

It also uses the original affinity heads.

- `vaf`: vertical affinity field
- `haf`: horizontal affinity field

## TuSimple Mapping

After preprocessing, TuSimple lane markings are ordered from left to right.

- `line 1`: left-most lane marking
- `line 2`: left inner lane marking
- `line 3`: right inner lane marking
- `line 4`: right-most lane marking

These IDs come from the generated `seg_label` instance map.

## TuSimple Lane Targets

The `lane_hm` head uses three channels.

- `ego_lanes_left`: lane area between `line 1` and `line 2`
- `ego_lanes`: lane area between `line 2` and `line 3`
- `ego_lanes_right`: lane area between `line 3` and `line 4`

These are filled lane regions.

## TuSimple Line Targets

The `line_hm` head also uses three channels.

- `ego_lines_left`: `line 1` right edge + `line 2` left edge
- `ego_lines`: `line 2` right edge + `line 3` left edge
- `ego_lines_right`: `line 3` right edge + `line 4` left edge

Each channel is the union of the two listed lane-marking edges.

## Visual Intuition

If visible lane markings are:

`1 | 2 | 3 | 4`

then:

- `ego_lanes_left` is the filled region between `1` and `2`
- `ego_lanes` is the filled region between `2` and `3`
- `ego_lanes_right` is the filled region between `3` and `4`
- `ego_lines_left` is the inner edge pair for lane `1-2`
- `ego_lines` is the inner edge pair for lane `2-3`
- `ego_lines_right` is the inner edge pair for lane `3-4`

## Implementation

- Lane-area targets are built in [datasets/tusimple.py](/home/chan/LaneAF/datasets/tusimple.py:100).
- Lane-line targets are built in [datasets/tusimple.py](/home/chan/LaneAF/datasets/tusimple.py:109).
- Shared TuSimple head metadata lives in [utils/tusimple_targets.py](/home/chan/LaneAF/utils/tusimple_targets.py:1).
