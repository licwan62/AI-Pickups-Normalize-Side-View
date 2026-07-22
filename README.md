# Pickup Side Profile Measurement Tool

Phase 1 implements the path from TSV/CSV and automatically detected vehicle bounds to a physically sized SVG.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item input\vehicles.tsv.example input\vehicles.tsv
```

Put images in `input/images/`. Each filename must match the table `id`, for example
`input/images/F150_01.jpg`. JPG, JPEG, PNG, and WebP are supported. An optional
`image_path` column can override the default lookup for individual rows.

## Run

```powershell
python main.py
```

The default run automatically segments the foreground, crops the likely vehicle bounds,
saves them to `output/<id>/points.json`, and independently stretches the crop to the
table's real length and height. Automatic detection is designed first for clean side-profile
images with a plain or studio-style background.

`crop_source.png` preserves only the tightly detected source pixels. It is embedded directly
into `vehicle.svg` and enlarged through the SVG millimetre coordinate system. No separate
high-pixel-count `crop.png` is generated. PPI therefore does not control Illustrator placement;
the SVG `width`, `height`, and `viewBox` remain authoritative.

To reuse previously reviewed bounds instead of detecting again:

```powershell
python main.py --reuse-points
```

For a difficult image that automatic segmentation cannot isolate, explicitly request the
manual fallback:

```powershell
python main.py --manual
```

Ratio distortion of at most 3% passes. From 3% through 6%, export pauses and records `WARNING`; after reviewing the crop, explicitly approve it with:

```powershell
python main.py --reuse-points --approve-warning
```

Distortion above 6% is always `BLOCKED`. Every item still writes `qc_report.json`, and the batch continues after item-level failures. See `output/run_summary.json` and `output/pickup_measure.log`.

Successful output includes `source.jpg`, `crop_source.png`, `points.json`,
`qc_report.json`, `vehicle.svg`, `annotated.svg`,
`annotation_points.json`, and `measurements.tsv`. The unannotated SVG embeds the source
crop and remains authoritative for exact vehicle dimensions in millimetres.

Phase 2 automatically derives a simplified red upper side-profile and chassis line, then
adds overall length, overall height, bed height, cab height, and neck height. The three
component heights are reported in centimetres and measured relative to the detected chassis
line. Every generated landmark and contour coordinate is saved in millimetres for later
Streamlit review and manual adjustment.

Per-vehicle process artifacts remain under `output/<id>/`. The root-level
`output/measurements.tsv` is the consolidated result and uses the columns `车型`, `车长`,
`车宽`, `CAB高`, `车头高`, `车颈高`, and `车尾高`. Annotated PDF output is intentionally
disabled; Illustrator-ready SVG is the final annotated artifact.

Annotation appearance is controlled in `config.yaml`, including the white background,
vehicle opacity, outline colour and thickness, dimension-line thickness, and font size.
The default outline is 8 mm and the default label size is 82 mm. Phase 2 also detects the
full-height centre door seam from vertical edge continuity, draws it on the cab, and reports
its real distance to the front bumper in both the artwork and `measurements.tsv`.
All displayed values use integer millimetres without a printed unit suffix. Dimension lines
include extension lines back to the measured geometry. The bed/cab intersection is detected
from the left edge of the sustained vertical transition between bed top and cab roof.

## Test

```powershell
pytest -q
```

Streamlit point review and manual annotation adjustment belong to Phase 3 and are not included yet.
