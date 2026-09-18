# SegmentX

**SegmentX: A Multi-Method Image Segmentation and Object Extraction Toolkit**

SegmentX is a modular Computer Vision project for CSE3010. It provides a command-line workflow for image preprocessing, multiple classical segmentation methods, automatic object extraction, and quantitative comparison.

## What it implements

- Image preprocessing: resizing, Gaussian denoising, grayscale conversion, optional CLAHE enhancement.
- K-Means color segmentation in CIELAB space.
- Seeded Region Growing segmentation.
- Canny-based edge segmentation with contour filling.
- Mean-shift filtering followed by compact color-region labeling.
- **Hybrid boundary-aware segmentation:** a project-specific method that combines K-Means color regions with Canny boundary evidence and mask refinement.
- Morphological mask cleanup.
- Connected-component based object extraction.
- Bounding boxes, object crops, centroids, areas, and object counts.
- Comparison tables with processing time and foreground ratio.
- Optional ground-truth evaluation using IoU, Dice, and pixel accuracy.
- Deterministic demo data and automated tests.
- Command-line execution with no GUI requirement.

## Project structure

```text
segmentx/
├── README.md
├── statement.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── kmeans.py
│   ├── region_growing.py
│   ├── edge_segmentation.py
│   ├── mean_shift.py
│   ├── object_extraction.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── pipeline.py
├── tests/
├── scripts/
│   └── generate_demo_data.py
├── data/
│   ├── input/
│   └── ground_truth/
├── outputs/
│   ├── masks/
│   ├── objects/
│   ├── comparisons/
│   └── reports/
├── diagrams/
└── report/
```

## Requirements

- Python 3.9 or newer
- pip
- OpenCV (`opencv-python`)
- NumPy
- pytest for testing

## Installation

### Option A: virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: install as a local package

```bash
pip install -e .
```

## Generate the demo dataset

A deterministic synthetic image and binary ground-truth mask are included in the repository. To regenerate them:

```bash
python scripts/generate_demo_data.py
```

The files are written to:

```text
data/input/demo_objects.png
data/ground_truth/demo_objects_mask.png
```

## Run the project

### Run one method

```bash
python main.py --input data/input/demo_objects.png --method kmeans
```

### Region growing

The seed is given as `x,y` in image coordinates.

```bash
python main.py --input data/input/demo_objects.png --method region-growing --seed 155,150
```

### Edge-based segmentation

```bash
python main.py --input data/input/demo_objects.png --method edge
```

### Mean-shift based segmentation

```bash
python main.py --input data/input/demo_objects.png --method mean-shift
```

### Boundary-aware hybrid segmentation

This project-specific mode combines K-Means color regions with Canny boundary evidence to refine the foreground mask.

```bash
python main.py --input data/input/demo_objects.png --method hybrid
```

### Run all methods and compare them

```bash
python main.py --input data/input/demo_objects.png --method compare
```

### Run all methods with a ground-truth mask

```bash
python main.py \
  --input data/input/demo_objects.png \
  --method compare \
  --ground-truth data/ground_truth/demo_objects_mask.png \
  --seed 155,150
```

### Useful parameters

```text
--clusters                 K-Means cluster count (default: 4)
--seed x,y                 Region-growing seed
--region-threshold         Region-growing intensity threshold (default: 18)
--canny-low                Canny lower threshold (default: 70)
--canny-high               Canny upper threshold (default: 160)
--mean-shift-spatial       Mean-shift spatial radius (default: 16)
--mean-shift-color         Mean-shift color radius (default: 24)
--output                   Output directory (default: outputs)
```

## Output artifacts

For an input named `demo_objects.png`, SegmentX produces files such as:

```text
outputs/
├── masks/
│   ├── demo_objects_kmeans_mask.png
│   ├── demo_objects_region-growing_mask.png
│   ├── demo_objects_edge_mask.png
│   └── demo_objects_mean-shift_mask.png
├── objects/
│   └── <method>/
│       ├── objects.json
│       └── object_XX.png
├── comparisons/
│   ├── <method>_segmentation.png
│   ├── <method>_annotated.png
│   └── demo_objects_comparison.csv
└── reports/
    └── demo_objects_comparison.json
```

The CSV contains measurable values for each run. IoU, Dice, and pixel accuracy are populated only when `--ground-truth` is provided.

## Testing

Run the complete automated test suite from the repository root:

```bash
pytest
```

The test suite covers preprocessing output, K-Means, Region Growing, Canny edges, Mean-Shift output dimensions, Hybrid mask generation, object extraction, and perfect-mask metric behaviour.

## Methodology

### 1. Preprocessing

The pipeline resizes large inputs while preserving aspect ratio, applies Gaussian filtering, converts to grayscale where needed, and can enhance grayscale contrast with CLAHE.

### 2. K-Means

Pixels are represented in CIELAB color space and grouped into a fixed number of color clusters. The dominant border cluster is treated as background for object-extraction purposes.

### 3. Region Growing

A user-supplied seed is expanded to neighbouring pixels while their intensity remains within a specified threshold of the evolving region mean.

### 4. Edge Based

Canny produces an edge map. Morphological closing attempts to close small gaps, after which sufficiently large contours are filled to form candidate foreground regions.

### 5. Mean-Shift Based

OpenCV pyramid mean-shift filtering shifts pixels toward local spatial/color modes. A compact K-Means labeling step is then applied to the mean-shifted image so downstream stages have explicit region labels.

### 6. Boundary-aware Hybrid

The Hybrid mode uses K-Means as a color-region proposal and Canny as boundary evidence. The two cues are combined before morphological refinement. This mode is a project-specific experiment intended to investigate whether a color-only foreground proposal can be made more useful by explicitly considering strong image boundaries.

### 7. Object Extraction

The selected mask is cleaned morphologically. Connected components are measured and filtered by a minimum area ratio. Each retained object receives an ID, bounding box, area and centroid, and is saved as an isolated crop.

### 8. Evaluation

When a binary ground-truth mask is supplied, the system computes foreground IoU, Dice coefficient and pixel accuracy. Without ground truth, the toolkit reports execution time, foreground coverage and extracted-object count instead of inventing accuracy values.

## Limitations and interpretation

- Automatic background selection assumes that the dominant label touching the image border is the background. Images where the foreground touches most borders may need manual preprocessing or a better background strategy.
- Region Growing is seed dependent. The provided center seed is only a convenience default; a domain-appropriate seed should be supplied for meaningful results.
- Edge-to-region conversion works best when boundaries are reasonably closed.
- The Mean-Shift method in this educational toolkit is intentionally hybrid: mean-shift filtering supplies the feature-space smoothing, followed by K-Means labeling for compact explicit regions.
- Quantitative comparison against ground truth is meaningful only when the reference mask follows the same foreground definition as the predicted mask.

## Design artifacts

The `diagrams/` directory contains source `.dot` files and rendered PNGs for:

- System architecture
- Process workflow
- Use case diagram
- Sequence diagram
- Component/class-level view

An ER diagram is not included because SegmentX does not require a persistent relational database for its core workflow; results are stored as image, CSV and JSON artifacts.

## Report

A project report is included at:

```text
report/SegmentX_Project_Report.pdf
```

Before portal submission, replace the placeholder student details on the cover page and update any project-specific screenshots/results if you use a different dataset.

## CSE3010 alignment

The project applies topics from the supplied CSE3010 syllabus, particularly preprocessing/filtering, edge detection, feature/region analysis, segmentation and object detection. The course also includes K-Means and Mean-Shift under pattern analysis and lists segmentation as an indicative experiment.

### Boundary-aware hybrid segmentation

The original four classical methods are kept as independent baselines. SegmentX also includes a project-specific hybrid stage. It first creates a K-Means color-based candidate foreground, detects strong boundaries with Canny, removes pixels that sit directly on those boundaries, recovers enclosed contour regions, and applies light morphology. This creates a fifth experiment that studies how color-region evidence and boundary evidence interact rather than treating the methods as isolated demonstrations.

This extension is intentionally modest: it reuses the tested components already present in the toolkit and adds a separate module (`src/hybrid_segmentation.py`) so the design remains easy to inspect and test.