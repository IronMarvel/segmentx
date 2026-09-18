# SegmentX - Project Statement

## Problem Statement

Image segmentation is a fundamental Computer Vision task in which an image is divided into meaningful regions. The behaviour of a segmentation method depends on factors such as image intensity, color similarity, noise, and the clarity of object boundaries. Implementations are often demonstrated separately, making it difficult to compare methods under a common workflow and to connect segmentation output to actual object extraction.

SegmentX addresses this problem through a unified command-line toolkit that preprocesses an input image, applies multiple segmentation approaches, converts the resulting regions into usable object masks, extracts individual objects, and records visual and quantitative results.

## Scope

The first version covers:

- Input image validation and preprocessing.
- K-Means color segmentation.
- Seeded Region Growing.
- Canny-based edge segmentation.
- Mean-shift based color-region segmentation.
- Boundary-aware hybrid segmentation combining K-Means and Canny evidence.
- Morphological mask cleanup.
- Connected-component object extraction.
- Bounding boxes, object crops, areas and centroids.
- Runtime/coverage/object-count comparison.
- Optional evaluation with binary ground-truth masks.
- Command-line execution and automated validation tests.

The project intentionally does not require a GUI and does not depend on a relational database.

## Target Users

- Computer Vision students who need to study and compare segmentation approaches.
- Instructors evaluating implementation of segmentation concepts.
- Developers who need a small, reproducible image segmentation baseline for experimentation.

## High-Level Features

1. Standardized preprocessing pipeline.
2. Multiple segmentation algorithms behind one interface.
3. Automatic extraction of connected objects from segmentation masks.
4. Reproducible CLI with configurable parameters.
5. Quantitative comparison and optional ground-truth metrics.
6. Exportable images, crops, JSON metadata and CSV summaries.

## Functional Requirements

### FR1 - Input and preprocessing

The system shall accept a readable image path, validate the input, resize oversized images while preserving aspect ratio, and prepare representations required by the selected segmentation algorithm.

### FR2 - Segmentation

The system shall support K-Means, Region Growing, Edge Based, Mean-Shift Based and a boundary-aware Hybrid segmentation mode.

### FR3 - Object extraction

The system shall convert a segmentation result into a cleaned binary mask, identify connected components, and export retained object crops with metadata.

### FR4 - Comparison and evaluation

The system shall record processing time, foreground coverage and object count. When a compatible binary reference mask is supplied, it shall also report IoU, Dice and pixel accuracy.

### FR5 - CLI execution

All primary project workflows shall be executable from the command line without a GUI-specific setup.

### FR6 - Output persistence

The system shall save masks, annotated images, object crops, JSON metadata and CSV comparisons under a configurable output directory.

## Non-Functional Requirements

### NFR1 - Performance

The toolkit shall resize very large inputs to a configurable maximum working resolution so classical algorithms remain practical on standard student hardware.

### NFR2 - Reliability

Invalid paths, undecodable images and invalid algorithm parameters shall produce clear errors and non-zero process status.

### NFR3 - Maintainability

Segmentation algorithms and utilities shall be separated into focused Python modules with a common orchestration layer.

### NFR4 - Usability

The main workflows shall be discoverable through `python main.py --help` and use explicit command-line arguments.

### NFR5 - Reproducibility

Randomized K-Means operations use a deterministic OpenCV random seed so repeated demo runs are comparable.

### NFR6 - Resource Efficiency

Only compact artifacts required for evaluation are written. Raw input data is not duplicated into every output folder.

### NFR7 - Testability

Core modules shall have automated tests using `pytest`.

## System Workflow

```text
Input image
   -> validation
   -> preprocessing
   -> selected segmentation method
   -> binary/region mask
   -> morphological cleanup
   -> connected components
   -> object metadata + crops
   -> measurements
   -> optional ground-truth metrics
   -> exported results
```

## Intended Outcome

The final toolkit should make the relationship between segmentation method, generated mask, extracted objects and measurable results explicit, providing a reproducible practical Computer Vision application rather than isolated algorithm demonstrations.

## Project-specific design decision

SegmentX keeps the four classical methods as baselines and adds a fifth, boundary-aware Hybrid mode. The Hybrid mode uses K-Means to propose color-based foreground regions and Canny to provide boundary evidence. The candidate mask is refined around strong boundaries before it enters the same object-extraction pipeline as the other methods.

This gives the project one explicitly compositional experiment: instead of comparing techniques only as isolated algorithms, the toolkit also studies how two different visual cues can be combined in a common segmentation pipeline.
