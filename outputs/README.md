# Outputs

This folder contains generated artifacts from the included demo run. New runs create:

- `masks/` - binary segmentation masks.
- `objects/` - object crops and JSON metadata by method.
- `comparisons/` - annotated images, segmentation visualizations and CSV comparisons.
- `reports/` - JSON comparison summaries.

To regenerate all demo outputs:

```bash
python main.py --input data/input/demo_objects.png --method compare --ground-truth data/ground_truth/demo_objects_mask.png --seed 155,150
```
