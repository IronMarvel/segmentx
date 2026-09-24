# Data

The repository includes a small deterministic synthetic dataset so the toolkit can be run and tested immediately without downloading a large external corpus.

- `input/demo_objects.png` - synthetic RGB/BGR image containing four geometric foreground objects.
- `ground_truth/demo_objects_mask.png` - binary foreground mask for the demo image.

To regenerate the files, run:

```bash
python scripts/generate_demo_data.py
```

For a course experiment with real images, place your own images under `data/input/` and, where available, matching binary reference masks under `data/ground_truth/`.
