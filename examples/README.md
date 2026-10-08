# Examples

`example_inference.py` scores one stored recording with the released ensemble
(`models/release/`), prints each output with a short explanation, and writes
the files of `scripts/infer.py` to `expected_output/<clip_id>/`:

```bash
python examples/example_inference.py asd_1
```

To score a video instead:

```bash
python scripts/infer.py --input path/to/video.mp4 --checkpoint models/release --output outputs/my_video
```

`input/` is a place for your own videos or `(300, 33, 2)` arrays. See
`docs/inference.md` for the meaning of every output field.
