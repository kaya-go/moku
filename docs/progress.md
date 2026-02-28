# Progress

## Current Status

**Phase**: Dataset creation (Step 1 of 5)

## Completed

- [x] Collected 3 raw COCO datasets from Roboflow
- [x] Analyzed dataset structures, categories, and overlap
- [x] Confirmed v1 is a subset of v10 (skip v1)
- [x] Confirmed go-chess has no overlap with v10 (different source images)
- [x] Defined harmonized category scheme: board (0), black_stone (1), white_stone (2)
- [x] Chose RT-DETR r18vd as the target model
- [x] Set up project structure with `src/moku/` library and notebooks
- [x] Created documentation (AGENTS.md, docs/)
- [x] Built dataset harmonization code in `src/moku/dataset.py`
- [x] Created `01_Build_Dataset.ipynb` notebook

## In Progress

- [ ] Run notebook and push dataset to `kaya-go/moku-v1` on HF Hub

## Next Steps

1. **Push dataset**: Execute `01_Build_Dataset.ipynb` to create and upload `kaya-go/moku-v1`
2. **Training notebook**: Create `02_Train_Model.ipynb` — fine-tune RT-DETR r18vd
3. **Evaluation notebook**: Create `03_Evaluate.ipynb` — compute mAP metrics
4. **ONNX export**: Create `04_Export_ONNX.ipynb` — export and verify ONNX model
5. **Publish model**: Upload ONNX model to `kaya-go/moku-v1` on HF Hub

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-28 | Skip Go game detection v1 dataset | Fully contained in v10 (92/92 base images overlap) |
| 2026-02-28 | Drop empty/corner categories | Inferred from geometry; reduces model complexity |
| 2026-02-28 | Use RT-DETR r18vd | No NMS, small model, HF transformers native, no vendor libs |
| 2026-02-28 | 3 categories: board, black_stone, white_stone | Minimal set sufficient for SGF conversion |
