# Progress

## Current Status

**Phase**: Training (Step 2 of 5)

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
- [x] Created training utilities in `src/moku/training.py`
- [x] Created `02_Train_Model.ipynb` — fine-tune RT-DETR locally with HF Trainer
- [x] Created `03_Evaluate.ipynb` — mAP evaluation, prediction visualization, run comparison

## In Progress

- [ ] Run `01_Build_Dataset.ipynb` and push dataset to `kaya-go/moku-v1` on HF Hub
- [ ] Run `02_Train_Model.ipynb` baseline training
- [ ] Run HP tuning sweep (6-8 configs: LR × weight decay)

## Next Steps

1. **Push dataset**: Execute `01_Build_Dataset.ipynb` to upload `kaya-go/moku-v1`
2. **Baseline training**: Run `02_Train_Model.ipynb` with default HPs
3. **HP tuning**: Run sweep configs, compare in `03_Evaluate.ipynb`, pick best
4. **Push best model**: Push best checkpoint to `kaya-go/moku-v1` on HF Hub
5. **ONNX export**: Create `04_Export_ONNX.ipynb` — export and verify ONNX model
6. **Publish model**: Upload ONNX model to `kaya-go/moku-v1` on HF Hub

## HP Tuning Strategy

- **Grid search** on `learning_rate` × `weight_decay` (4 × 2 = 8 configs)
- LR: `[5e-5, 1e-4, 2e-4, 5e-4]`, WD: `[1e-4, 1e-3]`
- 50 epochs each with cosine LR schedule + early stopping via `load_best_model_at_end`
- Compare using eval_loss curves and test mAP in `03_Evaluate.ipynb`
- For GPU runs: change `RUN_NAME` in `02_Train_Model.ipynb`, download results to `runs/`

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-28 | Skip Go game detection v1 dataset | Fully contained in v10 (92/92 base images overlap) |
| 2026-02-28 | Drop empty/corner categories | Inferred from geometry; reduces model complexity |
| 2026-02-28 | Use RT-DETR r18vd | No NMS, small model, HF transformers native, no vendor libs |
| 2026-02-28 | 3 categories: board, black_stone, white_stone | Minimal set sufficient for SGF conversion |
| 2026-02-28 | Simple grid search over LR × WD | Small dataset; LR is most impactful HP; 8 runs is manageable |
| 2026-02-28 | Random horizontal flip as sole augmentation | Minimal viable augmentation; more can be added later |
