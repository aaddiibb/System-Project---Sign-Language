Motion-S Phase 1 Export

Canonical files:
splits/train_split.csv
splits/val_split.csv
splits/test_split.csv
sample_manifest.csv
dataset_config.json
split_report.json

Phase 2 must:
1. Load these exact splits. NEVER recreate train/validation/test splits.
2. Build every retrieval index from splits/train_split.csv ONLY.
3. Use validation and test splits only as query/evaluation sets.
4. Attach the original Motion-S dataset separately for BVH and motion-feature files.
5. Resolve bvh_relative_path and motion_feature_relative_path below that attached dataset root.

Columns:
ID: id
Sentence: sentence
Gloss: gloss
RVQ levels: base_tokens, residual_1, residual_2, residual_3, residual_4, residual_5
RVQ cell format: JSON integer array; deserialize with json.loads(), never eval().
Canonical RVQ shape after stacking levels: [L, T] where L=6.

Integrity:
Original annotated rows: 12467
Usable complete rows: 12463
Explicitly quarantined IDs: 1361291, 25980, 32734, 4808283
No motion or token target was synthesized.

Checkpoint:
The provider RVQ-VAE checkpoint is not needed for Phase-2 nearest-neighbor retrieval.
It must be attached and verified before token decoding, reconstruction, or visualization.

Split fingerprint (SHA-256): e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d
