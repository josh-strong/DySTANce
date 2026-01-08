## Dynamic tool evaluation splits (reproducible)

Context: we have 18 tool predictors (derived from the CSV basenames in `data/openi/predictions/`). We need two settings:
- Train + test on all tools (baseline).
- Train on a subset of tools and evaluate on unseen tools (stress-test generalisation).

### Tool inventory (canonical names)
`densenet_medical_mae_pt_openi`, `densenet_mocov2_pt_openi`, `evax_base_cxr__pt_openi`, `evax_small_cxr__pt_openi`, `evax_tiny_cxr__pt_openi`, `resnet_biovil_pt_openi`, `resnet_medklip_pt_openi`, `resnet_mgca_pt_openi`, `evax_small_chexpert_pt_openi`, `evax_tiny_chexpert_pt_openi`, `densenet121_res224_chex`, `densenet121_res224_pc`, `densenet121_res224_mimic_ch`, `densenet121_res224_mimic_nb`, `resnet50_res512_all`, `densenet121_res224_all`, `densenet121_res224_nih`, `densenet121_res224_rsna`.

### Sampling rationale
- We fix deterministic seeds to avoid re-rolling splits and to make evaluation reproducible.
- Tools are alphabetically sorted before shuffling to avoid seed-dependent ordering quirks.
- We allocate 12 train tools / 6 unseen test tools (two-thirds vs one-third) to keep a varied train panel while holding out a meaningful unseen set.
- Use the held-out tools for both validation and test when measuring unseen-tool generalisation (i.e., keep train/val purely on the train-tool subset; test only on the held-out tool predictions).

### Splits (train tools → test tools)
- Split A (seed 2024):  
  Train: densenet121_res224_all, evax_tiny_chexpert_pt_openi, resnet_medklip_pt_openi, densenet121_res224_chex, evax_small_chexpert_pt_openi, resnet_mgca_pt_openi, densenet121_res224_mimic_ch, densenet_medical_mae_pt_openi, resnet50_res512_all, densenet_mocov2_pt_openi, densenet121_res224_nih, evax_tiny_cxr__pt_openi  
  Test: densenet121_res224_rsna, evax_small_cxr__pt_openi, densenet121_res224_mimic_nb, evax_base_cxr__pt_openi, densenet121_res224_pc, resnet_biovil_pt_openi

- Split B (seed 2025):  
  Train: densenet_medical_mae_pt_openi, resnet50_res512_all, evax_small_cxr__pt_openi, densenet121_res224_nih, evax_tiny_chexpert_pt_openi, evax_small_chexpert_pt_openi, evax_tiny_cxr__pt_openi, densenet121_res224_chex, densenet121_res224_mimic_nb, evax_base_cxr__pt_openi, densenet121_res224_rsna, densenet121_res224_pc  
  Test: densenet121_res224_all, densenet_mocov2_pt_openi, resnet_medklip_pt_openi, resnet_biovil_pt_openi, densenet121_res224_mimic_ch, resnet_mgca_pt_openi

- Split C (seed 2026):  
  Train: densenet121_res224_pc, densenet121_res224_mimic_ch, densenet121_res224_all, densenet121_res224_chex, evax_tiny_cxr__pt_openi, densenet121_res224_nih, evax_small_cxr__pt_openi, densenet_medical_mae_pt_openi, resnet_medklip_pt_openi, densenet121_res224_rsna, densenet_mocov2_pt_openi, evax_tiny_chexpert_pt_openi  
  Test: evax_base_cxr__pt_openi, resnet_biovil_pt_openi, resnet50_res512_all, resnet_mgca_pt_openi, evax_small_chexpert_pt_openi, densenet121_res224_mimic_nb

- Split D (seed 2027):  
  Train: resnet_mgca_pt_openi, resnet_medklip_pt_openi, evax_tiny_chexpert_pt_openi, densenet_mocov2_pt_openi, densenet121_res224_chex, densenet121_res224_nih, densenet_medical_mae_pt_openi, evax_base_cxr__pt_openi, evax_small_cxr__pt_openi, resnet_biovil_pt_openi, evax_tiny_cxr__pt_openi, densenet121_res224_rsna  
  Test: densenet121_res224_pc, evax_small_chexpert_pt_openi, densenet121_res224_all, densenet121_res224_mimic_ch, resnet50_res512_all, densenet121_res224_mimic_nb

- Split E (seed 2028):  
  Train: densenet121_res224_chex, densenet_mocov2_pt_openi, evax_tiny_cxr__pt_openi, evax_base_cxr__pt_openi, resnet_mgca_pt_openi, densenet121_res224_rsna, densenet_medical_mae_pt_openi, evax_tiny_chexpert_pt_openi, densenet121_res224_mimic_nb, resnet_biovil_pt_openi, evax_small_chexpert_pt_openi, densenet121_res224_mimic_ch  
  Test: densenet121_res224_nih, resnet50_res512_all, resnet_medklip_pt_openi, densenet121_res224_all, densenet121_res224_pc, evax_small_cxr__pt_openi

### Usage notes
- Setting (1) “all tools”: keep full tool set for train/val/test.
- Setting (2) “unseen tools”: for each split, restrict train/val to the listed train tools; hold-out tool predictions should appear only in test. Repeat across the five splits and report mean/variance.
- If you need a different ratio, rerun `python - <<'PY'` with the sorted tool list and adjust `train_size`; seeds above keep ordering deterministic.

