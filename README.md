# STGL-SAFE
Code structure and sample data for the STGL-SAFE framework, supporting dynamic safety assessment and operational boundary prediction for rescue aircraft in forest fire environments.
This repository provides code structure and sample data for the paper:

**“Safety assessment and operational boundary modeling for rescue aircraft in forest fire environments via spatiotemporal graph learning.”**

The repository is organized to reflect the main components described in the manuscript:

- Forest-fire environment digital twin modeling (Section 3.1)
- Rescue aircraft dynamics and safety operational boundary modeling (Section 3.2)
- Spatiotemporal graph learning–driven risk prediction (Section 4.2)
- STGL-SAFE safety assessment and boundary inference (Section 4.3)
- Case studies and comparative analysis (Section 5)

> **Note:** The code provided here is a minimal, research-oriented reference implementation skeleton. You can extend or replace the stubs with your full implementation.

## Repository Structure

See the directory tree in this repository; key components:

- `data/` – Sample environment tensors and graph structures
- `src/preprocessing/` – Data loading, normalization, spatiotemporal graph construction
- `src/models/` – STGNN architecture (GAT + temporal module + uncertainty head)
- `src/simulation/` – Fire environment model, aircraft dynamics, safety boundary
- `src/training/` – Training loop and loss functions
- `src/evaluation/` – Metrics (RMSE, IoU, ABD) and case runners
- `src/utils/` – Seeding, configuration, visualization helpers
- `examples/` – Scripts to run Case 1–3 on sample data

## Environment Setup

```bash
conda env create -f environment.yml
conda activate stgl-safe
```

or

```bash
pip install -r requirements.txt  # if you create one
```

## Running Examples

Train and evaluate the STGNN on sample data:

```bash
python src/main.py --train
python src/main.py --eval
```

Run the three case studies:

```bash
python examples/run_case1.py
python examples/run_case2.py
python examples/run_case3.py
```
