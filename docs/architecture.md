# Architecture

This document describes how the system is structured and how a request flows
through it. The guiding principle is a clean separation between **training**
(offline, produces an artifact) and **inference** (online, consumes the
artifact), with a single shared preprocessing path so the two never diverge.

## High-level flow

```
                            TRAINING (offline)
  ┌───────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐
  │ Synthetic /   │──▶│ Preprocessing│──▶│  SmallCNN │──▶│  Evaluation  │
  │ real dataset  │   │  (transforms)│   │  training │   │  + reports   │
  └───────────────┘   └──────────────┘   └─────┬─────┘   └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ model.pt     │  weights + metadata
                                        │ (checkpoint) │  (classes, norm stats)
                                        └──────┬───────┘
                                               │
                            INFERENCE (online) │
  ┌───────────┐   ┌──────────────┐   ┌─────────▼────┐   ┌──────────────┐
  │  Image    │──▶│ Validation + │──▶│   Predictor  │──▶│ JSON response│
  │  upload   │   │ Preprocessing│   │ (load + fwd) │   │ + disclaimer │
  └───────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │                                    │
        │                                    └──▶ optional Grad-CAM overlay (PNG)
        ▼
   FastAPI  /  Streamlit  /  CLI  — all go through the same Predictor
```

## Layers

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Configuration | `config.py` | Typed config from YAML + env overrides. Single source of truth for training and serving. |
| Constants | `constants.py` | Class taxonomy, image conventions, disclaimer. Prevents label drift. |
| Preprocessing | `preprocessing/transforms.py` | Image → tensor. **Shared** by training and inference. |
| Data | `data/synthetic.py`, `data/dataset.py` | Deterministic synthetic generator + ImageFolder loader. |
| Model | `models/cnn.py` | `SmallCNN` architecture and a `build_model` factory. |
| Training | `training/train.py`, `training/evaluate.py` | Training loop with early stopping; metrics + confusion matrix. |
| Inference | `inference/checkpoint.py`, `inference/predictor.py`, `inference/gradcam.py` | Checkpoint I/O, the `Predictor`, and Grad-CAM. |
| API | `api/main.py`, `api/schemas.py`, `api/dependencies.py` | FastAPI app, response models, model singleton. |
| Utils | `utils/logging.py`, `utils/seed.py` | Logging and reproducible seeding. |

## Why train/serve share one preprocessing function

A common source of silent production bugs in ML systems is *training/serving
skew* — the model is fed slightly different inputs in production than it saw in
training. This project eliminates that class of bug by routing both paths
through `preprocess_pil`, and by storing the normalization statistics **inside**
the checkpoint. At serve time the `Predictor` reads those exact statistics from
the model file, so it is impossible to normalize inference inputs differently
from training inputs.

## The checkpoint as a contract

`model.pt` is not just weights — it bundles:

- `state_dict`: the trained parameters.
- `metadata`: architecture name, class keys/order, image size, normalization
  mean/std, and the metrics recorded at training time.

This makes the artifact self-describing: the serving layer never hard-codes the
class list or the normalization; it reads them from the file. Retraining on a
different dataset (e.g. a real one) produces a new self-consistent artifact with
no code changes.

## Resilient startup

The API loads the model once at startup. If the checkpoint is missing, the
service still boots and reports `model_loaded: false` on `/health`; prediction
endpoints then return a clean `503` instead of crashing. This lets the container
start before a model volume is mounted, which is useful in orchestrated
deployments.
