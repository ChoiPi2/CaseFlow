# CaseFlow: Causality-Guided Spurious Evidence Pruning with GFlowNet for Faithful Graph Retrieval

## Overview
CaseFlow disentangles causally relevant subgraph chunks from spurious ones in knowledge graph question answering,
using a GFlowNet pruning policy trained with trajectory-level flow matching and hinge loss.

Pipeline:
1. **CausalEncoder pre-training** — InfoNCE losses (L_C, L_S, L_Y) + MGDA balancing
2. **Adaptive spherical K-means chunking** — relation-local chunk construction
3. **GFlowNet pruning** — SubTB loss + hinge loss

## Repository Structure

```text
train.py
infer.py
chunking.py
gflownet.py
rollout.py
prompts.py
utils.py

CausalEncoder/
  pretrain.py
  model.py
  losses.py
  gcn_conv.py

graph_builder.py
freebase.py
opts.py
requirements.txt
```

## Requirements
```
pip install -r requirements.txt
```

## Data And External Resources
CaseFlow expects the following resources to be prepared locally:

1. A Freebase-compatible SPARQL endpoint.
2. Question-answering datasets in HuggingFace `datasets` format.
3. Entity and relation embedding files used by `OfflineEmbeddingStore`.
4. A pre-trained encoder checkpoint for CaseFlow policy training and inference.
5. A trained CaseFlow policy checkpoint for inference.

## Encoder Pre-training
```bash
python CausalEncoder/pretrain.py \
  --pretrain_dataset grailqa \
  --depth 3 \
  --width 3 \
  --emb_dir embeddings \
  --emb_prefix grailqa \
  --save_path checkpoints/encoder.pth \
  --gpu_id 0
```


## CaseFlow Policy Training
After preparing an encoder checkpoint, train the pruning policy with:

```bash
python train.py \
  --dataset_name cwq \
  --split train \
  --depth 3 \
  --width 3 \
  --encoder_path checkpoints/encoder.pth \
  --emb_dir embeddings \
  --emb_prefix cwq \
  --save_path checkpoints/caseflow.pth \
  --gpu_id 0
```

## Inference
```bash
python infer.py \
  --dataset_name cwq \
  --split test \
  --depth 3 \
  --width 3 \
  --checkpoint checkpoints/caseflow_policy.pth \
  --encoder_path checkpoints/encoder.pth \
  --emb_dir embeddings \
  --emb_prefix cwq \
  --output output/caseflow_cwq_test.jsonl \
  --gpu_id 0
```


## Citation
Citation information will be added after the review process.
