import argparse
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def parse_args(argv=None, known_only=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CaseFlow Encoder: Causal KG Encoder Pre-training"
    )

    parser.add_argument("--pretrain_dataset", type=str, default="grailqa",
                        help="CaseFlow Encoder pretrain source: 'cwq', 'grailqa', or path to .jsonl file")
    parser.add_argument("--depth", type=int, default=3,
                        help="Number of hops for subgraph exploration (1-3)")
    parser.add_argument("--width", type=int, default=3,
                        help="Top-K relations to keep per hop (must be >= 1)")
    parser.add_argument("--llm", type=str, default="llama-3",
                        help="LLM backend for relation sampling")
    parser.add_argument("--limit_llm_in", type=int, default=8192,
                        help="Max LLM input tokens")
    parser.add_argument("--limit_llm_out", type=int, default=512,
                        help="Max LLM output tokens")
    parser.add_argument("--max_retry", type=int, default=5,
                        help="Retries for LLM format mismatches")
    parser.add_argument("--tau_infonce", type=float, default=0.2,
                        help="InfoNCE temperature τ for L_C and L_Y (typical: 0.07~0.2)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM sampling temperature for relation sampling (run_llm)")
    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI API key (empty = use local Llama)")

    parser.add_argument("--sbert_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model")
    parser.add_argument("--emb_dir", type=str,
                        default=str(PROJECT_ROOT / "embeddings"),
                        help="Path to pre-baked SBERT embedding store")
    parser.add_argument("--emb_prefix", type=str, default="grailqa",
                        help="Embedding file prefix")

    parser.add_argument("--hidden", type=int, default=384,
                        help="Hidden dimension (auto-set to BERT dim if not specified)")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of GCN layers in BIG module")
    parser.add_argument("--drop_out", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--att_temperature", type=float, default=1.0,
                        help="Temperature τ for AttCov softmax (>1.0 = softer attention, "
                             "prevents 0/1 polarization of α_c/α_s. Recommended: 1.0~2.0)")

    parser.add_argument("--att_loss", type=str2bool, default=False,
                        help="Enable attention-separation loss in Ls_total.")
    parser.add_argument("--att_loss_threshold", type=float, default=5.0,
                        help="Apply attention loss only if it exceeds this threshold.")

    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs over the dataset")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (Adam)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="L2 regularization")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Graphs per batch during training")
    parser.add_argument("--train_model", type=str, default="infonce_mgda",
                        choices=["infonce", "infonce_mgda"],
                        help="Training mode: infonce=InfoNCE simple sum, infonce_mgda=InfoNCE+MGDA Pareto")
    parser.add_argument("--mgda_model", type=str, default="loss+",
                        choices=["loss", "loss+", "l2"],
                        help="Gradient normalization mode for MGDA")
    parser.add_argument("--c", type=float, default=1.0,
                        help="Weight for L_C")
    parser.add_argument("--o", type=float, default=1.0,
                        help="Weight for L_S")
    parser.add_argument("--co", type=float, default=1.0,
                        help="Weight for L_Y")
    parser.add_argument("--lambda_var", type=float, default=0.3,
                        help="Weight for L_S variance regularization.")

    parser.add_argument("--save_path", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "encoder.pth"),
                        help="Path to save pre-trained CausalEncoder weights")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N questions (0=disabled)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_questions", type=int, default=0,
                        help="Limit number of questions (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="CUDA device id")

    if known_only:
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)

    if args.width <= 0:
        raise ValueError("--width must be >= 1.")
    setup_seed(args.seed)
    return args


def parse_caseflow_train_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaseFlow training")

    parser.add_argument("--dataset_name", type=str, default="webqsp", choices=["webqsp", "cwq", "metaqa3hop"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, required=True)

    parser.add_argument("--llm", type=str, default="llama-3")
    parser.add_argument("--limit_llm_in", type=int, default=8192)
    parser.add_argument("--limit_llm_out", type=int, default=512)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--emb_dir", type=str, default=str(PROJECT_ROOT / "embeddings"))
    parser.add_argument("--emb_prefix", type=str, default="")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encoder_path", type=str, default=str(PROJECT_ROOT / "checkpoints" / "encoder.pth"))

    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--kmeans_iters", type=int, default=5)
    parser.add_argument("--sim_one", type=float, default=0.85)
    parser.add_argument("--k1_quantile", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--n_rollouts", type=int, default=16)
    parser.add_argument("--gflow_alpha", type=float, default=1.0)
    parser.add_argument("--gflow_beta", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--w_subtb", type=float, default=1.0)
    parser.add_argument("--w_hinge", type=float, default=0.05)
    parser.add_argument("--disable_hinge_loss", action="store_true")
    parser.add_argument("--hinge_warmup_steps", type=int, default=3000)
    parser.add_argument("--gnn_hidden", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--min_chunks", type=int, default=1)
    parser.add_argument("--gflow_temp", type=float, default=1.0)
    parser.add_argument("--rand_explore_prob", type=float, default=0.05)
    parser.add_argument("--pcau_floor", type=float, default=0.05)
    parser.add_argument("--pcau_weight", type=float, default=1.0)
    parser.add_argument("--lambda_warmup_steps", type=int, default=1000)

    parser.add_argument("--hinge_pair_topk", type=int, default=0)
    parser.add_argument("--eval_pair_mode", type=str, default="all", choices=["all", "actual_vs_others"])
    parser.add_argument("--hinge_pair_gap", type=float, default=0.0)
    parser.add_argument("--hinge_margin", type=float, default=0.1)
    parser.add_argument("--hinge_max_pairs_per_state", type=int, default=4)
    parser.add_argument("--hinge_pair_horizon", type=int, default=2)

    parser.add_argument("--save_path", type=str, default=str(PROJECT_ROOT / "checkpoints" / "caseflow_policy.pth"))
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args(argv)
    if args.width <= 0:
        raise ValueError("--width must be >= 1.")
    if args.hinge_pair_horizon <= 0:
        raise ValueError("--hinge_pair_horizon must be >= 1.")
    return args


def parse_caseflow_infer_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaseFlow inference")

    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "caseflow_policy.pth"))
    parser.add_argument("--dataset_name", type=str, default="webqsp", choices=["webqsp", "cwq", "metaqa3hop"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "output" / "caseflow_infer.jsonl"))
    parser.add_argument("--sim_one", type=float, default=None)
    parser.add_argument("--k1_quantile", type=float, default=None)

    parser.add_argument("--emb_dir", type=str, default=str(PROJECT_ROOT / "embeddings"))
    parser.add_argument("--emb_prefix", type=str, default="")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encoder_path", type=str, default=str(PROJECT_ROOT / "checkpoints" / "encoder.pth"))

    parser.add_argument("--llm", type=str, default="llama-3")
    parser.add_argument("--limit_llm_in", type=int, default=8192)
    parser.add_argument("--limit_llm_out", type=int, default=512)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--gflow_alpha", type=float, default=1.0)
    parser.add_argument("--gflow_beta", type=float, default=1.0)
    parser.add_argument("--min_chunks", type=int, default=2)
    parser.add_argument("--gflow_temp", type=float, default=1.0)
    parser.add_argument("--top_b2", type=int, default=4)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--pcau_floor", type=float, default=0.05)
    parser.add_argument("--pcau_weight", type=float, default=1.0)
    parser.add_argument("--gnn_layers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args(argv)
    if args.width <= 0:
        raise ValueError("--width must be >= 1.")
    return args


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args: argparse.Namespace, width: int = 70):
    print("=" * width)
    print("CaseFlow Encoder Pre-training Configuration")
    print("=" * width)
    for k, v in vars(args).items():
        dots = "." * max(1, width - len(k) - len(str(v)) - 2)
        print(f"  {k}{dots}{v}")
    print("=" * width)
