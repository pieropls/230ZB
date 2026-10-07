"""One-time download of the data, models and reference code (only needed to re-run experiments).

    python download.py            # everything (~17 GB)
    python download.py --core     # only what the paper reproduction needs (~11 GB)
    python download.py --check    # report what is already on disk

Data goes to ./data, the official reference code to ./external, models to the
Hugging Face cache (~/.cache/huggingface, or $HF_HOME). Safe to re-run: finished
files are skipped. After this the project runs offline.
"""
import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, scan_cache_dir

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXTERNAL = ROOT / "external"

C4_SHARDS = [0]  # ~30 MB, ~27k news-like articles; we use 2,000
CORE_MODELS = {
    "facebook/opt-1.3b": "generator (paper)",
    "facebook/opt-2.7b": "oracle for perplexity (paper)",
    "google-t5/t5-large": "span-replacement attack (paper)",
}
EXT_MODELS = {
    "Qwen/Qwen2.5-1.5B": "base model for extension 1c (same architecture/tokenizer as the instruct model)",
    "Qwen/Qwen2.5-1.5B-Instruct": "instruction-tuned model (1c) and paraphraser (2a)",
    "sentence-transformers/all-MiniLM-L6-v2": "meaning kept after attacks (2a)",
}
REPOS = {   # official code, pinned to the commit our green lists were validated against
    "lm-watermarking": ("https://github.com/jwkirchenbauer/lm-watermarking.git",
                        "82922516930c02f8aa322765defdb5863d07a00e"),
}
TOKENIZER_FILES = ["*.json", "*.txt", "*.model", "*.spm", "*.tiktoken", "vocab.*", "merges.txt"]


def get_c4():
    """C4 RealNewsLike shard(s) into data/c4."""
    out = []
    for i in C4_SHARDS:
        fn = f"realnewslike/c4-train.{i:05d}-of-00512.json.gz"
        path = hf_hub_download("allenai/c4", fn, repo_type="dataset", local_dir=DATA / "c4")
        with gzip.open(path, "rt") as f:
            n = sum(1 for _ in f)
        out.append(f"{fn}: {n:,} articles")
    return out


def get_parquet_dataset(repo_id, patterns, dest):
    """Parquet files of a Hugging Face dataset into dest."""
    snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns, local_dir=dest)
    files = sorted(Path(dest).rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files matched {patterns} in {repo_id}")
    return [str(p.relative_to(ROOT)) for p in files]


def get_humaneval():
    """HumanEval (164 problems)."""
    return get_parquet_dataset("openai/openai_humaneval", ["*.parquet"], DATA / "humaneval")


def get_triviaqa():
    """TriviaQA validation questions and answer aliases (no evidence documents)."""
    for patterns in (["rc.nocontext/validation-*"], ["unfiltered.nocontext/validation-*"]):
        try:
            return get_parquet_dataset("mandarjoshi/trivia_qa", patterns, DATA / "triviaqa")
        except RuntimeError:
            continue
    raise RuntimeError("TriviaQA validation parquet not found")


def get_model(repo_id):
    """Weights and tokenizer files of a model into the Hugging Face cache."""
    files = HfApi().list_repo_files(repo_id)
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["pytorch_model*.bin"]
    snapshot_download(repo_id, allow_patterns=TOKENIZER_FILES + weights)
    return "safetensors" if weights == ["*.safetensors"] else "bin"


def get_repo(name, url, commit):
    """Clone a git repo into external/ and check out a fixed commit."""
    dest = EXTERNAL / name
    if dest.exists():
        return "already cloned"
    EXTERNAL.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    return f"cloned at {commit[:7]}"


def cache_sizes():
    """Size on disk (GB) of each cached model."""
    try:
        return {r.repo_id: r.size_on_disk / 1e9 for r in scan_cache_dir().repos if r.repo_type == "model"}
    except Exception:
        return {}


def main():
    """Download every item, keep going on failures, write data/MANIFEST.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true", help="skip extension models")
    ap.add_argument("--check", action="store_true", help="report what is on disk, download nothing")
    args = ap.parse_args()

    if args.check:
        sizes = cache_sizes()
        for m in {**CORE_MODELS, **EXT_MODELS}:
            print(f"{'OK ' if m in sizes else '-- '} {m:45s} {sizes.get(m, 0):5.1f} GB")
        for p in sorted(q for q in DATA.rglob("*.*") if ".cache" not in q.parts) if DATA.exists() else []:
            print(f"OK  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.1f} MB")
        return

    DATA.mkdir(exist_ok=True)
    steps = [("C4 RealNewsLike", get_c4), ("HumanEval", get_humaneval), ("TriviaQA", get_triviaqa)]
    steps += [(f"git {n}", lambda n=n, u=u, c=c: get_repo(n, u, c)) for n, (u, c) in REPOS.items()]
    models = {**CORE_MODELS, **({} if args.core else EXT_MODELS)}
    steps += [(f"model {m}", lambda m=m: get_model(m)) for m in models]

    report, failed = {}, []
    for name, fn in steps:
        print(f"\n==> {name}", flush=True)
        try:
            report[name] = fn()
            print(f"    done: {report[name]}")
        except Exception as e:  # keep going; summarise at the end
            failed.append(name)
            report[name] = f"FAILED: {e}"
            print(f"    FAILED: {e}", file=sys.stderr)

    sizes = cache_sizes()
    manifest = {"report": report, "model_sizes_gb": {m: round(sizes.get(m, 0), 2) for m in models}}
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\nSummary")
    for name, r in report.items():
        print(f"  {'FAIL' if name in failed else 'ok  '} {name}: {r}")
    print(f"  models on disk: {sum(manifest['model_sizes_gb'].values()):.1f} GB")
    print("  manifest: data/MANIFEST.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
