"""CLI: python run.py <step or experiment> [--n N] [--only SUBSTRING]. `python run.py list` shows the pipeline."""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

import core as C
import params as P
import stats as S


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def opt_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(P.MODELS["opt"])


# ---- setup and checks ------------------------------------------------------------------------
def prompts(args):
    """C4 RealNewsLike prompts and human baselines (results/prompts_c4.jsonl.gz)."""
    rows = C.build_c4_prompts(opt_tokenizer())
    if os.path.exists(C.PROMPTS_FILE):
        os.remove(C.PROMPTS_FILE)
    C.append_jsonl(C.PROMPTS_FILE, rows)
    L = [len(r["prompt_ids"]) for r in rows]
    log(f"{len(rows)} prompts; prompt tokens median {np.median(L):.0f}, mean {np.mean(L):.0f}, max {max(L)}")


def equiv(args):
    """Our green lists vs the official WatermarkBase._get_greenlist_ids (must PASS)."""
    sys.path.insert(0, os.path.join(C.ROOT, "external", "lm-watermarking"))
    from watermark_processor import WatermarkBase
    from transformers import AutoTokenizer
    lines, ok_all = [], True
    for key, n in (("opt", 1000), ("qwen", 200)):
        tok = AutoTokenizer.from_pretrained(P.MODELS[key])
        V = C.vocab_size(tok)
        prevs = np.random.default_rng(1).integers(0, V, n)
        for gamma in (0.25, 0.5):
            ref = WatermarkBase(vocab=list(tok.get_vocab().values()), gamma=gamma, seeding_scheme="simple_1")
            ref.rng = torch.Generator(device="cpu")
            n_ok = 0
            for prev in prevs:
                theirs = ref._get_greenlist_ids(torch.tensor([int(prev)])).numpy()
                ours = C.greenlist_ids(int(prev), V, gamma)
                same_set = np.array_equal(np.sort(theirs), ours)
                same_order = np.array_equal(theirs, np.argsort(C.rank_table(int(prev), V))[:len(theirs)])
                n_ok += same_set and same_order
            ok = n_ok == n
            ok_all &= ok
            lines.append(f"{P.MODELS[key]:28s} V={V:7d} gamma={gamma:4.2f}  {n_ok}/{n} previous-token ids "
                         f"give identical green lists (same ids, same order)  {'PASS' if ok else 'FAIL'}")
            log(lines[-1])
    head = ["Green-list equivalence: core.greenlist_ids vs external/lm-watermarking/watermark_processor.py",
            "WatermarkBase._get_greenlist_ids (simple_1 seeding, hash key 15485863, CPU torch.Generator).",
            f"torch {torch.__version__}", ""]
    lines.append("\nOVERALL: " + ("PASS" if ok_all else "FAIL"))
    with open(os.path.join(C.SUMMARY, "greenlist_equivalence.txt"), "w") as f:
        f.write("\n".join(head + lines) + "\n")
    if not ok_all:
        sys.exit(1)


def human(args):
    """Green ranks and z of the human baseline completions (context = last prompt token)."""
    tok = opt_tokenizer()
    V = C.vocab_size(tok)
    path = C.raw_path("human", "c4")
    if os.path.exists(path):
        os.remove(path)
    rows = []
    for r in C.read_jsonl(C.PROMPTS_FILE):
        prev0 = r["prompt_ids"][-1]
        d = C.detect(r["baseline_ids"], prev0, V, 0.5)
        re_ids = tok(r["baseline_text"], add_special_tokens=False).input_ids
        rows.append(dict(idx=r["idx"], T=len(r["baseline_ids"]), ids=r["baseline_ids"], ranks=d["ranks"],
                         z=d["z"], z_norep=d["z_norep"], z_reenc=C.detect(re_ids, prev0, V, 0.5)["z"],
                         text=r["baseline_text"]))
    C.append_jsonl(path, rows)
    z = np.array([r["z"] for r in rows])
    log(f"human baselines: n={len(z)} z mean {z.mean():.3f} sd {z.std():.3f}, share z>4: {(z > 4).mean():.4f}")


def check(args):
    """Smoke test: fp16 logits, 16 watermarked and 16 unwatermarked generations, padding-invariant NLL."""
    model, tok = C.load_lm("opt")
    rows = C.read_jsonl(C.PROMPTS_FILE)[:16]
    ids, att = C.left_pad([r["prompt_ids"] for r in rows[:4]], tok.pad_token_id)
    with torch.no_grad():
        lg16 = model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE)).logits.float()
    finite = bool(torch.isfinite(lg16[att.bool().to(C.DEVICE)]).all())
    m32, _ = C.load_lm("opt", dtype=torch.float32)
    with torch.no_grad():
        lg32 = m32(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE)).logits.float()
    a = att.bool().to(C.DEVICE)
    agree = float((lg16.argmax(-1)[a] == lg32.argmax(-1)[a]).float().mean())
    maxdiff = float((lg16.log_softmax(-1) - lg32.log_softmax(-1))[a].abs().max())
    del m32
    C.free()
    log(f"fp16 logits finite: {finite}; argmax agreement fp16 vs fp32: {agree:.4f}; max |dlogp|: {maxdiff:.3f}")

    V, eos = C.vocab_size(tok), C.eos_ids(model, tok)
    prompts_ = [dict(idx=r["idx"], ids=r["prompt_ids"]) for r in rows]
    out = {}
    for delta in (2.0, 0.0):
        cfg = P.C(gamma=0.5, delta=delta, n=16)
        t0 = time.time()
        res = []
        for b in C.make_batches(prompts_, 1, P.MAX_NEW):
            res += C.generate_batch(model, tok, b, cfg, V, eos)
        out[delta] = res
        z = np.array([r["z"] for r in res])
        log(f"delta={delta}: z = {np.round(np.sort(z), 2).tolist()}  ({time.time() - t0:.1f}s)")
    log(f"delta=2: {sum(r['z'] > 4 for r in out[2.0])}/16 with z > 4; delta=0: "
        f"{sum(r['z'] > 4 for r in out[0.0])}/16")
    hz = [C.detect(r["baseline_ids"], r["prompt_ids"][-1], V, 0.5)["z"] for r in rows]
    log(f"human z = {np.round(np.sort(hz), 2).tolist()}")
    items = [(r["prompt_ids"], r["baseline_ids"]) for r in rows[:4]]
    batched = C.nll_batch(model, items, tok.pad_token_id)
    single = [C.nll_batch(model, [it], tok.pad_token_id)[0] for it in items]
    log(f"NLL batched {np.round(batched, 4).tolist()} vs single {np.round(single, 4).tolist()}")
    path = os.path.join(C.SUMMARY, "phase0_check.jsonl")
    if os.path.exists(path):
        os.remove(path)
    C.append_jsonl(path, [dict(fp16_finite=finite, argmax_agree=agree, z_wm=[r["z"] for r in out[2.0]],
                               z_nowm=[r["z"] for r in out[0.0]], z_human=hz, nll_batched=batched,
                               nll_single=single)])


# ---- generation experiments (configs in params.EXPERIMENTS) ---------------------------------------
def generate(exp, args):
    """Run every config of an experiment, one model in memory at a time."""
    cfgs = P.EXPERIMENTS[exp][1]
    if args.only:
        cfgs = [c for c in cfgs if args.only in c["id"]]
    for key in dict.fromkeys(c["model"] for c in cfgs):
        model, tok = C.load_lm(key)
        for cfg in cfgs:
            if cfg["model"] == key:
                C.run_config(dict(cfg, n=min(cfg["n"], args.n)) if args.n else cfg, model, tok, log=log)
        del model
        C.free()


def keys(args):
    """Length side effect: gamma 0.5, delta 2 on the first 100 prompts under 3 other hash keys."""
    model, tok = C.load_lm("opt")
    for key in P.OTHER_KEYS:
        P.HASH_KEY = key                      # this process only; green lists are rebuilt for the new key
        C.rank_table.cache_clear()
        C.green_mask.cache_clear()
        cfg = dict(P.C(gamma=0.5, delta=2, n=P.N_KEYS), keep=False)
        cfg["id"] += f"-key{key}"
        C.run_config(cfg, model, tok, log=log)
    del model
    C.free()


# ---- attacks --------------------------------------------------------------------------------------
def attack_sources(beams=False, n=P.N_ATTACK):
    """The first n watermarked texts (gamma 0.5, delta 2; kept T) and their human baselines."""
    cid = "opt-news-beam8-d2-g0.5" if beams else "opt-news-m-d2-g0.5"
    gen = sorted(C.read_jsonl(C.gen_path(cid)), key=lambda r: r["idx"])
    if not beams:
        gen = [r for r in gen if P.T_KEEP[0] <= r["T"] <= P.T_KEEP[1]]
    gen = gen[:n]
    out = [dict(src="beam" if beams else "wm", idx=r["idx"], text=r["text"], ids=r["ids"]) for r in gen]
    if not beams:
        hum = {r["idx"]: r for r in C.read_jsonl(C.raw_path("human", "c4"))}
        out += [dict(src="human", idx=r["idx"], text=hum[r["idx"]]["text"], ids=hum[r["idx"]]["ids"]) for r in gen]
    return out


def para(args):
    """Ext. 2a: paraphrase attack with Qwen2.5-1.5B-Instruct at three strengths (watermarked + human texts)."""
    path = C.raw_path("attacks", "para")
    done = {r["key"] for r in C.read_jsonl(path)}
    model, tok = C.load_lm("qwen-it")
    srcs = attack_sources()
    for strength in P.PARA_PROMPTS:
        todo = [s for s in srcs if f"para-{s['src']}-{s['idx']}-{strength}" not in done]
        for c in range(0, len(todo), 32):
            chunk = todo[c:c + 32]
            out = C.paraphrase(model, tok, [s["text"] for s in chunk], strength, seed=c)
            C.append_jsonl(path, [dict(key=f"para-{s['src']}-{s['idx']}-{strength}", src=s["src"], idx=s["idx"],
                                       strength=strength, text=t) for s, t in zip(chunk, out)])
            log(f"  para {strength}: {c + len(chunk)}/{len(todo)}")
    del model
    C.free()


def t5(args):
    """Table 9: T5 span attack with multi-word fills, eps 0.1 / 0.3 (watermarked + human texts)."""
    path = C.raw_path("attacks", "t5")
    done = {(r["src"], r["idx"]) for r in C.read_jsonl(path) if r["eps"] == P.T5_EPS_MULTI[-1]}
    srcs = [s for s in attack_sources() if (s["src"], s["idx"]) not in done]
    log(f"t5: {len(srcs)} sequences to attack")
    model, tok = C.load_t5()
    for c in range(0, len(srcs), P.T5_CHUNK):
        chunk = srcs[c:c + P.T5_CHUNK]
        states = [C.t5_state(s["text"], s["idx"] + (10 ** 6 if s["src"] == "human" else 0)) for s in chunk]
        snaps = C.t5_attack(model, tok, states, P.T5_EPS_MULTI, log=log)
        C.append_jsonl(path, [dict(key=f"t5-{s['src']}-{s['idx']}-{e}", src=s["src"], idx=s["idx"], eps=e,
                                   succ=st["succ"], att=st["att"], text="".join(st["pieces"]), state=st)
                              for e in P.T5_EPS_MULTI for s, st in zip(chunk, snaps[e])])
        log(f"  t5 chunk {c // P.T5_CHUNK + 1}/{-(-len(srcs) // P.T5_CHUNK)}")
    del model
    C.free()


def t5w(args):
    """Table 9: T5 span attack, single-word replacements (paper's rule), eps 0.1-0.7, 100 + 100 texts."""
    path = C.raw_path("attacks", "t5w")
    done = {(r["src"], r["idx"]) for r in C.read_jsonl(path) if r["eps"] == P.T5_EPS[-1]}
    srcs = [s for s in attack_sources(n=P.N_ATTACK_SINGLE) if (s["src"], s["idx"]) not in done]
    log(f"t5w: {len(srcs)} sequences to attack")
    model, tok = C.load_t5()
    for c in range(0, len(srcs), P.T5_CHUNK):
        chunk = srcs[c:c + P.T5_CHUNK]
        states = [C.t5_state(s["text"], s["idx"] + (10 ** 6 if s["src"] == "human" else 0)) for s in chunk]
        snaps = C.t5_attack(model, tok, states, P.T5_EPS, log=log, single_word=True)
        C.append_jsonl(path, [dict(key=f"t5w-{s['src']}-{s['idx']}-{e}", src=s["src"], idx=s["idx"], eps=e,
                                   succ=st["succ"], att=st["att"], text="".join(st["pieces"]))
                              for e in P.T5_EPS for s, st in zip(chunk, snaps[e])])
        log(f"  t5w chunk {c // P.T5_CHUNK + 1}/{-(-len(srcs) // P.T5_CHUNK)}")
    del model
    C.free()


def score(args):
    """Re-tokenize attacked texts with OPT; z, token edit share and MiniLM similarity (attacks/scored)."""
    tok = opt_tokenizer()
    V = C.vocab_size(tok)
    prompts_ = {r["idx"]: r["prompt_ids"] for r in C.read_jsonl(C.PROMPTS_FILE)}
    srcs = {(s["src"], s["idx"]): s for s in attack_sources() + attack_sources(beams=True)}
    rows = [dict(key=f"none-{s['src']}-{s['idx']}", attack="none", level=0, src=s["src"], idx=s["idx"],
                 text=s["text"]) for s in srcs.values()]       # unattacked references, re-encoded the same way
    for name in ("t5", "t5w"):
        for r in C.read_jsonl(C.raw_path("attacks", name)):
            rows.append(dict(key=r["key"], attack=name, level=r["eps"], src=r["src"], idx=r["idx"], text=r["text"],
                             succ=r["succ"]))
    for r in C.read_jsonl(C.raw_path("attacks", "para")):
        rows.append(dict(key=r["key"], attack="para", level=r["strength"], src=r["src"], idx=r["idx"], text=r["text"]))
    rows = [r for r in rows if (r["src"], r["idx"]) in srcs]
    emb_o = C.embed([srcs[(r["src"], r["idx"])]["text"] for r in rows])
    emb_a = C.embed([r["text"] for r in rows])
    for r, eo, ea in zip(rows, emb_o, emb_a):
        orig = srcs[(r["src"], r["idx"])]["ids"]
        ids = tok(r["text"], add_special_tokens=False).input_ids
        d = C.detect(ids, prompts_[r["idx"]][-1], V, 0.5)
        r.update(opt_ids=ids, T=len(ids), ranks=d["ranks"], z=d["z"], z_norep=d["z_norep"],
                 edit=S.edit_distance(orig, ids) / len(orig), cos=float(eo @ ea))
    path = C.raw_path("attacks", "scored")
    if os.path.exists(path):
        os.remove(path)
    C.append_jsonl(path, rows)
    log(f"scored {len(rows)} texts")


def dilution(args):
    """Ext. 2b: 600-token human documents with one watermarked span of 60 / 150 / 300 tokens."""
    V = C.vocab_size(opt_tokenizer())
    rng = np.random.default_rng(0)
    pool = [r["ids"] for r in C.read_jsonl(C.raw_path("human", "c4")) if r["idx"] >= P.N_GEN_PROMPTS]
    wm = [r["ids"] for r in sorted(C.read_jsonl(C.gen_path("opt-news-m-d2-g0.5")), key=lambda r: r["idx"])
          if P.T_KEEP[0] <= r["T"] <= P.T_KEEP[1]]
    L, gamma = P.DILUTION_LEN, 0.5

    def human_tokens(n):
        js = rng.choice(len(pool), 3, replace=False)
        return sum((pool[j] for j in js), [])[:n]

    def score_doc(ids):
        g = S.green(C.token_ranks(ids, 2, V), gamma, V)      # first token's context: </s>
        return dict(z=float(S.z_of(g, gamma)), winmax=float(S.winmax(g, gamma)))

    out = [dict(kind="human", share=0.0, **score_doc(human_tokens(L))) for _ in range(P.N_DILUTION_NEG)]
    for span in P.DILUTION_SPANS:
        for _ in range(P.N_DILUTION):
            a, b = rng.choice(len(wm), 2, replace=False)
            w = (wm[a] + wm[b])[:span]
            h = human_tokens(L - span)
            pos = int(rng.integers(0, L - span + 1))
            out.append(dict(kind="mixed", share=span / L, pos=pos, **score_doc(h[:pos] + w + h[pos:])))
    path = C.raw_path("dilution", "docs")
    if os.path.exists(path):
        os.remove(path)
    C.append_jsonl(path, out)
    log(f"dilution: {len(out)} documents")


# ---- oracle perplexity -------------------------------------------------------------------------
def ppl_jobs(oracle):
    """(file, rows, items) for every file the oracle scores; items are (context ids, target ids)."""
    jobs = []
    if oracle == "oracle":
        prompts_ = {r["idx"]: r["prompt_ids"] for r in C.read_jsonl(C.PROMPTS_FILE)}
        files = sorted(f for f in glob.glob(C.gen_path("opt-news-*")) if ".ppl." not in f)
        files += [C.raw_path("human", "c4"), C.raw_path("attacks", "scored")]
        for f in files:
            rows = [r for r in C.read_jsonl(f) if r["idx"] < P.N_GEN_PROMPTS]
            if "-news-m-" in f:     # rows the analysis uses: kept lengths, plus all lengths for Ext 1b
                rows = [r for r in rows if P.T_KEEP[0] <= r["T"] <= P.T_KEEP[1] or r["idx"] < P.N_EXT]
            ids = [r["opt_ids"] if "opt_ids" in r else r["ids"] for r in rows]
            jobs.append((f, rows, [(prompts_[r["idx"]], x) for r, x in zip(rows, ids)]))
    else:   # base Qwen judges the instruct model's news generations (same tokenizer)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(P.MODELS["qwen"])
        for f in sorted(f for f in glob.glob(C.gen_path("qwen-it-news-*")) if ".ppl." not in f):
            rows = C.read_jsonl(f)
            pr = {p["idx"]: p["ids"] for p in C.load_prompts("news", "qwen-it", tok, max(r["idx"] for r in rows) + 1)}
            jobs.append((f, rows, [(pr[r["idx"]], r["ids"]) for r in rows]))
    return jobs


def run_ppl(oracle):
    """Mean NLL of generated tokens given the prompt, written next to each file as <name>.ppl.jsonl.gz."""
    jobs = ppl_jobs(oracle)
    model, tok = C.load_lm(oracle)
    for f, rows, items in jobs:
        side = f.replace(".jsonl", ".ppl.jsonl")
        done = {r["key"] for r in C.read_jsonl(side)}
        todo = [(r.get("key", r["idx"]), it) for r, it in zip(rows, items)
                if r.get("key", r["idx"]) not in done and len(it[1]) > 0]
        if not todo:
            continue
        log(f"ppl {os.path.basename(f)}: {len(todo)} rows")
        for s in range(0, len(todo), 64):
            chunk = todo[s:s + 64]
            nll = C.nll_all(model, tok, [it for _, it in chunk])
            C.append_jsonl(side, [dict(key=k, nll=v, ppl=float(np.exp(v))) for (k, _), v in zip(chunk, nll)])
    del model
    C.free()


def ppl(args):
    """Oracle perplexity (OPT-2.7B) of OPT news generations, human baselines and attacked texts."""
    run_ppl("oracle")


def ppl_qwen(args):
    """Perplexity of the Qwen-Instruct news generations under base Qwen2.5-1.5B (Ext. 1c)."""
    run_ppl("qwen")


# ---- pipeline order ---------------------------------------------------------------------------------
STEPS = dict(prompts=prompts, equiv=equiv, human=human, check=check)
LATER = dict(keys=keys, para=para, t5=t5, t5w=t5w, score=score, dilution=dilution, ppl=ppl, ppl_qwen=ppl_qwen)
ORDER = list(STEPS) + list(P.EXPERIMENTS) + list(LATER)
STEPS.update(LATER)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="a step or experiment name, or 'list'")
    ap.add_argument("--n", type=int, default=None, help="cap the number of prompts per config (quick tests)")
    ap.add_argument("--only", default=None, help="run only the configs whose id contains this substring")
    args = ap.parse_args()
    if args.experiment == "list":
        for name in ORDER:
            desc = P.EXPERIMENTS[name][0] if name in P.EXPERIMENTS else STEPS[name].__doc__
            print(f"{name:12s} {desc.rstrip('.')}")
    elif args.experiment in STEPS:
        STEPS[args.experiment](args)
    elif args.experiment in P.EXPERIMENTS:
        generate(args.experiment, args)
    else:
        sys.exit(f"unknown step or experiment: {args.experiment} (see `python run.py list`)")
