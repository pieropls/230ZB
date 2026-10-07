"""Data, models, watermark processor, generation, detection, perplexity and attacks."""
import gc
import glob
import gzip
import json
import os
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor,
                          LogitsProcessorList)

import params as P
from stats import spike_modulus, z_of

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "raw")
SUMMARY = os.path.join(ROOT, "results", "summary")
PROMPTS_FILE = os.path.join(ROOT, "results", "prompts_c4.jsonl.gz")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32


# ---- io ----------------------------------------------------------------------
def _open(path, mode):
    return gzip.open(path, mode + "t") if path.endswith(".gz") else open(path, mode)


def read_jsonl(path):
    """All rows of a .jsonl or .jsonl.gz file ([] if missing)."""
    if not os.path.exists(path):
        return []
    with _open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    """Append rows to a .jsonl or .jsonl.gz file (gzip members concatenate)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def raw_path(exp, name):
    """results/raw/<exp>/<name>.jsonl.gz"""
    return os.path.join(RAW, exp, f"{name}.jsonl.gz")


def gen_path(cid):
    """File of one generation config."""
    return raw_path("gen", cid)


# ---- models -------------------------------------------------------------------
def load_lm(key, dtype=None):
    """Causal LM and its left-padding tokenizer, on DEVICE."""
    name = P.MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype or DTYPE).to(DEVICE).eval()
    return model, tok


def free():
    """Release device memory; call after `del model` in the caller."""
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def eos_ids(model, tok):
    """All end-of-sequence ids of a model (Qwen-Instruct has two)."""
    e = model.generation_config.eos_token_id
    e = [e] if isinstance(e, int) else list(e or [])
    return sorted(set(e) | {tok.eos_token_id})


def vocab_size(tok):
    """V as in the paper's code: len(tokenizer.get_vocab())."""
    return len(tok.get_vocab())


# ---- data -------------------------------------------------------------------------
def build_c4_prompts(tok, n=P.N_PROMPTS):
    """Shard 0, shuffled with seed 0; last 200 tokens = human baseline, the rest (>= 50 tokens) = prompt."""
    with gzip.open(os.path.join(ROOT, P.C4_FILE), "rt") as f:
        docs = [json.loads(line)["text"] for line in f]
    order = np.random.default_rng(0).permutation(len(docs))
    rows = []
    for j in order:
        ids = tok(docs[j]).input_ids                  # OPT prepends </s> (BOS), as the paper's code does
        bos, body = ids[:1], ids[1:]
        if len(body) < P.BASELINE_LEN + P.MIN_PROMPT:
            continue
        prompt, base = body[:-P.BASELINE_LEN], body[-P.BASELINE_LEN:]
        prompt = bos + prompt[-(P.MAX_PROMPT - 1):]
        rows.append(dict(idx=len(rows), doc=int(j), prompt_ids=prompt, baseline_ids=base,
                         prompt_text=tok.decode(prompt, skip_special_tokens=True),
                         baseline_text=tok.decode(base, skip_special_tokens=True)))
        if len(rows) == n:
            break
    return rows


def last_words(text, n):
    """The last n space-separated words of a text."""
    return " ".join(text.split(" ")[-n:]).strip()


def chat_ids(tok, user):
    """Token ids of a one-turn chat prompt (generation prompt appended)."""
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   tokenize=True, return_dict=True)["input_ids"]


def load_prompts(domain, model_key, tok, n):
    """List of dicts with idx and prompt ids (plus task id or answers), for any (domain, model)."""
    it = model_key in P.INSTRUCT
    if domain == "news":
        rows = read_jsonl(PROMPTS_FILE)[:n]
        if model_key == "opt":
            return [dict(idx=r["idx"], ids=r["prompt_ids"]) for r in rows]
        texts = [last_words(r["prompt_text"], P.QWEN_NEWS_WORDS) for r in rows]
        users = [P.NEWS_IT_PROMPT.format(text=t) for t in texts] if it else texts
        extra = [{} for _ in rows]
    elif domain == "code":
        df = pd.read_parquet(glob.glob(os.path.join(ROOT, P.HUMANEVAL_GLOB), recursive=True)[0]).head(n)
        texts = df["prompt"].tolist()
        users = [P.CODE_IT_PROMPT.format(text=t) for t in texts] if it else texts
        extra = [dict(task=t) for t in df["task_id"]]
    elif domain == "qa":
        df = pd.read_parquet(glob.glob(os.path.join(ROOT, P.TRIVIAQA_GLOB), recursive=True)[0]).head(n)
        users = [P.QA_PROMPT.format(q=q) for q in df["question"]]
        extra = [dict(aliases=list(a["normalized_aliases"])) for a in df["answer"]]
    else:
        raise ValueError(domain)
    ids = [chat_ids(tok, u) if it else tok(u).input_ids for u in users]
    return [dict(idx=i, ids=x, **e) for i, (x, e) in enumerate(zip(ids, extra))]


# ---- watermark: KGW green lists, simple_1 seeding ------------------------------------------
@lru_cache(maxsize=2048)
def rank_table(prev, V):
    """rank[t] = position of token t in randperm(V) seeded with HASH_KEY * prev (CPU generator).
    Token t is green for gamma iff rank[t] < int(gamma * V): exactly the reference
    `vocab_permutation[:int(V * gamma)]`, for every gamma at once."""
    g = torch.Generator(device="cpu")
    g.manual_seed(P.HASH_KEY * int(prev))
    perm = torch.randperm(V, generator=g).numpy()
    rank = np.empty(V, dtype=np.int32)
    rank[perm] = np.arange(V, dtype=np.int32)
    return rank


def greenlist_ids(prev, V, gamma):
    """Sorted green token ids after token prev."""
    return np.flatnonzero(rank_table(prev, V) < int(gamma * V))


@lru_cache(maxsize=4096)
def green_mask(prev, V, k, width):
    """Boolean mask over the logits width; columns >= V (padding) are never green."""
    m = torch.zeros(width, dtype=torch.bool)
    m[:V] = torch.from_numpy(rank_table(prev, V) < k)
    return m


def token_ranks(ids, prev0, V):
    """Green rank of every token given its predecessor; ids >= V get rank V (never green)."""
    prevs = [prev0] + list(ids[:-1])
    return [int(rank_table(p, V)[t]) if t < V else V for p, t in zip(prevs, ids)]


class KGW(LogitsProcessor):
    """Adds delta to green logits. Runs before transformers' temperature warper, as in the paper's code.
    Optionally logs, per step and row, spike entropy on the raw distribution (S_paper), on the sampled
    distribution softmax(l / tau) with alpha_eff = e^(delta / tau) (S_sampled), and Shannon entropy."""

    def __init__(self, V, gamma, delta, tau=1.0, log=True):
        self.V, self.gamma, self.delta, self.tau, self.log = V, gamma, delta, tau, log
        self.k = int(gamma * V)
        self.zs = spike_modulus(gamma, delta)
        self.zs_eff = spike_modulus(gamma, delta / tau)
        self.steps = []

    def __call__(self, input_ids, scores):
        if self.log:
            x = scores.float()
            p, pt = x.softmax(-1), (x / self.tau).softmax(-1)
            s_paper = (p / (1 + self.zs * p)).sum(-1)
            s_samp = (pt / (1 + self.zs_eff * pt)).sum(-1)
            h = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
            self.steps.append(torch.stack([s_paper, s_samp, h], 1))
        if self.delta:
            prev = input_ids[:, -1].tolist()
            mask = torch.stack([green_mask(t, self.V, self.k, scores.shape[-1]) for t in prev])
            scores = scores + self.delta * mask.to(scores.device, scores.dtype)
        return scores


# ---- detection -------------------------------------------------------------------------
def detect(ids, prev0, V, gamma):
    """Ranks, green flags, z with repeats counted (paper) and z ignoring repeated (context, token) pairs."""
    ranks = token_ranks(ids, prev0, V)
    k = int(gamma * V)
    green = [int(r < k) for r in ranks]
    seen, green_u = set(), []
    for pair, g in zip(zip([prev0] + list(ids[:-1]), ids), green):
        if pair not in seen:
            seen.add(pair)
            green_u.append(g)
    return dict(ranks=ranks, green=green, z=float(z_of(green, gamma)), z_norep=float(z_of(green_u, gamma)))


# ---- generation ----------------------------------------------------------------------------
def make_batches(prompts, beams, max_new):
    """Longest prompts first; pack while batch * (prompt + new) * beams fits the token budget."""
    prompts = sorted(prompts, key=lambda p: -len(p["ids"]))
    budget = P.TOKEN_BUDGET if beams == 1 else P.BEAM_TOKEN_BUDGET
    batches, cur = [], []
    for p in prompts:
        L = len(cur[0]["ids"]) if cur else len(p["ids"])
        if cur and ((len(cur) + 1) * (L + max_new) * beams > budget or len(cur) >= P.MAX_BATCH):
            batches.append(cur)
            cur = []
        cur.append(p)
    return batches + ([cur] if cur else [])


def left_pad(seqs, pad):
    """Left-padded id tensor and attention mask."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        att[i, L - len(s):] = 1
    return ids, att


def suppress_eos(cfg):
    """Beam search and greedy decoding on news run the full 200 tokens, as in the paper."""
    return cfg["decode"] == "beam" or (cfg["decode"] == "greedy" and cfg["domain"] == "news")


def generate_batch(model, tok, batch, cfg, V, eos):
    """Watermarked generation for one batch of prompts; one result row per prompt."""
    sample = cfg["decode"] == "sample"
    ids, att = left_pad([p["ids"] for p in batch], tok.pad_token_id)
    proc = KGW(V, cfg["gamma"], cfg["delta"], tau=P.TEMP if sample else 1.0, log=cfg["beams"] == 1)
    gcfg = GenerationConfig(max_new_tokens=cfg["max_new"], do_sample=sample, num_beams=cfg["beams"],
                            eos_token_id=eos, pad_token_id=tok.pad_token_id,
                            suppress_tokens=eos if suppress_eos(cfg) else None,
                            **(dict(temperature=P.TEMP, top_k=0, top_p=1.0) if sample else {}))
    torch.manual_seed(cfg["seed"] + batch[0]["idx"])
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), generation_config=gcfg,
                             logits_processor=LogitsProcessorList([proc]))
    gen = out[:, ids.shape[1]:].cpu().tolist()
    ent = torch.stack(proc.steps, 1).cpu().numpy() if proc.steps else None   # (B, steps, 3)
    rows = []
    for i, p in enumerate(batch):
        g = gen[i]
        T = next((t for t, x in enumerate(g) if x in eos), len(g))
        g = g[:T]
        prev0 = p["ids"][-1]
        d = detect(g, prev0, V, cfg["gamma"])
        text = tok.decode(g, skip_special_tokens=True)
        re_ids = tok(text, add_special_tokens=False).input_ids
        row = dict(idx=p["idx"], cfg=cfg["id"], seed=cfg["seed"], model=cfg["model"], domain=cfg["domain"],
                   decode=cfg["decode"], beams=cfg["beams"], gamma=cfg["gamma"], delta=cfg["delta"],
                   prompt_len=len(p["ids"]), T=T, ids=g, ranks=d["ranks"], green=d["green"],
                   z=d["z"], z_norep=d["z_norep"],
                   z_reenc=detect(re_ids, prev0, V, cfg["gamma"])["z"], T_reenc=len(re_ids), text=text)
        if ent is not None:
            e = np.round(ent[i, :T], 4)
            row.update(S_paper=e[:, 0].tolist(), S_sampled=e[:, 1].tolist(), H=e[:, 2].tolist())
        for key in ("task", "aliases"):
            if key in p:
                row[key] = p[key]
        rows.append(row)
    return rows


def run_config(cfg, model, tok, log=print):
    """Generate the missing rows of one config, appending after every batch (resumable). With cfg['keep'],
    n counts rows with T in T_KEEP and further prompts are drawn (in idx order) until n rows are kept."""
    path = gen_path(cfg["id"])
    V, eos = vocab_size(tok), eos_ids(model, tok)
    pool = load_prompts(cfg["domain"], cfg["model"], tok, P.N_GEN_PROMPTS if cfg["keep"] else cfg["n"])
    while True:
        rows = read_jsonl(path)
        done = {r["idx"] for r in rows}
        if cfg["keep"]:
            have = sum(P.T_KEEP[0] <= r["T"] <= P.T_KEEP[1] for r in rows)
            rest = [p for p in pool if p["idx"] not in done]
            if have >= cfg["n"] or not rest:
                log(f"{cfg['id']}: {have} kept of {len(rows)} generated" + ("" if have >= cfg["n"] else
                    f"  (prompts exhausted; target was {cfg['n']})"))
                return
            rate = max(have / len(rows), 0.3) if rows else 1.0
            todo = rest[:int(np.ceil((cfg["n"] - have) / rate))] if rows else rest[:cfg["n"]]
        else:
            todo = [p for p in pool[:cfg["n"]] if p["idx"] not in done]
            if not todo:
                log(f"{cfg['id']}: complete ({len(done)} rows)")
                return
        batches = make_batches(todo, cfg["beams"], cfg["max_new"])
        log(f"{cfg['id']}: {len(done)} done, {len(todo)} to go in {len(batches)} batches")
        t0, n_gen = time.time(), 0
        for b in batches:
            append_jsonl(path, generate_batch(model, tok, b, cfg, V, eos))
            if DEVICE == "mps":
                torch.mps.empty_cache()     # release cached blocks between batches (beam search on 24 GB)
            n_gen += len(b)
            log(f"  {cfg['id']}: {len(done) + n_gen}  {(time.time() - t0) / n_gen:.2f} s/seq")
        if not cfg["keep"]:
            return


# ---- perplexity -------------------------------------------------------------------------
def nll_batch(model, items, pad):
    """Mean NLL of each target given its context; contexts are cut from the left to fit the model."""
    L = model.config.max_position_embeddings
    items = [(c[-(L - len(t)):], t) for c, t in items]
    ids, att = left_pad([c + t for c, t in items], pad)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    nt = max(len(t) for _, t in items)
    with torch.no_grad():
        logits = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), position_ids=pos.to(DEVICE),
                       logits_to_keep=nt + 1).logits[:, :-1].float()
    logp = torch.log_softmax(logits, -1).cpu()
    out = []
    for i, (_, t) in enumerate(items):
        lp = logp[i, nt - len(t):].gather(-1, torch.tensor(t)[:, None])[:, 0]
        out.append(float(-lp.mean()))
    return out


def nll_all(model, tok, items):
    """Mean NLL per (context, target) item, batched by total length; results in input order."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]) - len(items[i][1]))
    res, cur = [None] * len(items), []

    def flush():
        for j, v in zip(cur, nll_batch(model, [items[j] for j in cur], tok.pad_token_id)):
            res[j] = v
        if DEVICE == "mps":
            torch.mps.empty_cache()                     # variable shapes otherwise fill the MPS cache

    for i in order:
        L = len(items[cur[0]][0]) + len(items[cur[0]][1]) if cur else 0
        if cur and (len(cur) + 1) * L > P.PPL_BATCH_TOKENS:
            flush()
            cur = []
        cur.append(i)
    if cur:
        flush()
    return res


# ---- QA scoring ----------------------------------------------------------------------------
def normalize_answer(s):
    """TriviaQA's official normalization (punctuation -> space, drop articles)."""
    s = s.replace("_", " ").lower()
    s = "".join(" " if (not ch.isalnum() and not ch.isspace()) else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_scores(text, aliases):
    """Exact match and token F1 of the first answer line against normalized aliases."""
    pred = normalize_answer(text.strip().split("\n")[0])
    em = float(pred in aliases)

    def f1(a):
        p, g = pred.split(), a.split()
        common = sum(min(p.count(w), g.count(w)) for w in set(p))
        if common == 0:
            return 0.0
        pr, rc = common / len(p), common / len(g)
        return 2 * pr * rc / (pr + rc)
    return em, max((f1(a) for a in aliases), default=0.0)


# ---- attacks: paraphrase (Extension 2a) ---------------------------------------------------------
def sample_texts(model, tok, prompt_ids, max_new, seed, temperature=P.TEMP, batch=16):
    """Unwatermarked sampling (temperature only), longest prompts first; texts in input order."""
    eos = eos_ids(model, tok)
    gcfg = GenerationConfig(max_new_tokens=max_new, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                            eos_token_id=eos, pad_token_id=tok.pad_token_id)
    order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))
    out = [None] * len(prompt_ids)
    for s in range(0, len(order), batch):
        js = order[s:s + batch]
        ids, att = left_pad([prompt_ids[j] for j in js], tok.pad_token_id)
        torch.manual_seed(seed + js[0])
        with torch.no_grad():
            gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), generation_config=gcfg)
        for j, g in zip(js, gen[:, ids.shape[1]:].cpu().tolist()):
            T = next((t for t, x in enumerate(g) if x in eos), len(g))
            out[j] = tok.decode(g[:T], skip_special_tokens=True).strip()
    return out


def paraphrase(model, tok, texts, strength, seed):
    """Rewrite texts with the instruct model at one of the PARA_PROMPTS strengths."""
    users = [P.PARA_PROMPTS[strength] + P.PARA_SUFFIX.format(text=t) for t in texts]
    return sample_texts(model, tok, [chat_ids(tok, u) for u in users], P.PARA_MAX_NEW, seed)


# ---- attacks: T5 span replacement (Section 7.1) ----------------------------------------------------
def load_t5():
    """T5-Large in bfloat16 (2x faster than fp32 on mps, same replacements in our check)."""
    from transformers import AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(P.MODELS["t5"])
    model = AutoModelForSeq2SeqLM.from_pretrained(P.MODELS["t5"], dtype=torch.bfloat16).to(DEVICE).eval()
    return model, tok


def t5_state(text, seed):
    """Text split into alternating non-word / word pieces (odd indices are words)."""
    return dict(pieces=re.split(r"(\w+)", text), replaced=[], succ=0, att=0,
                rng=np.random.default_rng(seed).bit_generator.state)


def t5_round(model, tok, states, single_word=False):
    """One mask per sequence: replace a random not-yet-replaced word by T5-Large's first candidate
    (50-way beam search, 20 returned) that differs from it; single_word: the candidate must be one word."""
    s0, s1 = tok.convert_tokens_to_ids("<extra_id_0>"), tok.convert_tokens_to_ids("<extra_id_1>")
    jobs = []
    for s in states:
        done = set(s["replaced"])
        words = [i for i in range(1, len(s["pieces"]), 2) if i not in done]
        if not words:
            continue
        rng = np.random.default_rng()
        rng.bit_generator.state = s["rng"]
        i = int(rng.choice(words))
        s["rng"] = rng.bit_generator.state
        ids = tok("".join(s["pieces"][:i]) + "<extra_id_0>" + "".join(s["pieces"][i + 1:])).input_ids[:-1]
        p = ids.index(s0)
        jobs.append((s, i, ids[max(0, p - P.T5_WINDOW): p + P.T5_WINDOW + 1] + [tok.eos_token_id]))
    gcfg = GenerationConfig(num_beams=P.T5_BEAMS, num_return_sequences=P.T5_K, max_new_tokens=P.T5_MAX_NEW,
                            do_sample=False, early_stopping=True,
                            decoder_start_token_id=model.config.decoder_start_token_id,
                            eos_token_id=[s1, tok.eos_token_id],    # a span ends at <extra_id_1>
                            pad_token_id=tok.pad_token_id)
    for b in range(0, len(jobs), P.T5_BATCH):
        chunk = jobs[b:b + P.T5_BATCH]
        L = 2 * P.T5_WINDOW + 2                       # fixed shape: no mps recompiles between calls
        ids = torch.tensor([w + [tok.pad_token_id] * (L - len(w)) for _, _, w in chunk])
        att = (torch.arange(L)[None] < torch.tensor([len(w) for _, _, w in chunk])[:, None]).long()
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), generation_config=gcfg)
        out = out.cpu().tolist()
        for j, (s, i, _) in enumerate(chunk):
            s["att"] += 1
            word = s["pieces"][i]
            for o in out[j * P.T5_K:(j + 1) * P.T5_K]:
                if s0 not in o or s1 not in o[o.index(s0):]:       # need a complete span
                    continue
                o = o[o.index(s0) + 1:]
                o = o[:o.index(s1)]
                cand = tok.decode(o, skip_special_tokens=True).strip()
                if cand and cand != word and (not single_word or re.fullmatch(r"\w+", cand)):
                    s["pieces"][i] = cand
                    s["replaced"].append(i)
                    s["succ"] += 1
                    break


def t5_attack(model, tok, states, eps_list, T=P.MAX_NEW, log=print, single_word=False):
    """Advance every state round by round. A level eps is recorded the first round a sequence has
    eps*T successes or 3*eps*T attempts (or no word left). Returns {eps: [snapshot per state]}."""
    snaps = {e: [None] * len(states) for e in eps_list}
    rounds = 0
    while True:
        for e in eps_list:
            for j, s in enumerate(states):
                left = len(s["pieces"]) // 2 - len(s["replaced"])
                if snaps[e][j] is None and (s["succ"] >= round(e * T) or s["att"] >= round(3 * e * T) or left == 0):
                    snaps[e][j] = json.loads(json.dumps(s))
        active = [s for j, s in enumerate(states) if snaps[eps_list[-1]][j] is None]
        if not active:
            return snaps
        t5_round(model, tok, active, single_word)
        rounds += 1
        if rounds % 20 == 0:
            log(f"    t5 round {rounds}: {len(active)} active, mean successes "
                f"{np.mean([s['succ'] for s in states]):.1f}")


# ---- meaning kept (MiniLM sentence embeddings) ---------------------------------------------------
def embed(texts, batch=64):
    """Mean-pooled, L2-normalized all-MiniLM-L6-v2 embeddings."""
    from transformers import AutoModel
    tok = AutoTokenizer.from_pretrained(P.MODELS["minilm"])
    model = AutoModel.from_pretrained(P.MODELS["minilm"]).to(DEVICE).eval()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"][..., None].float()
        e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    del model
    free()
    return torch.cat(out).numpy()


# ---- loading results --------------------------------------------------------------------------
def load_rows(path, kept=False):
    """Rows of a raw file sorted by idx, with oracle ppl merged in; kept=True applies the T = 200 +- 5 filter."""
    rows = read_jsonl(path)
    side = {r["key"]: r["ppl"] for r in read_jsonl(path.replace(".jsonl", ".ppl.jsonl"))}
    for r in rows:
        r["ppl"] = side.get(r.get("key", r["idx"]))
    if kept:
        rows = [r for r in rows if P.T_KEEP[0] <= r["T"] <= P.T_KEEP[1]]
    return sorted(rows, key=lambda r: r["idx"])


def load_gen(cid, kept=False):
    """Rows of one generation config."""
    return load_rows(gen_path(cid), kept)
