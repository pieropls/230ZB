"""Every knob and every experiment grid, as plain dicts and lists."""

# ---- models ----------------------------------------------------------------
MODELS = {
    "opt": "facebook/opt-1.3b",            # generator (paper)
    "oracle": "facebook/opt-2.7b",         # perplexity oracle (paper)
    "t5": "google-t5/t5-large",            # span-replacement attack (paper)
    "qwen": "Qwen/Qwen2.5-1.5B",           # Extension 1c, base model
    "qwen-it": "Qwen/Qwen2.5-1.5B-Instruct",  # Extension 1c instruct model, paraphraser in 2a
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # meaning kept after attacks (2a)
}
INSTRUCT = {"qwen-it"}                     # models that get a chat template

# ---- watermark (KGW, simple_1 seeding) ---------------------------------------
HASH_KEY = 15485863
OTHER_KEYS = [15485867, 32452843, 49979687]   # length side-effect check under other keys
GAMMAS = [0.1, 0.25, 0.5, 0.75, 0.9]       # every row stores green ranks, so any gamma can be rescored
TEMP = 0.7

# ---- data --------------------------------------------------------------------
C4_FILE = "data/c4/realnewslike/c4-train.00000-of-00512.json.gz"
HUMANEVAL_GLOB = "data/humaneval/**/*.parquet"
TRIVIAQA_GLOB = "data/triviaqa/**/*.parquet"
N_PROMPTS = 2000                           # prompts 0..999 are for generation; 1000+ give human text (Ext 2b)
N_GEN_PROMPTS = 1000
BASELINE_LEN = 200
MIN_PROMPT = 50
MAX_PROMPT = 2048 - 200
MAX_NEW = 200
T_KEEP = (195, 205)                        # the paper's T = 200 +- 5 filter (multinomial news runs)
QA_PROMPT = ("The following is a trivia question with a single correct factual answer. "
             "Please provide the answer to the question.\n\nQuestion: {q}\n\nAnswer: ")
NEWS_IT_PROMPT = "Continue this news article:\n\n{text}"
CODE_IT_PROMPT = "Complete this Python function. Reply with code only.\n\n{text}"
QWEN_NEWS_WORDS = 150                      # Qwen news prompts: last ~150 words of the C4 prompt

# ---- sample sizes --------------------------------------------------------------
N_HEAD = 500
N_TABLE = 300
N_GRID = 100
N_BEAM = 150
N_ATTACK = 200                             # multi-word T5 and paraphrase attacks
N_ATTACK_SINGLE = 100                      # single-word T5 attack
N_EXT = 200
N_QA = 500
N_CODE = 164
N_KEYS = 100

# ---- batching ------------------------------------------------------------------
MAX_BATCH = 32
TOKEN_BUDGET = 24000                       # batch * (prompt + new) * beams <= budget
BEAM_TOKEN_BUDGET = 12000                  # beam search copies its cache when reordering: half the budget
PPL_BATCH_TOKENS = 8000

# ---- attacks -------------------------------------------------------------------
T5_EPS = [0.1, 0.3, 0.5, 0.7]              # single-word attack
T5_EPS_MULTI = [0.1, 0.3]                  # multi-word attack
T5_WINDOW = 60
T5_BEAMS, T5_K = 50, 20
T5_MAX_NEW = 10
T5_CHUNK = 32                              # sequences attacked together (one mask per sequence per round)
T5_BATCH = 8                               # windows per T5 generate call (x 50 beams)
PARA_PROMPTS = {
    "light": "Lightly edit this text: change about one word in five, keep everything else.",
    "medium": "Paraphrase this text sentence by sentence.",
    "full": "Rewrite this text completely in your own words, keeping the meaning.",
}
PARA_SUFFIX = "\n\nText:\n{text}\n\nOutput only the rewritten text."
PARA_MAX_NEW = 300
DILUTION_LEN = 600
DILUTION_SPANS = [60, 150, 300]            # 10%, 25%, 50% of 600 tokens
N_DILUTION = 200                           # documents per share
N_DILUTION_NEG = 1000                      # human-only documents for calibration


# ---- experiment grids ----------------------------------------------------------
def C(model="opt", domain="news", decode="sample", beams=1, gamma=0.5, delta=2.0, n=N_HEAD,
      max_new=MAX_NEW, seed=0):
    """One generation config; delta = 0 runs do not depend on gamma (stored ranks rescore any gamma)."""
    if delta == 0:
        gamma = 0.5
    dec = {"sample": "m", "greedy": "greedy", "beam": f"beam{beams}"}[decode]
    cid = f"{model}-{domain}-{dec}-d{delta:g}" + ("" if delta == 0 else f"-g{gamma:g}")
    # OPT news multinomial runs: n counts rows kept after the T filter, as in the paper's script,
    # which drew prompts until enough rows passed (prompts 0..N_GEN_PROMPTS-1 only)
    keep = model == "opt" and domain == "news" and decode == "sample"
    return dict(id=cid, model=model, domain=domain, decode=decode, beams=beams, gamma=gamma,
                delta=float(delta), n=n, max_new=max_new, seed=seed, keep=keep)


def B8(gamma, delta, n=N_BEAM):
    """8-beam search config."""
    return C(decode="beam", beams=8, gamma=gamma, delta=delta, n=n)


# experiment name -> (description, configs); several experiments share configs (and files)
EXPERIMENTS = {
    "headline": ("§4.1: gamma 0.5, delta 2 (multinomial and 4-beam) and the delta = 0 negatives",
                 [C(gamma=0.5, delta=2, n=N_HEAD), C(delta=0, n=N_HEAD),
                  C(decode="beam", beams=4, gamma=0.5, delta=2, n=N_BEAM)]),
    "table8": ("Table 8 / Fig. 4: multinomial delta 1, 2, 5 x gamma 0.25, 0.5; 8-beam delta 2",
               [C(gamma=g, delta=d, n=N_HEAD if (g, d) == (0.5, 2) else N_TABLE)
                for d in (1, 2, 5) for g in (0.25, 0.5)] + [B8(0.25, 2), B8(0.5, 2)]),
    "fig2_grid": ("Fig. 2 left, Fig. 3a/b, Fig. 7: multinomial delta x gamma grid",
                  [C(gamma=g, delta=d, n=N_GRID) for d in (1, 2, 5, 10) for g in GAMMAS]
                  + [C(gamma=g, delta=0.5, n=N_GRID) for g in (0.25, 0.5)]),
    "fig2_beams": ("Fig. 2 right and Table 8 8-beam rows: greedy and 8-beam search, gamma 0.5",
                   [C(decode="greedy", gamma=0.5, delta=d, n=N_GRID) for d in (0, 0.5, 1, 2, 5, 10)]
                   + [B8(0.5, d, N_GRID) for d in (0, 0.5, 1, 5, 10)]),
    "fig3c_beams": ("Fig. 3c and Table 8 8-beam rows: 8-beam search, gamma 0.25",
                    [B8(0.25, d, N_GRID) for d in (0.5, 1, 5, 10)]),
    "ext1b": ("Ext. 1b: news, code (HumanEval), short answers (TriviaQA) with OPT-1.3B",
              [C(gamma=0.25, delta=2, n=N_TABLE), C(delta=0, n=N_HEAD),
               C(domain="code", gamma=0.25, delta=2, n=N_CODE), C(domain="code", delta=0, n=N_CODE),
               C(domain="qa", decode="greedy", gamma=0.5, delta=2, n=N_QA, max_new=16),
               C(domain="qa", decode="greedy", delta=0, n=N_QA, max_new=16)]),
    "ext1c": ("Ext. 1c: Qwen2.5-1.5B base vs Instruct on news and code",
              [C(model=m, domain=d, gamma=0.25, delta=dl, n=N_EXT if d == "news" else N_CODE)
               for m in ("qwen", "qwen-it") for d in ("news", "code") for dl in (2, 0)]),
    "ext1c_qa": ("Ext. 1c: Qwen base vs Instruct on TriviaQA, and the Instruct model at delta 4 on news",
                 [C(model=m, domain="qa", decode="greedy", gamma=0.5, delta=d, n=N_QA, max_new=16)
                  for m in ("qwen", "qwen-it") for d in (2, 0)]
                 + [C(model="qwen-it", domain="news", gamma=0.25, delta=4, n=N_EXT)]),
}
