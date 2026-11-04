# Session-embedding training algorithm (v1): the original spec

This is the pseudocode the implementation in `wfemb/` follows, as provided by the user. Where the implementation
departs from it, `docs/RESULTS.md` says so and explains why.

**What it trains:** a recurrent state machine. It reads one (screenshot, input events) step at a time and outputs:
- a **direction** `d_t`: the intent it currently believes;
- a **confidence** `κ_t`: how sure it is.

**Training target:** at every step, the frozen text embedding of the session's human instruction.

**Inference:** screenshots and events only.

Notation: `B` = sessions per batch, `T` = steps per session (≈19 on average), `D` = embedding dimension, `N` = number of training instructions (≈22K).

---

## 1. Dataset creation (offline, once)

```python
# ---------- 1a. Load and filter AgentNet (HF: xlangai/AgentNet) ----------
sessions = []
for traj in load_agentnet_jsonl():
    if traj.meta.task_description_alignment in {"Low", "Not aligned"}:   # ~449 off-task sessions
        continue
    if len(traj.steps) < 2:
        continue
    # Keep failed attempts (task_completed == False): with intent as the target, they are valid.
    steps = []
    for s in traj.steps:
        events = parse_pyautogui(s.value.code)                   # a step can hold several calls
        events = [e for e in events if e.type != "terminate"]    # no artificial end-of-session cue
        steps.append(Step(image=s.image, events=events))
    steps.append(Step(image=traj.final_screenshot, events=[]))   # keep the real final screen
    sessions.append(Session(id=traj.task_id, instruction=traj.instruction, steps=steps,
                            os=traj.meta.system, app=traj.meta.app, domain=traj.meta.domain))

def parse_pyautogui(code) -> list[Event]:
    # Event = (type, x, y, x2, y2, scroll, keys, text); coords already normalized to [0,1]
    #   click/double/triple/right/middle -> type, x, y
    #   moveTo/dragTo                    -> type, x, y (start), x2, y2 (end)
    #   scroll/hscroll                   -> type, scroll amount
    #   press/hotkey                     -> type, keys (list of key tokens)
    #   write/typewrite                  -> type, text (typed string)
    ...

# ---------- 1b. Grouped splits (nothing leaks across splits) ----------
group_id = {}      # sessions whose (normalized) instructions are identical share a group
for s in sessions:
    group_id[s.id] = hash(normalize_ws(s.instruction.lower()))
test_apps   = sample_apps(fraction=0.10)                  # whole apps held out
test        = [s for s in sessions if s.app in test_apps]
rest        = [s for s in sessions if s.app not in test_apps]
val, train  = group_split(rest, by=group_id, frac=0.05)   # no group is split across sets
# WONDERBREAD (several demos per workflow) serves as an extra external test set.

# ---------- 1c. Teacher keys: frozen text embeddings of instructions ----------
TextEmbed = load_frozen_text_embedder()                   # off-the-shelf sentence embedder; never trained
for s in sessions:
    s.key = l2_normalize(TextEmbed(s.instruction))        # D-vector
U_train = stack([s.key for s in train])                   # N x D candidate set for the loss

# One-time check that the keys aren't dominated by literals: for ~200 random instructions, look at
# their top-10 neighbours in U_train. If shared filenames or values drive the neighbours rather than
# intent, re-embed with incidental literals replaced by placeholders ("<FILE>", "<VALUE>").
# Keep app and site names.

# ---------- 1c'. Paraphrase keys (offline; used for augmentation in §2) ----------
for s in sessions:
    cands = llm_paraphrase(s.instruction, n=P_PARA)       # same intent, different wording; keep app/site names
    cand_keys = [l2_normalize(TextEmbed(p)) for p in cands]
    # Drop paraphrases that drift in meaning: they must stay close to the original instruction
    s.para_keys = [s.key] + [u for u in cand_keys if u @ s.key > THETA_PARA]   # THETA_PARA ≈ 0.85
# fn_mask (1d) and evaluation always use the original key s.key; paraphrases are used only as
# training positives.

# ---------- 1d. False-negative mask (precomputed) ----------
# For each training session i: candidates that must NOT count as negatives.
for i, s in enumerate(train):
    sims = U_train @ U_train[i]
    s.fn_mask = [j for j in range(N)
                 if j != i and (group_id[train[j].id] == group_id[s.id] or sims[j] > THETA_FN)]
    # THETA_FN: conservative (≈0.8–0.9), set by inspecting neighbour pairs.

# ---------- 1e. Frame feature cache (frozen vision encoder) ----------
VisionEnc = load_frozen_vision_encoder()                  # spatial tokens; keeps GUI text readable
for s in sessions:
    for st in s.steps:
        img = resize_keep_aspect(load_png(st.image), max_side=RES)   # don't squash to a square
        st.frame_tokens = VisionEnc(img).to(bf16)        # n_tok x D_v (pooled to a fixed n_tok)
        for e in st.events:                               # embed typed strings once with the text model
            if e.text: e.text_vec = TextEmbed(e.text)
write_shards(sessions, format="per-trajectory")           # e.g. ~57 GB at 64 tokens x 1024 dims, bf16

# Pre-training check: can a linear probe on matched pairs tell decisive details apart from the
# cached tokens (which menu item was picked, which checkbox state)? If not, raise RES or add a
# high-res crop around the action point, then rebuild the cache.
```

---

## 2. Data augmentation: paraphrased targets

```python
def sample_positive_keys(batch):
    # Each session's positive is a randomly chosen wording of its instruction, re-drawn every time
    # the session is seen. The model then has to fit the intent, not one embedding of one phrasing.
    return stack([choice(s.para_keys) for s in batch.sessions])   # B x D
```

---

## 3. Model

```python
class SessionStateMachine(nn.Module):
    def __init__(self):
        self.event_enc  = EventEncoder()   # type embed + Fourier(x,y,x2,y2) + key-token embed
                                           # + proj(text_vec), summed over a step's events -> D
        self.no_event   = Parameter(D)     # used for the final screen (no action)
        self.frame_pool = CrossAttnPool(queries=Q, dim=D)   # event-conditioned attention over frame tokens
        self.fuse       = MLP(2*D -> D)
        self.cell       = GRUCell(D, H)    # gated recurrent state (the "state machine")
        self.norm       = LayerNorm(H)
        self.h0         = Parameter(H)
        self.dir_head   = Linear(H -> D)
        self.conf_head  = Linear(H -> 1)   # init the bias so κ starts low (≈ 1/0.2)

    def step(self, h, frame_tokens, events):
        a = self.event_enc(events) if events else self.no_event
        f = self.frame_pool(query=a, tokens=frame_tokens)   # the action decides where to look
        z = self.fuse(cat(f, a))
        h = self.norm(self.cell(z, h))
        d = l2_normalize(self.dir_head(h))                  # intent direction
        k = clamp(softplus(self.conf_head(h)) + K_MIN, max=K_MAX)   # confidence (inverse temperature)
        return h, d, k

    def forward(self, batch):                               # batch: padded B x T
        h = self.h0.expand(B, H)
        ds, ks = [], []
        for t in range(T_max):                              # full BPTT; T is small
            h_new, d, k = self.step(h, batch.frames[:, t], batch.events[:, t])
            h = where(batch.valid[:, t], h_new, h)          # freeze state after a session ends
            ds.append(d); ks.append(k)
        return stack(ds, 1), stack(ks, 1)                   # B x T x D, B x T x 1

# Inference: the same step() call per frame; the state h stays O(H) however long the session runs.
# Retrieval embedding = d_T (cosine); κ_T is the confidence score.
```

---

## 4. Loss: per-step InfoNCE against frozen instruction keys

```python
def loss_fn(d, k, batch, U, pos_keys):  # d: B x T x D, k: B x T x 1, U: N x D (frozen keys), pos_keys: B x D
    # Score every step against ALL training instruction keys (N ≈ 22K), not only in-batch ones:
    #  - every same-app instruction is always a negative, so no clustered batching is needed
    #  - the softmax is a distribution over the whole intent vocabulary
    logits = k * (d @ U.T)                                         # B x T x N, computed in fp32
    # Replace each session's own column with its paraphrased positive (augmentation, §2)
    pos = k.squeeze(-1) * einsum("btd,bd->bt", d, pos_keys)        # B x T
    logits[arange(B), :, batch.idx] = pos
    # Mask false negatives (same instruction group, or a near-duplicate intent) -> -inf
    for b, s in enumerate(batch.sessions):
        logits[b, :, s.fn_mask] = -inf
    target = batch.idx                                             # B: each session's own key index in U
    ce = cross_entropy(logits.reshape(B*T, N), target.repeat_interleave(T), reduction="none")
    ce = ce.reshape(B, T) * batch.valid                            # ignore padding
    per_session = ce.sum(1) / batch.valid.sum(1)                   # each session weighs the same
    return per_session.mean()

# Why this yields uncertainty: cross-entropy is a proper scoring rule. On a prefix that sessions
# with different intents share, the minimiser spreads probability over each plausible intent. A low
# κ_t is the only way to express a flat distribution, so κ_t is learnt low early and rises once
# decisive evidence appears. Nothing forces κ to rise monotonically.
```

---

## 5. Training loop

```python
def train_run(seed, cfg):                     # one run per GPU; 8 GPUs = 8 seeds / ablations
    set_seed(seed)
    model = SessionStateMachine().cuda()
    U = U_train.cuda()                        # frozen, ~22K x D
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    sched = cosine_with_warmup(opt, warmup=cfg.warmup, total=cfg.steps)
    loader = LengthBucketedLoader(train_shards, batch_sessions=cfg.B)   # pad among similar lengths

    for step, batch in enumerate(loader):
        pos_keys = sample_positive_keys(batch).cuda()       # paraphrase augmentation
        with autocast(bf16):
            d, k = model(batch)
        loss = loss_fn(d.float(), k.float(), batch, U, pos_keys)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()

        if step % cfg.eval_every == 0:
            metrics = evaluate(model, val)
            log(metrics); keep_best(model, key="val_session_R@5")

# Run a 1-hour pilot first to measure throughput and memory, then set B and the step count.
# Build the frame cache on all 8 GPUs in parallel first; then each GPU runs its own job.
```

---

## 6. Evaluation (held-out val/test; never scored by the teacher)

```python
def evaluate(model, split):
    d, k = embed_all(model, split)                     # d_t, κ_t at every step of every session
    E = d[:, -1]                                       # terminal direction
    return {
      # Session-to-session retrieval, with relevance from independent judgments
      # (duplicate-instruction groups, WONDERBREAD repeats, a trace-judged pool)
      "session_R@k/MRR (within-app, cross-app)": retrieval(E, relevance=judged_pairs(split)),
      # Does the uncertainty behave as intended?
      "entropy_vs_t": [entropy(softmax(k[:, t] * d[:, t] @ U_ref.T)) for t in steps],  # should fall
      "true_key_rank_vs_t": rank_of_true_instruction(d, k, U_ref),                     # should improve
      "calibration_ECE": ece(prob_on_true_key(d, k, U_ref)),
      # Shortcut check: OS/theme should be LESS decodable from E than from raw frame features
      "nuisance_probe": linear_probe(E, labels=split.os) - linear_probe(mean_frame(split), split.os),
    }
# U_ref = keys for a fixed reference set of instructions (val/test instructions + distractors).
# Baselines: action n-gram TF-IDF, untrained mean/last-frame features, the text-key oracle.
```

---

## Settings to choose

`THETA_FN` (false-negative threshold), `P_PARA` / `THETA_PARA` (number of paraphrases, drift filter), `RES` and `n_tok` (frame detail), `D`, `H`, `K_MIN` / `K_MAX`, `B`.

**Upgrade only when a check fails:**
1. **Multimodal ambiguity:** if an ambiguous prefix's top prediction is neither true candidate, give the readout K directions with mixture weights.
2. **Decisive details unreadable:** add a high-res crop around the action point, then fine-tune the top vision layers.
