# workflow-embeddings

Recurrent session-embedding model for computer-use workflows, trained on AgentNet (OpenCUA). A gated
recurrent state machine reads one (screenshot, input events) step at a time. At each step it outputs a direction
in a frozen text-embedding space (the intent it currently believes) and a confidence κ. Training targets are
per-step InfoNCE against every training instruction's frozen text key.

- **Results, design notes and departures from the v1 pseudocode:** [docs/RESULTS.md](docs/RESULTS.md)
- **Code:** `wfemb/` (data prep, frozen caches, model, loss, training, evaluation, baselines)
- **Cluster jobs:** `scripts/*.sbatch` (CHAI `rnn`, Slurm); sweeps: `sweeps/*.txt`

```bash
uv sync                                   # torch 2.5.1 (cu121), transformers 4.53
scripts/submit_sweep.sh sweeps/r8.txt     # on rnn: one single-GPU training job per line
python -m wfemb.summarize r8_             # compare runs
```
