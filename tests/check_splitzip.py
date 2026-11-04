"""On rnn: verify SplitZip against the real AgentNet split archives (member count, PNG decode, coverage)."""
import io, pickle, random, time
from PIL import Image
from wfemb import paths
from wfemb.splitzip import SplitZip

sessions = pickle.load(open(paths.PREP / "sessions.pkl", "rb"))
need = {"ubuntu": set(), "win_mac": set()}
for s in sessions:
    need[s["source"]].update(s["images"])
for src, zp in paths.ZIPS.items():
    t = time.time()
    z = SplitZip(zp)
    idx = z.basename_index()
    missing = need[src] - set(idx)
    methods = {}
    for v in z.members.values():
        methods[v[3]] = methods.get(v[3], 0) + 1
    print(f"{src}: {len(z.parts)} parts, {len(z.members)} members, CD parsed in {time.time() - t:.1f}s, "
          f"methods={methods}, needed={len(need[src])}, missing={len(missing)}")
    names = random.Random(0).sample(sorted(need[src] & set(idx)), 50)
    t = time.time()
    for n in names:
        im = Image.open(io.BytesIO(z.read(idx[n])))
        im.load()
    print(f"  decoded 50 random PNGs in {time.time() - t:.1f}s, last size {im.size} mode {im.mode}")
