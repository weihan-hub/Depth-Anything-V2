"""Quick viability probe: load frozen relative vitl, fit a global affine
disparity->metric mapping on a batch, report val metrics warm-start vs
per-image-oracle. Confirms checkpoint loads and the mapping premise holds."""
import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from calder.config import paths
from calder.lib.dataset import CalderDepthDataset
from calder.app.finetune.relative_to_metric import RelativeToMetricModel
from util.metric import eval_depth

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max-depth", type=float, default=20.0)
args = ap.parse_args()
dev = "cuda"
remap = ("/mnt/data/data/calder/fine_tune_dataset", "/Data/wwh/fine_tune_dataset")

ds = CalderDepthDataset(args.manifest, mode='val', size=(518, 518), path_remap=remap)
ds.entries = ds.entries[:args.n]
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)

model = RelativeToMetricModel('vitl', paths.RELATIVE_VITL, mode='affine',
                              max_depth=args.max_depth).to(dev).eval()

# collect rel + depth for a global fit
rels, deps = [], []
per_img_oracle = {k: 0.0 for k in ['abs_rel', 'rmse', 'd1']}
n_ok = 0
with torch.no_grad():
    for s in dl:
        img = s['image'].to(dev).float()
        depth = s['depth'].to(dev)[0]
        valid = s['valid_mask'].to(dev)[0]
        rel = model.relative_depth(img)
        rel = F.interpolate(rel[:, None], depth.shape[-2:], mode='nearest')[0, 0]
        m = (valid == 1) & (depth > 0.01) & (depth <= args.max_depth)
        if m.sum() < 100:
            continue
        r, d = rel[m], depth[m]
        rels.append(r.flatten()); deps.append(d.flatten())
        # per-image oracle affine (best possible for a global monotone map)
        t = 1.0 / d
        s_ = ((r - r.mean()) * (t - t.mean())).mean() / ((r - r.mean()) ** 2).mean().clamp(min=1e-8)
        b_ = t.mean() - s_ * r.mean()
        pd = (1.0 / (s_ * r + b_).clamp(min=1.0 / args.max_depth)).clamp(1e-3, args.max_depth)
        cur = eval_depth(pd, d)
        for k in per_img_oracle: per_img_oracle[k] += cur[k]
        n_ok += 1

rel_all = torch.cat(rels); dep_all = torch.cat(deps)
print(f"rel range [{rel_all.min():.3f}, {rel_all.max():.3f}] mean {rel_all.mean():.3f}")
print(f"depth range [{dep_all.min():.3f}, {dep_all.max():.3f}] mean {dep_all.mean():.3f}")

# global affine fit in disparity space
t = 1.0 / dep_all
s_ = ((rel_all - rel_all.mean()) * (t - t.mean())).mean() / ((rel_all - rel_all.mean()) ** 2).mean().clamp(min=1e-8)
b_ = t.mean() - s_ * rel_all.mean()
print(f"GLOBAL affine fit: scale={s_.item():.5f} shift={b_.item():.5f}")

# eval global fit per image
glob = {k: 0.0 for k in ['abs_rel', 'rmse', 'd1']}
n2 = 0
with torch.no_grad():
    for s in dl:
        img = s['image'].to(dev).float()
        depth = s['depth'].to(dev)[0]; valid = s['valid_mask'].to(dev)[0]
        rel = model.relative_depth(img)
        rel = F.interpolate(rel[:, None], depth.shape[-2:], mode='nearest')[0, 0]
        m = (valid == 1) & (depth > 0.01) & (depth <= args.max_depth)
        if m.sum() < 100: continue
        pd = (1.0 / (s_ * rel[m] + b_).clamp(min=1.0 / args.max_depth)).clamp(1e-3, args.max_depth)
        cur = eval_depth(pd, depth[m])
        for k in glob: glob[k] += cur[k]
        n2 += 1

print(f"\n[per-image ORACLE affine]  abs_rel={per_img_oracle['abs_rel']/n_ok:.4f} "
      f"rmse={per_img_oracle['rmse']/n_ok:.4f} d1={per_img_oracle['d1']/n_ok:.4f}")
print(f"[GLOBAL learned affine  ]  abs_rel={glob['abs_rel']/n2:.4f} "
      f"rmse={glob['rmse']/n2:.4f} d1={glob['d1']/n2:.4f}")
print("\nIf GLOBAL is close to ORACLE -> one global mapping suffices. "
      "If GLOBAL >> ORACLE -> relative backbone is not scale-consistent across "
      "images and a per-image/conditioned mapping is needed.")
