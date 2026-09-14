#!/usr/bin/env python
import os, sys, json, time, math, argparse, random
import numpy as np, pandas as pd, torch
from BrainGFM_pretrain import build_encoder, Pretrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="./data/pretrain", help="directory with <atlas>_crops.npy [B,2,N,N] and <atlas>_manifest.csv")
ap.add_argument("--out", default="./checkpoint/pretrain")
ap.add_argument("--atlases", default="schaefer,schaefer200,schaefer300,aal116,aal3v1,power264,shen268,gordon333")
ap.add_argument("--neg_scope", choices=["all", "dataset"], default="dataset")
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--warmup_epochs", type=float, default=1.0)
ap.add_argument("--mask_start", type=float, default=0.3); ap.add_argument("--mask_end", type=float, default=0.5); ap.add_argument("--mask_ramp_epochs", type=int, default=5)
ap.add_argument("--w_rec", type=float, default=1.0); ap.add_argument("--w_adj", type=float, default=0.1); ap.add_argument("--w_cl", type=float, default=1.0); ap.add_argument("--w_xatlas", type=float, default=0.5)
ap.add_argument("--xatlas_prob", type=float, default=0.5)
ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--batch_table", default="100:256,116:224,166:160,200:144,264:88,268:88,300:72,333:56")
ap.add_argument("--amp", action="store_true"); ap.add_argument("--mem_probe", action="store_true"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--probe_dir", default=None, help="optional ABIDE I directory (schaefer.npy + schaefer_manifest.csv) for a frozen-encoder linear probe")
ap.add_argument("--probe_pheno", default=None, help="ABIDE I phenotypic .xlsx used by the probe")
ap.add_argument("--probe_every", type=int, default=2); ap.add_argument("--resume", default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_node_id", action="store_true"); ap.add_argument("--no_mask_adj", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); dev = "cuda"; torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
BATCH = {int(k): int(v) for k, v in (kv.split(":") for kv in args.batch_table.split(","))}
ATLASES = args.atlases.split(",")
DSID, dsid = {}, {}


def load_data():
    data, keys, index = {}, {}, {}
    for a in ATLASES:
        p = os.path.join(args.data, f"{a}_crops.npy")
        arr = np.load(p, mmap_mode="r") if (args.smoke or args.mem_probe) else np.load(p)
        man = pd.read_csv(os.path.join(args.data, f"{a}_manifest.csv"))
        k = (man.dataset.astype(str) + "|" + man.subject_uid.astype(str) + "|" + man.session.astype(str)).values
        assert len(k) == arr.shape[0], (a, len(k), arr.shape)
        data[a], keys[a], index[a] = arr, k, {kk: i for i, kk in enumerate(k)}
        for d in man.dataset.astype(str).unique(): DSID.setdefault(d, len(DSID))
        dsid[a] = np.array([DSID[d] for d in man.dataset.astype(str)], dtype=np.int64)
        print(f"  {a:12s} {arr.shape} {arr.dtype}", flush=True)
    return data, keys, index


def to_gpu(arr, idx):
    return torch.from_numpy(np.ascontiguousarray(arr[idx])).to(dev, non_blocking=True).float()


def linear_probe(encoder):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    X = np.load(os.path.join(args.probe_dir, "schaefer.npy")); man = pd.read_csv(os.path.join(args.probe_dir, "schaefer_manifest.csv"))
    ph = pd.read_excel(args.probe_pheno)[["SUB_ID", "DX_GROUP", "SITE_ID", "AGE_AT_SCAN"]]
    df = man.assign(SUB_ID=man.subject.astype(int)).merge(ph, on="SUB_ID", how="left")
    y = (df.DX_GROUP.values == 1).astype(int); site = pd.factorize(df.SITE_ID)[0]; age = df.AGE_AT_SCAN.values
    encoder.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(X), 128):
            xb = torch.from_numpy(X[s:s + 128]).float().to(dev); ab = (xb > args.thr).float(); ab = (ab + ab.transpose(1, 2)) / 2
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                outs.append(encoder(xb, ab, "schaefer", "none").float().cpu())
    encoder.train(); Fe = torch.cat(outs).numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=88)
    asd = roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), Fe, y, cv=skf, method="predict_proba")[:, 1])
    sba = balanced_accuracy_score(site, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), Fe, site, cv=skf))
    age_r = np.corrcoef(age, cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=100.0)), Fe, age, cv=5))[0, 1]
    return dict(asd_auc=float(asd), site_balacc=float(sba), age_r=float(age_r))


def main():
    print("loading data:", flush=True); data, keys, index = load_data()
    enc = build_encoder(node_id_emb=not args.no_node_id); model = Pretrainer(enc, temperature=args.temperature).to(dev)
    print(f"params: total {sum(p.numel() for p in model.parameters())/1e6:.2f}M, encoder {sum(p.numel() for p in enc.parameters())/1e6:.2f}M", flush=True)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_.endswith(".bias") or "token" in n_ or "embedding" in n_ or "mask" in n_ or "node_id" in n_ else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.999))
    mask_adj = not args.no_mask_adj
    probe = args.probe_dir is not None and args.probe_pheno is not None

    if args.mem_probe:
        for a in ATLASES:
            N = data[a].shape[2]
            for bs in sorted({BATCH[N], int(BATCH[N] * 1.25)}):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    idx = np.arange(min(bs, data[a].shape[0])); xa, xb = to_gpu(data[a], (idx, 0)), to_gpu(data[a], (idx, 1))
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                        rec, al, cl, g1 = model(xa, xb, a, 0.3, mask_adj, args.thr); clx = model.xatlas(g1, xb, a, args.thr)
                    (rec + 0.1 * al + cl + 0.5 * clx).backward(); opt.zero_grad(set_to_none=True)
                    print(f"  {a:12s} N={N:3d} bs={bs:3d} reserved={torch.cuda.max_memory_reserved()/1e9:5.1f} GB", flush=True)
                except torch.OutOfMemoryError:
                    print(f"  {a:12s} N={N:3d} bs={bs:3d} OOM", flush=True); torch.cuda.empty_cache()
        return

    def epoch_schedule(rng):
        sched = []
        for a in ATLASES:
            n = data[a].shape[0]; bs = BATCH[data[a].shape[2]]; perm = rng.permutation(n)
            sched += [(a, np.sort(perm[i:i + bs])) for i in range(0, n, bs) if len(perm[i:i + bs]) >= 8]
        rng.shuffle(sched); return sched
    rng = np.random.RandomState(args.seed)
    steps_per_epoch = len(epoch_schedule(np.random.RandomState(0))); total = steps_per_epoch * args.epochs; warm = int(steps_per_epoch * args.warmup_epochs)
    print(f"steps/epoch={steps_per_epoch} total={total} warmup={warm} node_id={not args.no_node_id} mask_adj={mask_adj} xatlas_prob={args.xatlas_prob} neg_scope={args.neg_scope} datasets={len(DSID)}", flush=True)
    start_epoch, step = 0, 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"] + 1; step = ck["step"]
    log = open(os.path.join(args.out, "train_log.tsv"), "a")
    if probe and start_epoch == 0 and not args.smoke:
        pr = linear_probe(model.encoder); print(f"PROBE epoch 0: {pr}", flush=True); log.write(f"PROBE 0 {json.dumps(pr)}\n"); log.flush()
    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); sched = epoch_schedule(rng)
        if args.smoke: sched = sched[:20]
        ratio = args.mask_start + (args.mask_end - args.mask_start) * min(1.0, epoch / max(1, args.mask_ramp_epochs))
        agg = dict(rec=0.0, adj=0.0, cl=0.0, clx=0.0, n=0, nx=0)
        for i, (a, idx) in enumerate(sched):
            lr = args.lr * (step + 1) / warm if step < warm else args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g_ in opt.param_groups: g_["lr"] = lr
            xa, xb = to_gpu(data[a], (idx, 0)), to_gpu(data[a], (idx, 1))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                grp = torch.from_numpy(dsid[a][idx]).to(dev) if args.neg_scope == "dataset" else None
                rec, adj_loss, cl, g1 = model(xa, xb, a, ratio, mask_adj, args.thr, grp)
                loss = args.w_rec * rec + args.w_adj * adj_loss + args.w_cl * cl
                clx = None
                if args.xatlas_prob > 0 and random.random() < args.xatlas_prob and len(ATLASES) > 1:
                    b = random.choice([o for o in ATLASES if o != a]); kb = index[b]
                    pairs = [(j, kb[k]) for j, k in enumerate(keys[a][idx]) if k in kb]
                    random.shuffle(pairs); pairs = pairs[:BATCH[data[b].shape[2]]]
                    if len(pairs) >= 16:
                        ja = torch.tensor([p[0] for p in pairs], device=dev); jb = np.array([p[1] for p in pairs])
                        xc = to_gpu(data[b], (jb, 1))
                        clx = model.xatlas(g1[ja], xc, b, args.thr, None if grp is None else grp[ja]); loss = loss + args.w_xatlas * clx
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            agg["rec"] += rec.item(); agg["adj"] += adj_loss.item(); agg["cl"] += cl.item(); agg["n"] += 1
            if clx is not None: agg["clx"] += clx.item(); agg["nx"] += 1
            if (i + 1) % 200 == 0 or args.smoke:
                el = time.time() - t0
                print(f"  ep {epoch+1} step {i+1}/{len(sched)} rec={agg['rec']/agg['n']:.4f} adj={agg['adj']/agg['n']:.4f} cl={agg['cl']/agg['n']:.4f} clx={agg['clx']/max(agg['nx'],1):.4f} lr={lr:.2e} mask={ratio:.2f} | {el/60:.1f} min, eta {el/(i+1)*(len(sched)-i-1)/60:.1f} min", flush=True)
        n = max(agg["n"], 1)
        line = f"EPOCH {epoch+1}/{args.epochs} rec={agg['rec']/n:.4f} adj={agg['adj']/n:.4f} cl={agg['cl']/n:.4f} clx={agg['clx']/max(agg['nx'],1):.4f} mask={ratio:.2f} time={(time.time()-t0)/60:.1f}min"
        if probe and not args.smoke and ((epoch + 1) % args.probe_every == 0 or epoch + 1 == args.epochs):
            pr = linear_probe(model.encoder); line += f" | PROBE {json.dumps(pr)}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "args": vars(args)}, os.path.join(args.out, "ckpt_last.pth"))
        enc_sd = {("encoder." + k): v for k, v in model.encoder.state_dict().items()}
        torch.save(enc_sd, os.path.join(args.out, "BrainGFM_pretrained.pth"))
        if (epoch + 1) % 2 == 0: torch.save(enc_sd, os.path.join(args.out, f"encoder_epoch{epoch+1}.pth"))
    print("PRETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
