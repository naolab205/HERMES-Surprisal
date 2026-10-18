#!/usr/bin/env python
"""Seven-fold character-bag surprisal decoding, matched to the frozen protocol.

Protocol parity with the frozen JBHI run
----------------------------------------
Same 10 participants, same two books, same 7 contiguous item blocks per book
(test = block f, validation = block f+1, remainder train), same label file
(``uer/gpt2-chinese-cluecorpussmall``, 192-token context), same endpoints:
trial-level Spearman over single readings, item-level Spearman after averaging
readings of a row, and high-vs-low tercile item AUC.

The MM_mid arm changes the representation and architecture: fixed-window
character EEG and eye epochs, a surface Laplacian, hierarchical sequence
encoding, and a gated eye residual. The stored MM_mid results used global
unlabelled eye-channel statistics; new runs default to training-fold statistics.
All arms in this script are exploratory under a shared item-block protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation.align import inverse_sqrt, mean_covariance  # noqa: E402
from foundation.data import (  # noqa: E402
    CACHE_DIR, CharBagDataset, ShardedEpochs, attach_labels_and_splits,
    baseline_correct, load_char_bags, load_char_targets, load_row_bags,
    per_epoch_mad,
)
from foundation.gpu_store import GpuBatcher, GpuEpochStore  # noqa: E402
from foundation.eye_clean import EyeRegressionCleaner  # noqa: E402
from foundation.egan import EEGNetBag, EvokedGeometryNet  # noqa: E402
from foundation.eganv2 import EganSeq  # noqa: E402
from foundation.baselines import build_baseline  # noqa: E402
from foundation.deconvnet import DeconvNet  # noqa: E402
from foundation.hiernet import HierSeqNet, MultiModalHierSeqNet  # noqa: E402
from foundation.seqnet import RowSequenceNet  # noqa: E402
from cogalign.io.retrospective_splits import attach_retrospective_split  # noqa: E402
from foundation.model import (  # noqa: E402
    CharBagModel, SpatialTransform, correlation_loss, sum_constrained_loss,
)


def build_spatial(
    bags, frame: pd.DataFrame, shards: ShardedEpochs, *,
    use_csd: bool, use_ea: bool, rng: np.random.Generator, baseline: bool = True,
    csd_dir: Path = CACHE_DIR,
    ea_shrinkage: float = 0.05, ea_rank_tol: float | None = None,
    n_fit_epochs: int = 4000, eye_cleaner=None, eye_array=None,
) -> SpatialTransform:
    """Fit the per-recording spatial operator and scale on TRAINING epochs only."""
    n_groups = len(bags.groups)
    n_channels = shards.shards[0].shape[1]
    csd = (
        np.load(csd_dir / "csd_matrix.npy").astype(np.float64)
        if use_csd
        else np.eye(n_channels)
    )

    matrices = np.zeros((n_groups, n_channels, n_channels), dtype=np.float32)
    scales = np.ones(n_groups, dtype=np.float32)
    train = frame[frame["split"] == "train"]
    for group_id in range(n_groups):
        rows = train[train["group_id"] == group_id]
        if rows.empty:
            matrices[group_id] = csd.astype(np.float32)
            continue
        picks = np.concatenate(
            [bags.slot_index[int(i)] for i in rows["obs_index"].to_numpy()]
        )
        # Cost scales with epochs x time, so hold the sampled tensor to a fixed
        # budget. A row sequence is ~9x longer than a character epoch, and a
        # covariance estimate needs samples rather than seconds; 4000 x 71 is the
        # budget the character arms used.
        budget = n_fit_epochs * 71
        take = max(200, budget // max(int(shards.shards[0].shape[-1]), 1))
        if len(picks) > take:
            picks = rng.choice(picks, size=take, replace=False)
        sample = shards.take(np.sort(picks))
        if eye_cleaner is not None:
            eye_sample = torch.as_tensor(np.asarray(eye_array[np.sort(picks)]).copy(),
                                         device=eye_cleaner.device)
            sample = eye_cleaner.apply(
                torch.as_tensor(sample, device=eye_cleaner.device).unsqueeze(1),
                eye_sample.unsqueeze(1),
                torch.full((len(picks),), group_id, dtype=torch.long,
                           device=eye_cleaner.device),
            ).squeeze(1).cpu().numpy()
        if baseline:
            sample = sample - sample[..., :13].mean(axis=-1, keepdims=True)
        spatial = csd
        # A float32 matmul over a flattened time axis gives the same covariance
        # and scale as the float64 einsum, orders of magnitude faster; the einsum
        # form stalled for tens of minutes on 660-sample sequences.
        flat = sample.transpose(1, 0, 2).reshape(sample.shape[1], -1).astype(np.float32)
        if use_ea:
            aligned = (csd.astype(np.float32) @ flat).reshape(
                sample.shape[1], sample.shape[0], sample.shape[2]
            ).transpose(1, 0, 2)
            spatial = inverse_sqrt(
                mean_covariance(aligned, shrinkage=ea_shrinkage),
                rank_tol=ea_rank_tol,
            ) @ csd
        scale = float((spatial.astype(np.float32) @ flat).std())
        matrices[group_id] = spatial.astype(np.float32)
        scales[group_id] = 1.0 / max(scale, 1e-8)
    return SpatialTransform(matrices, scales)


def score_frame(frame: pd.DataFrame) -> dict[str, float]:
    trial_rho = float(spearmanr(frame["y"], frame["pred"]).statistic)
    item = frame.groupby("item_id").agg(
        y=("y", "mean"), pred=("pred", "mean"), tercile=("tercile", "first")
    )
    item_rho = float(spearmanr(item["y"], item["pred"]).statistic)
    keep = item["tercile"] != 1
    auc = float(roc_auc_score((item.loc[keep, "tercile"] == 2).astype(int),
                              item.loc[keep, "pred"]))
    return {
        "trial_spearman": trial_rho,
        "item_spearman": item_rho,
        "item_auc_hi_vs_lo": auc,
        "n_items": int(len(item)),
        "n_obs": int(len(frame)),
    }


def compute_eye_stats(eye: np.ndarray, slot_index: list[np.ndarray],
                      frame: pd.DataFrame, scope: str = "train") -> tuple[list[float], list[float]]:
    """Standardize physical eye channels using valid samples from the chosen split."""
    if scope == "train":
        obs_indices = frame.loc[frame["split"] == "train", "obs_index"].to_numpy(dtype=int)
        if len(obs_indices) == 0:
            raise ValueError("No training observations for eye standardization")
        epoch_indices = np.concatenate([slot_index[i] for i in obs_indices])
        selected = eye[epoch_indices]
    elif scope == "global":
        selected = eye
    else:
        raise ValueError(f"Unknown eye statistics scope: {scope}")
    valid = np.asarray(selected[:, 3, :], dtype=np.float32) > 0.5
    if not valid.any():
        raise RuntimeError("No valid eye samples for standardization")
    return (
        [float(np.asarray(selected[:, c, :], dtype=np.float32)[valid].mean())
         for c in range(3)] + [0.0],
        [float(np.asarray(selected[:, c, :], dtype=np.float32)[valid].std())
         for c in range(3)] + [1.0],
    )


def sequence_mask(lengths: np.ndarray, width: int) -> np.ndarray:
    """Per-sample valid-time mask for the padded row sequences."""
    index = np.arange(width)[None, :]
    return (index < np.asarray(lengths)[:, None]).astype(np.float32)


def within_count_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                           book: torch.Tensor, n_slots: torch.Tensor,
                           min_target_gap: float = 0.25) -> torch.Tensor:
    """Rank rows only against different items with matching book and count.

    The book/count fields form training pairs; neither is passed into the
    decoder. Same-item readings have identical targets and are excluded by the
    target-gap condition. This loss directly addresses the count shortcut in
    pooled item Spearman without adding a target-derived test-time feature.
    """
    n = len(pred)
    a, b = torch.triu_indices(n, n, offset=1, device=pred.device)
    gap = target[a] - target[b]
    matched = ((book[a] == book[b]) & (n_slots[a] == n_slots[b]) &
               (gap.abs() >= min_target_gap))
    if not bool(matched.any()):
        return pred.sum() * 0.0
    signed = gap[matched].sign() * (pred[a[matched]] - pred[b[matched]])
    return F.softplus(-signed).mean()


def prepare(eeg, args):
    """Per-epoch preprocessing that belongs to the arm, not to the cache.

    Character epochs carry a 100 ms pre-onset window, so they are baseline
    corrected. Row epochs start at row onset and have no baseline; the frozen
    pipeline normalised them by their own median/MAD instead, which is exactly
    the factor under test here.
    """
    if args.per_epoch_mad:
        # MAD scaling and baseline correction are alternative normalisations,
        # not a pipeline: the frozen builder applied MAD to the un-baselined
        # segment. Chaining them would also divide a baseline-corrected channel
        # by its own near-zero MAD, which diverges.
        return per_epoch_mad(eeg)
    if args.representation == "char":
        eeg = baseline_correct(eeg)
    return eeg


def predict(model, loader, device, readout: str = "row", args=None) -> np.ndarray:
    model.eval()
    out = np.zeros(loader.n_samples, dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            eeg = prepare(batch["eeg"], args)
            pred = (model(eeg, batch["mask"], batch["group"], batch.get("eye"))
                    if args is not None and args.arch in ("mmhier", "deconv")
                    else model(eeg, batch["mask"], batch["group"]))[readout]
            out[batch["index"].cpu().numpy()] = pred.float().cpu().numpy()
    if not np.isfinite(out).all():
        raise RuntimeError(
            f"{int((~np.isfinite(out)).sum())} of {len(out)} predictions are not "
            "finite; the spatial operator or the loss has diverged"
        )
    return out


def run_fold(args, bags, shards, store, fold: int, device: torch.device) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + fold)
    torch.manual_seed(args.seed + fold)

    frame = bags.obs.copy()
    frame["obs_index"] = np.arange(len(frame))
    if args.split_manifest:
        frame = attach_retrospective_split(
            frame, args._split_manifest, fold,
            label_csv=args.label_csv or str(ROOT / "data_cache/surprisal_1829a99ad4.csv"),
        )
    else:
        frame = attach_labels_and_splits(frame, fold=fold, n_blocks=args.n_blocks,
                                         label_csv=args.label_csv)
    if args.eval_split == "val":
        # Keep the full EEG cache for matched input loading but make test rows
        # inaccessible to this fold's training/evaluation loaders.
        frame = frame.loc[frame["split"] != "test"].copy()

    if args.eye_clean:
        cleaner = EyeRegressionCleaner(len(bags.groups), device,
                                       shards.shards[0].shape[-1])
        train = frame[frame["split"] == "train"]
        for group_id in range(len(bags.groups)):
            rows = train.loc[train["group_id"] == group_id, "obs_index"].to_numpy()
            picks = np.unique(np.concatenate([bags.slot_index[int(i)] for i in rows]))
            cleaner.fit_group(store, group_id,
                              torch.as_tensor(picks, dtype=torch.long, device=device))
        store.eye_clean = cleaner
        print(f"[eye-clean] fold={fold} train-only valid samples per group "
              f"{cleaner.fit_counts.cpu().tolist()}", flush=True)
    else:
        store.eye_clean = None

    if args.arch == "mmhier" and not args.no_eye:
        args._eye_stats = compute_eye_stats(args._eye_array, bags.slot_index, frame,
                                            args.eye_stats_scope)
        print(f"[eye-stats] fold={fold} scope={args.eye_stats_scope} "
              f"mean={np.round(args._eye_stats[0], 3)} "
              f"std={np.round(args._eye_stats[1], 3)}", flush=True)

    spatial = build_spatial(
        bags, frame, shards, use_csd=args.csd, use_ea=args.ea, rng=rng,
        baseline=(args.representation == "char"),
        ea_shrinkage=args.ea_shrinkage, ea_rank_tol=args.ea_rank_tol,
        csd_dir=Path(args.cache_dir) if args.cache_dir else CACHE_DIR,
        eye_cleaner=store.eye_clean, eye_array=args._eye_array if args.eye_clean else None,
    )
    if args.arch in ("conformer", "atcnet", "msvtnet", "ctnet"):
        model = build_baseline(args.arch, spatial, n_times=shards.shards[0].shape[2],
                               dropout=args.dropout).to(device)
    elif args.arch == "deconv":
        model = DeconvNet(
            spatial, d_model=args.d_model, stride=args.seq_stride,
            time_depth=args.seq_depth, char_depth=args.char_depth,
            max_slots=args.max_slots, block=args.seq_block, dropout=args.dropout,
            deconv_steps=args.deconv_steps, deconv_radius=args.deconv_radius,
            use_deconv=not args.no_deconv, use_eye=args.deconv_eye,
            eye_stats=getattr(args, "_eye_stats", None),
            char_head=args.char_supervision,
        ).to(device)
    elif args.arch == "mmhier":
        model = MultiModalHierSeqNet(
            spatial, d_model=args.d_model, stride=args.seq_stride,
            time_depth=args.seq_depth, char_depth=args.char_depth,
            max_slots=args.max_slots, block=args.seq_block, dropout=args.dropout,
            use_eye=not args.no_eye, use_char_block=not args.no_char_block,
            eye_stats=getattr(args, "_eye_stats", None), rotary=not args.no_rope,
            late_fusion=args.late_fusion, overlap_demix=args.overlap_demix,
            centered_evidence=args.centered_evidence, llm_prior=args.llm_prior,
        ).to(device)
    elif args.arch == "eganseq":
        model = EganSeq(
            spatial, n_times=shards.shards[0].shape[2], d_model=args.d_model,
            stride=args.seq_stride, time_depth=args.seq_depth,
            char_depth=args.char_depth, max_slots=args.max_slots,
            block=args.seq_block, dropout=args.dropout,
            n_filters=args.n_filters, n_prototypes=args.n_prototypes,
            n_manifold=args.n_manifold, use_geometry=not args.no_geometry,
            couple=not args.no_couple, pool_mode=args.pool_mode,
            use_char_block=not args.no_char_block,
        ).to(device)
    elif args.arch == "hier":
        model = HierSeqNet(
            spatial, d_model=args.d_model, stride=args.seq_stride,
            time_depth=args.seq_depth, char_depth=args.char_depth,
            max_slots=args.max_slots, block=args.seq_block, dropout=args.dropout,
            use_char_block=not args.no_char_block,
        ).to(device)
    elif args.arch == "seq":
        model = RowSequenceNet(
            spatial, d_model=args.d_model, stride=args.seq_stride,
            depth=args.seq_depth, block=args.seq_block, dropout=args.dropout,
        ).to(device)
    elif args.arch == "egan":
        model = EvokedGeometryNet(
            spatial, n_times=shards.shards[0].shape[2], dropout=args.dropout,
            n_filters=args.n_filters, n_prototypes=args.n_prototypes,
            n_manifold=args.n_manifold, couple=not args.no_couple,
            use_geometry=not args.no_geometry, pool_mode=args.pool_mode,
        ).to(device)
    elif args.arch == "eegnet":
        model = EEGNetBag(spatial, n_times=shards.shards[0].shape[2],
                          dropout=args.dropout).to(device)
    else:
        model = CharBagModel(
            spatial, pooling=args.pooling, dropout=args.dropout,
            char_head=args.char_supervision, width=args.width,
        ).to(device)

    if args.matched_train_rng:
        # Candidate-only parameters are instantiated after shared layers but
        # still consume random draws. Reset stochastic training operations so
        # baseline/candidate differences do not come from dropout masks.
        torch.manual_seed(args.seed + fold + 100000)

    generator = torch.Generator(device=device).manual_seed(args.seed + fold)
    char_scale = 1.0
    loaders = {}
    for split in (("train", "val") if args.eval_split == "val"
                  else ("train", "val", "test")):
        subset = frame[frame["split"] == split].reset_index(drop=True)
        subset.attrs.update(frame.attrs)
        char_targets, char_valid = None, None
        if args.char_supervision:
            char_targets, char_valid = load_char_targets(
                subset, bags.slot_index, bags.obs, max_slots=args.max_slots,
                physical_slots=args._physical_slots, return_validity=True,
            )
        if split == "train" and char_targets is not None:
            # Scale of the character targets on TRAINING bags only.
            train_mask = ((np.arange(char_targets.shape[1])[None, :] <
                           subset["n_slots"].to_numpy()[:, None]) & (char_valid[:, None] > .5))
            char_scale = float(char_targets[train_mask].std())
        if args.char_supervision:
            # Always use REAL training contribution spread to normalize all
            # arms' instance term, including the row-label/shuffled controls.
            if args.char_target_mode == "row":
                for i, n in enumerate(subset.n_slots.to_numpy(dtype=int)):
                    char_targets[i, :n] = float(subset.y.iloc[i])
            elif args.char_target_mode == "within-row-shuffle" and split == "train":
                for i, item in enumerate(subset.item_id):
                    n = min(int(subset.n_slots.iloc[i]), args.max_slots)
                    shuffle_seed = int.from_bytes(hashlib.sha256(
                        f"{args.seed}|{item}".encode()).digest()[:8], "little")
                    char_targets[i, :n] = np.random.default_rng(shuffle_seed).permutation(
                        char_targets[i, :n])
            print(f"[char-labels] fold={fold} split={split} mode={args.char_target_mode} "
                  f"complete={int(char_valid.sum())}/{len(char_valid)}", flush=True)
        loaders[split] = (
            subset,
            GpuBatcher(
                store,
                subset["obs_index"].to_numpy(),
                subset["y"].to_numpy(dtype=np.float32),
                subset["group_id"].to_numpy(dtype=np.int64),
                book_ids=subset["book"].map({"GarnettDream": 0,
                                             "LittlePrince": 1}).to_numpy(dtype=np.int64),
                item_ids=(pd.factorize(subset["item_id"])[0]
                          if split == "train" and args.item_pair_batches else None),
                paired_items=(split == "train" and args.item_pair_batches),
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                char_targets=char_targets,
                char_valid=char_valid,
                # A row sequence's valid length correlates 0.997 with presentation
                # duration, which the leakage contract forbids as a model input.
                # Handing the true length to the model through the time mask
                # leaks it: single-reading rho jumped to 0.32 on fold 0 while item
                # rho fell, the signature of a row-constant nuisance rather than a
                # better neural representation. By default the mask stays uniform
                # so padding is indistinguishable from signal; --mask-lengths
                # restores the leaking behaviour for the audit arm only.
                # A uniform mask still has to span the full time axis -- the
                # sequence model pools over time, so a one-element slot mask is
                # not a substitute. Every row therefore reports the padded width,
                # which makes padding indistinguishable from signal.
                time_lengths=(
                    (subset["length"].to_numpy() if args.mask_lengths
                     else np.full(len(subset), int(shards.shards[0].shape[-1])))
                    if args.representation == "rowseq" else None
                ),
                generator=generator,
            ),
        )

    if args.char_supervision:
        print(f"  fold{fold} train character-target scale {char_scale:.3f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(loaders["train"][1])) * args.epochs
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = {"val": -np.inf, "state": None, "epoch": -1,
            "trial": float("nan"), "item": float("nan")}
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        total = 0.0
        for batch in loaders["train"][1]:
            eeg = prepare(batch["eeg"], args)
            mask = batch["mask"]
            group = batch["group"]
            y = batch["y"]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = (model(eeg, mask, group, batch.get("eye"))
                       if args.arch in ("mmhier", "deconv") else model(eeg, mask, group))
                loss = torch.nn.functional.mse_loss(out["row"], y)
                loss = loss + args.lambda_corr * correlation_loss(out["row"], y)
                if args.item_mean_weight > 0:
                    # Adjacent rows in the training batch are two distinct
                    # readings of one item. The model still sees one reading
                    # at inference; this term optimizes their mean prediction.
                    if not torch.equal(batch["item"][0::2], batch["item"][1::2]):
                        raise RuntimeError("Paired training batch lost item alignment")
                    mean_pred = out["row"].reshape(-1, 2).mean(dim=1)
                    loss = loss + args.item_mean_weight * F.mse_loss(
                        mean_pred, y[0::2]
                    )
                if args.within_count_rank_weight > 0:
                    loss = loss + args.within_count_rank_weight * within_count_rank_loss(
                        out["row"], y, batch["book"], mask.sum(dim=1)
                    )
                if "aux" in out and args.lambda_aux > 0:
                    # MSVTNet's intermediate supervision: each multi-scale branch
                    # gets its own loss so the convolutional branches and the
                    # transformer integrate, as the published design requires.
                    for aux in out["aux"]:
                        loss = loss + args.lambda_aux * torch.nn.functional.mse_loss(aux, y)
                if args.char_supervision:
                    # True per-character labels under the exact sum identity.
                    constrained, row_from_chars = sum_constrained_loss(
                        out["char"], mask, batch["y_char"], y,
                        lambda_instance=args.lambda_instance,
                        lambda_consistency=args.lambda_consistency,
                        char_scale=char_scale,
                        supervision_valid=batch["char_supervision_valid"],
                    )
                    loss = loss + constrained
                    complete = batch["char_supervision_valid"] > .5
                    loss = loss + args.lambda_corr * correlation_loss(row_from_chars[complete], y[complete])
                elif args.lambda_char > 0:
                    # Fallback used by the step-1 arms: every character inherits
                    # the row label. Weaker, but it needs no character labels.
                    char_mse = ((out["char"] - y.unsqueeze(1)) ** 2 * mask).sum() / mask.sum()
                    loss = loss + args.lambda_char * char_mse
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss.detach())

        val_frame, val_loader = loaders["val"]
        val_frame = val_frame.copy()
        val_frame["pred"] = predict(model, val_loader, device, args.readout, args)
        val_metrics = score_frame(val_frame)
        monitor = val_metrics[args.selection_metric]
        print(
            f"  fold{fold} ep{epoch:02d} loss={total / max(1, len(loaders['train'][1])):.4f} "
            f"val_trial={val_metrics['trial_spearman']:+.4f} "
            f"val_item={val_metrics['item_spearman']:+.4f} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        if monitor > best["val"] + 1e-4:
            best = {
                "val": monitor,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "trial": val_metrics["trial_spearman"],
                "item": val_metrics["item_spearman"],
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    eval_frame, eval_loader = loaders[args.eval_split]
    eval_frame = eval_frame.copy()
    eval_frame["pred"] = predict(model, eval_loader, device, args.readout, args)
    metrics = score_frame(eval_frame)
    if args.arch == "deconv":
        metrics["deconv_gate_magnitude"] = model.deconv_report()
    metrics.update({"fold": fold, "best_epoch": best["epoch"],
                    "best_val_trial": best["trial"],
                    "best_val_item": best["item"],
                    "selection_metric": args.selection_metric,
                    "eval_split": args.eval_split})
    keep = ["subject", "book", "run", "row_idx", "item_id", "y", "tercile", "pred"]
    return metrics, eval_frame[keep].assign(fold=fold, eval_split=args.eval_split)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="name of this ablation arm")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--eval-split", choices=["val", "test"], default="test",
                    help="val screens new candidates without reading held-out item blocks")
    ap.add_argument("--selection-metric", choices=["trial_spearman", "item_spearman"],
                    default="trial_spearman",
                    help="validation metric for checkpoint selection; default reproduces archive")
    ap.add_argument("--n-blocks", type=int, default=7)
    ap.add_argument("--split-manifest", default=None,
                    help="audited retrospective sentence split CSV; use with --eval-split val")
    ap.add_argument("--csd", action="store_true", help="apply the surface Laplacian")
    ap.add_argument("--ea", action="store_true", help="apply Euclidean Alignment")
    ap.add_argument("--ea-shrinkage", type=float, default=0.05)
    ap.add_argument("--ea-rank-tol", type=float, default=None,
                    help="project out directions below this fraction of the largest eigenvalue")
    ap.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--arch", choices=["cnn", "egan", "eegnet", "seq", "hier", "eganseq", "mmhier", "deconv",
                             "conformer", "atcnet", "msvtnet", "ctnet"], default="cnn")
    ap.add_argument("--char-depth", type=int, default=2)
    ap.add_argument("--no-rope", action="store_true",
                    help="single-factor ablation: drop rotary position only, keeping "
                         "RMSNorm, SwiGLU, LayerScale and DropPath")
    ap.add_argument("--deconv-steps", type=int, default=2)
    ap.add_argument("--deconv-radius", type=int, default=2)
    ap.add_argument("--no-deconv", action="store_true",
                    help="ablate the schedule-conditioned deconvolution")
    ap.add_argument("--deconv-eye", action="store_true",
                    help="add the eye branch to the deconvolution model")
    ap.add_argument("--no-eye", action="store_true",
                    help="ablate the eye branch of the multimodal model")
    ap.add_argument("--eye-stats-scope", choices=["train", "global"], default="train",
                    help="train is the prospective protocol; global reproduces the archived MM_mid run")
    ap.add_argument("--eye-clean", action="store_true",
                    help="fit train-only eye-to-EEG nuisance regression and use EEG-only decoder")
    ap.add_argument("--overlap-demix", action="store_true",
                    help="add a zero-initialized adjacent-window EEG overlap correction")
    ap.add_argument("--centered-evidence", action="store_true",
                    help="add a zero-initialized within-row neural-moment readout")
    ap.add_argument("--llm-prior", choices=["none", "pretrained", "random"],
                    default="none",
                    help="frozen Chinese GPT-2 blocks over EEG embeddings; random is the capacity control")
    ap.add_argument("--no-char-block", action="store_true",
                    help="ablate the cross-character sequence block")
    ap.add_argument("--seq-block", choices=["transformer", "ssm", "modern"], default="transformer")
    ap.add_argument("--late-fusion", action="store_true",
                    help="separate cross-character blocks per modality, fused at row level")
    ap.add_argument("--mask-lengths", action="store_true",
                    help="expose true sequence length through the time mask; "
                         "this LEAKS presentation duration and is for audit only")
    ap.add_argument("--seq-depth", type=int, default=2)
    ap.add_argument("--seq-stride", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-filters", type=int, default=12)
    ap.add_argument("--n-prototypes", type=int, default=6)
    ap.add_argument("--n-manifold", type=int, default=12)
    ap.add_argument("--no-geometry", action="store_true",
                    help="ablate the learnable-xDAWN SPD branch")
    ap.add_argument("--pool-mode", choices=["precision", "mean"], default="precision")
    ap.add_argument("--no-couple", action="store_true",
                    help="ablate the geometry-to-waveform FiLM coupling")
    ap.add_argument("--width", type=float, default=1.0,
                    help="multiplier on every hidden stage of the EEG encoder")
    ap.add_argument("--lambda-corr", type=float, default=0.5)
    ap.add_argument("--within-count-rank-weight", type=float, default=0.0,
                    help="training-only pairwise rank loss within book and character count")
    ap.add_argument("--item-pair-batches", action="store_true",
                    help="sample two distinct training readers per item in adjacent batch positions")
    ap.add_argument("--item-mean-weight", type=float, default=0.0,
                    help="extra MSE on paired-reader mean prediction; requires item-pair-batches")
    ap.add_argument("--lambda-char", type=float, default=0.2)
    ap.add_argument("--lambda-aux", type=float, default=0.3,
                    help="MSVTNet auxiliary branch loss weight")
    ap.add_argument("--char-supervision", action="store_true",
                    help="supervise individual characters under the exact sum identity")
    ap.add_argument("--char-target-mode", choices=["real", "row", "within-row-shuffle"],
                    default="real", help="matched local-supervision controls; inference remains EEG-only")
    ap.add_argument("--lambda-instance", type=float, default=1.0)
    ap.add_argument("--lambda-consistency", type=float, default=1.0)
    ap.add_argument("--representation", choices=["char", "row_frozen", "row_amp", "rowseq"],
                    default="char",
                    help="char = character-onset-locked; row_* = whole row warped to 128 samples")
    ap.add_argument("--per-epoch-mad", action="store_true",
                    help="apply the frozen builder's median/MAD scaling at load time")
    ap.add_argument("--readout", choices=["row", "row_from_chars"], default="row",
                    help="which head is scored; row_from_chars is built only from characters")
    ap.add_argument("--max-slots", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-dir", default=None,
                    help="alternative character cache, e.g. a longer epoch window")
    ap.add_argument("--label-csv", default=None,
                    help="replacement label file; default is the frozen run's own")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--matched-train-rng", action="store_true",
                    help="reset stochastic training RNG after model construction")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.char_supervision and (args.arch not in ("cnn", "deconv") or args.representation != "char"):
        raise ValueError("Character supervision requires a real character head (cnn/deconv) and character epochs")
    if args.split_manifest:
        if args.eval_split != "val" or args.n_blocks != 7:
            raise ValueError("Retrospective split requires --eval-split val and seven blocks")
        args._split_manifest = pd.read_csv(args.split_manifest)
        if not {"item_id", "block", "included"}.issubset(args._split_manifest):
            raise ValueError("Split manifest lacks audited item/block/included fields")
    if args.item_mean_weight < 0 or (args.item_mean_weight > 0 and not args.item_pair_batches):
        raise ValueError("--item-mean-weight must be nonnegative and requires --item-pair-batches")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[arm] {args.arm} csd={args.csd} ea={args.ea} pooling={args.pooling} device={device}")

    if args.representation == "rowseq":
        from foundation.data import load_row_sequences
        bags = load_row_sequences()
        args.max_slots = 1
    elif args.representation == "char":
        bags = load_char_bags(cache_dir=Path(args.cache_dir) if args.cache_dir else CACHE_DIR)
    else:
        bags = load_row_bags(source=args.representation.replace("row_", ""))
        args.max_slots = 1
    shards = ShardedEpochs(bags.epochs)
    if args.char_supervision:
        cache_path = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
        args._physical_slots = np.concatenate([
            pd.read_csv(cache_path / f"meta_{g.split('|')[0]}_{g.split('|')[1]}.csv",
                        usecols=["char_slot"]).char_slot.to_numpy(dtype=int)
            for g in bags.groups
        ])
    print(f"[data] {len(shards)} character epochs, {len(bags.obs)} row observations, "
          f"{len(bags.groups)} recordings", flush=True)
    eye = None
    if (args.arch == "mmhier" and not args.no_eye) or getattr(args, "deconv_eye", False) or args.eye_clean:
        from foundation.data import load_eye_char
        eye, coverage = load_eye_char(
            bags, char_cache=Path(args.cache_dir) if args.cache_dir else None)
        print(f"[eye] {eye.shape}  coverage {100 * coverage:.1f}% "
              f"(missing archives are zero-filled with validity 0)", flush=True)
        args._eye_array = eye
    store = GpuEpochStore(shards, bags.slot_index, device=device,
                          max_slots=args.max_slots, eye=eye)
    print(f"[store] {store.bytes_on_device / 2**30:.1f} GiB resident on {device}", flush=True)

    rows, predictions = [], []
    for fold in args.folds:
        metrics, frame = run_fold(args, bags, shards, store, fold, device)
        metrics["arm"] = args.arm
        rows.append(metrics)
        predictions.append(frame)
        print(f"[fold {fold}] {json.dumps({k: v for k, v in metrics.items() if k != 'arm'})}",
              flush=True)

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out_dir / "tables" / f"charbag_{args.arm}_per_fold.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out_dir / "tables" / f"charbag_{args.arm}_predictions.csv", index=False
    )
    summary = {
        "arm": args.arm,
        "csd": args.csd,
        "ea": args.ea,
        "pooling": args.pooling,
        "n_folds": len(rows),
        **{
            f"{metric}_{stat}": float(getattr(per_fold[metric], stat)())
            for metric in ("trial_spearman", "item_spearman", "item_auc_hi_vs_lo")
            for stat in ("mean", "std")
        },
    }
    (out_dir / "tables" / f"charbag_{args.arm}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
