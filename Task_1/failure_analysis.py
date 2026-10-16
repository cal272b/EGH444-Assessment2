"""Failure analysis helpers for evaluated runs (EGH444 Task 2 -> motivates Task 3).

All functions read the outputs written by rgb_segmentation.py --evaluate-checkpoint:
  <eval run>/<split>/confusion_matrix.npy, per_image.csv, metrics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_runs import NYU40


def top_confusions(eval_dir, split='test', k=12, min_share=0.1):
    """Most frequent class -> class mistakes, as % of the true class's pixels.

    Only classes with at least `min_share` % of labelled pixels are listed, so tiny
    classes don't dominate the ranking with a handful of pixels.
    """
    cm = np.load(Path(eval_dir) / split / 'confusion_matrix.npy').astype(float)
    gt = cm.sum(1)
    share = 100 * gt / gt.sum()
    rate = 100 * cm / np.maximum(gt[:, None], 1)
    np.fill_diagonal(rate, 0)
    rate[share < min_share, :] = 0
    order = np.argsort(-rate, axis=None)[:k]
    rows = []
    for flat in order:
        t, p = np.unravel_index(flat, rate.shape)
        rows.append({'true': NYU40[t], 'predicted_as': NYU40[p],
                     '%_of_true_pixels': round(rate[t, p], 1),
                     'true_class_IoU_%': round(100 * cm[t, t] / max(gt[t] + cm[:, t].sum() - cm[t, t], 1), 1)})
    return pd.DataFrame(rows)


def scene_breakdown(runs, split='test', bins=3):
    """Mean per-image mIoU by brightness and clutter tertiles, for each model.

    runs: {display name: eval run folder}. Tertile edges come from the first run so
    every model is binned on the same images.
    """
    names = list(runs)
    frames = {n: pd.read_csv(Path(runs[n]) / split / 'per_image.csv') for n in names}
    ref = frames[names[0]]
    if 'brightness' not in ref:
        raise ValueError('Re-evaluate with the current rgb_segmentation.py to get scene descriptors.')
    out = []
    for column, labels in [('brightness', ['dark', 'medium', 'bright']),
                           ('num_classes', ['few classes', 'some', 'cluttered'])]:
        bucket = pd.qcut(ref[column].rank(method='first'), bins, labels=labels)
        for label in labels:
            ids = set(ref.loc[bucket == label, 'id'])
            row = {'factor': column, 'bin': label, 'images': len(ids),
                   'range': f"{ref.loc[bucket == label, column].min():.2f}-{ref.loc[bucket == label, column].max():.2f}"}
            for n in names:
                f = frames[n]
                row[f'{n}_mIoU_%'] = round(100 * f.loc[f.id.isin(ids), 'mIoU'].mean(), 1)
            out.append(row)
    return pd.DataFrame(out)


def boundary_summary(runs, split='test'):
    """Pixel accuracy near ground-truth class edges versus object interiors."""
    rows = []
    for n, d in runs.items():
        m = json.loads((Path(d) / split / 'metrics.json').read_text())
        if 'boundary_PixelAcc' not in m:
            continue
        rows.append({'model': n, 'boundary_PixelAcc_%': round(100 * m['boundary_PixelAcc'], 1),
                     'interior_PixelAcc_%': round(100 * m['interior_PixelAcc'], 1),
                     'gap_points': round(100 * (m['interior_PixelAcc'] - m['boundary_PixelAcc']), 1)})
    return pd.DataFrame(rows)


def worst_images(eval_dir, split='test', n=8):
    """Lowest-mIoU images with their scene descriptors, for picking qualitative failure examples."""
    f = pd.read_csv(Path(eval_dir) / split / 'per_image.csv')
    return f.sort_values('mIoU').head(n)


def per_image_delta(base_dir, other_dir, split='test'):
    """Per-image mIoU change (other - base), sorted from most helped to most hurt."""
    a = pd.read_csv(Path(base_dir) / split / 'per_image.csv')
    b = pd.read_csv(Path(other_dir) / split / 'per_image.csv')
    m = a.merge(b, on='id', suffixes=('_base', '_other'))
    m['delta_mIoU_points'] = 100 * (m['mIoU_other'] - m['mIoU_base'])
    keep = ['id', 'mIoU_base', 'mIoU_other', 'delta_mIoU_points', 'brightness_base', 'num_classes_base']
    return m[[c for c in keep if c in m]].sort_values('delta_mIoU_points', ascending=False).reset_index(drop=True)


def confusion_change(base_dir, other_dir, pairs, split='test'):
    """How often each (true, predicted) confusion happens before and after, as % of the true class's pixels."""
    rows = []
    cms = [np.load(Path(d) / split / 'confusion_matrix.npy').astype(float) for d in (base_dir, other_dir)]
    for true, pred in pairs:
        t, p = NYU40.index(true), NYU40.index(pred)
        before, after = (100 * cm[t, p] / max(cm[t].sum(), 1) for cm in cms)
        rows.append({'true': true, 'predicted_as': pred, 'before_%': round(before, 1),
                     'after_%': round(after, 1), 'change_points': round(after - before, 1)})
    return pd.DataFrame(rows)


def show_examples(runs, ids, data_root, split='test', title=None, show_depth=False):
    """Plot RGB | ground truth | each model's prediction for chosen image IDs (any image, not just the first 12)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    names = list(runs)
    preds = {n: np.load(Path(runs[n]) / split / 'predictions.npz') for n in names}
    palette = np.random.default_rng(0).integers(30, 256, (40, 3), dtype=np.uint8)   # same colours as rgb_segmentation

    def colour(mask):
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        ok = mask < 40
        out[ok] = palette[mask[ok]]
        return out

    cols = 2 + len(names) + int(show_depth)
    fig, axes = plt.subplots(len(ids), cols, figsize=(2.6 * cols, 2.1 * len(ids)), squeeze=False)
    for r, sid in enumerate(ids):
        sid = str(sid)
        pred0 = preds[names[0]][sid]
        h, w = pred0.shape
        root = Path(data_root)
        rgb_path = next(p for p in (root / 'RGB' / f'{sid}.jpg', root / 'RGB' / f'{sid}.png') if p.exists())
        rgb = np.array(Image.open(rgb_path).convert('RGB').resize((w, h), Image.Resampling.BILINEAR))
        label = np.array(Image.open(root / 'Label' / f'{sid}.png').resize((w, h), Image.Resampling.NEAREST)).astype(int)
        gt = np.where((label >= 1) & (label <= 40), label - 1, 255)
        panels = [('RGB', rgb)]
        if show_depth:
            d = np.array(Image.open(root / 'Depth' / f'{sid}.png'))
            d = d[..., 0] if d.ndim == 3 else d
            panels.append(('Depth', plt.cm.turbo(np.array(Image.fromarray(d).resize((w, h), Image.Resampling.NEAREST)) / 255.0)))
        panels += [('Ground truth', colour(gt))] + [(n, colour(preds[n][sid])) for n in names]
        for c, (name, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
        axes[r, 0].set_ylabel(sid, fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig
