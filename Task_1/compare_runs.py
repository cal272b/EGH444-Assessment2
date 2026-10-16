"""Side-by-side comparison of evaluated RGB baselines (EGH444 Task 1).

Reads the val/ and test/ folders written by rgb_segmentation.py
(--evaluate-checkpoint) and produces:
  comparison_summary.csv        mIoU, pixel accuracy, Dice/F1, ms/image, params
  classwise_iou_<split>.csv     per-class IoU for every model + difference
  classwise_iou_<split>.png     grouped bar chart, classes sorted by frequency
  qualitative_<split>.png       same images, one row per model (RGB|GT|pred|errors)

Usage from Python (e.g. Colab):
    from compare_runs import compare
    compare({'DeepLabV3': Path('runs/.../deeplab-eval'), 'FCN': Path('runs/.../fcn-eval')},
            out_dir=Path('Task_1/comparison'))

All models must have been evaluated on the same splits.json, labels and size,
otherwise the comparison is refused.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# NYUv2-40 class names in label order (mapped IDs 0-39 = raw IDs 1-40).
NYU40 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
         'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator',
         'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person',
         'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
         'otherfurniture', 'otherprop']


def _load(run_dir, split):
    path = Path(run_dir) / split / 'metrics.json'
    return json.loads(path.read_text()) if path.is_file() else None


def _check_same_protocol(runs):
    """Refuse to compare runs evaluated on different samples or settings."""
    reference = None
    for name, run_dir in runs.items():
        config = json.loads((Path(run_dir) / 'config.json').read_text())
        splits = json.loads((Path(run_dir) / 'splits.json').read_text())
        eval_size = config.get('eval_size') or config['size']
        key = (json.dumps(splits, sort_keys=True), tuple(eval_size), config['label_scheme'])
        if reference is None:
            reference = (name, key)
        elif key != reference[1]:
            raise ValueError(f'{name} was evaluated with different splits/eval size/labels from {reference[0]}.')


def _parameter_counts(runs):
    """Count trainable-weight parameters from each evaluated checkpoint (includes any depth branch)."""
    try:
        import torch
        counts = {}
        for name, run_dir in runs.items():
            ckpt = json.loads((Path(run_dir) / 'config.json').read_text()).get('evaluate_checkpoint')
            state = torch.load(ckpt, map_location='cpu', weights_only=True)['model_state']
            counts[name] = sum(v.numel() for k, v in state.items()
                               if 'running_' not in k and 'num_batches' not in k) / 1e6
        return counts
    except Exception:
        pass
    return _parameter_counts_by_rebuild(runs)


def _parameter_counts_by_rebuild(runs):
    """Fallback: count parameters by rebuilding each architecture (no weights needed)."""
    try:
        from rgb_segmentation import build_model
    except ImportError:
        return {}
    counts = {}
    for name, run_dir in runs.items():
        results = next(Path(run_dir).glob('*_results.txt'), None)
        arch = results.stem.replace('_results', '') if results else None
        if arch:
            model = build_model(arch, False)
            counts[name] = sum(p.numel() for p in model.parameters()) / 1e6
    return counts


def compare(runs, out_dir, splits=('val', 'test'), qualitative_count=4):
    """runs: {display name: evaluation run folder}. Returns summary rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_same_protocol(runs)
    params = _parameter_counts(runs)
    names = list(runs)

    summary = []
    for split in splits:
        metrics = {n: _load(runs[n], split) for n in names}
        if any(m is None for m in metrics.values()):
            print(f'[skip] {split}: not evaluated for every model')
            continue
        for n in names:
            m = metrics[n]
            summary.append({'split': split, 'model': n,
                            'mIoU_%': round(100 * m['mIoU'], 2),
                            'PixelAcc_%': round(100 * m['PixelAcc'], 2),
                            'mean_Dice_F1_%': round(100 * m['mean_Dice_F1'], 2),
                            'boundary_PixelAcc_%': round(100 * m['boundary_PixelAcc'], 2) if 'boundary_PixelAcc' in m else '',
                            'interior_PixelAcc_%': round(100 * m['interior_PixelAcc'], 2) if 'interior_PixelAcc' in m else '',
                            'inference_ms_per_image': round(m['inference_ms_per_image'], 2),
                            'params_M': round(params[n], 2) if n in params else ''})

        # Class-wise IoU table, ordered by how many labelled pixels each class has.
        pixels = np.array(metrics[names[0]]['valid_pixels_per_class'])
        order = np.argsort(-pixels)
        iou = {n: np.array([np.nan if v is None else v for v in metrics[n]['IoU_per_class']]) for n in names}
        rows = []
        for c in order:
            row = {'class_id': int(c), 'class': NYU40[c],
                   'pixel_share_%': round(100 * pixels[c] / pixels.sum(), 3)}
            for n in names:
                row[f'{n}_IoU_%'] = round(100 * iou[n][c], 2)
            if len(names) == 2:
                row[f'diff_{names[0]}_minus_{names[1]}'] = round(100 * (iou[names[0]][c] - iou[names[1]][c]), 2)
            rows.append(row)
        _write_csv(out_dir / f'classwise_iou_{split}.csv', rows)

        fig, ax = plt.subplots(figsize=(16, 4.5))
        width = 0.8 / len(names)
        x = np.arange(len(order))
        for i, n in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * width, 100 * iou[n][order], width,
                   label=f"{n} (mIoU {100 * metrics[n]['mIoU']:.1f}%)")
        ax.set_xticks(x, [NYU40[c] for c in order], rotation=70, ha='right', fontsize=8)
        ax.set_ylabel('IoU (%)')
        ax.set_ylim(0, 100)
        ax.set_title(f'Class-wise IoU on {split} (classes sorted by pixel frequency, most common left)')
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f'classwise_iou_{split}.png', dpi=150)
        _show()
        plt.close(fig)

        # Qualitative: the same image IDs for every model, stacked vertically.
        common = sorted(set.intersection(*[{p.stem for p in (Path(runs[n]) / split).glob('*.png')} for n in names]),
                        key=lambda s: int(s) if s.isdigit() else s)[:qualitative_count]
        if common:
            fig, axes = plt.subplots(len(common), 1, figsize=(10, 2.6 * len(common) * len(names) / 2 + 0.5))
            axes = np.atleast_1d(axes)
            for ax, sid in zip(axes, common):
                panels = [np.array(Image.open(Path(runs[n]) / split / f'{sid}.png')) for n in names]
                ax.imshow(np.concatenate(panels, axis=0))
                ax.set_yticks([(i + 0.5) * panels[0].shape[0] for i in range(len(names))], names)
                ax.set_xticks([(i + 0.5) * panels[0].shape[1] / 4 for i in range(4)],
                              ['RGB', 'Ground truth', 'Prediction', 'Errors (red)'])
                ax.set_title(f'Image {sid}', fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f'qualitative_{split}.png', dpi=150)
            _show()
            plt.close(fig)

    _write_csv(out_dir / 'comparison_summary.csv', summary)
    return summary


def _show():
    """Display inline in notebooks; do nothing in headless scripts."""
    if 'inline' in matplotlib.get_backend().lower():
        plt.show()


def _write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs=2, action='append', metavar=('NAME', 'DIR'), required=True,
                   help='Display name and evaluation folder; repeat for each model')
    p.add_argument('--out', type=Path, default=Path('comparison'))
    a = p.parse_args()
    for row in compare({n: Path(d) for n, d in a.run}, a.out):
        print(row)
