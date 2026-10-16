# EGH444 Task 1 - RGB-only baselines: DeepLabV3-ResNet50 and FCN-ResNet50
# Adapted from EGH444_Assessment2_Template.py using TorchVision.
# (Previously deeplab_rgb.py; FCN added so both baselines share one protocol.)
# Depth is not used in this part of the project.
#
# Both models use the same dilated ResNet50 backbone (output stride 8), the same
# splits, labels, resolution, optimiser and schedule. The only difference is the
# head: FCN uses a single 3x3 conv head; DeepLabV3 uses ASPP (parallel atrous
# convs + image pooling) for multi-scale context. This isolates the effect of
# context modelling for the Task 1 comparison.
#
# Run these commands from the repo root, with this file inside Task_1/.
# Keep NYUDepthv2 beside the repo, or give --data-root its location.
# The dataset needs RGB/, Label/, train.txt and test.txt.
#
# Train:
# python Task_1/rgb_segmentation.py --data-root ../NYUDepthv2 --model deeplabv3 --run-name deeplab-baseline
# python Task_1/rgb_segmentation.py --data-root ../NYUDepthv2 --model fcn --run-name fcn-baseline
#
# Evaluate (the model type is read from the checkpoint):
# python Task_1/rgb_segmentation.py --data-root ../NYUDepthv2 --evaluate-checkpoint
# runs/rgb/fcn-baseline/fcn/best.pt --run-name fcn-final
# Add --validation-only to skip testing.
#
# For a quick check, add --epochs 1 --limit 8 --no-pretrained when training.
# This checks that the code runs; it is not a proper performance comparison.
#
# Settings used:
# - RGB images and masks resized to 120 x 160 (height x width).
# - 40 classes. Labels 1-40 become 0-39; unlabeled pixels (0) are ignored.
# - Seed 42; 636 training, 159 validation and 654 test images.
# - Batch size 2; up to 12 epochs, stopping after 4 without better val IoU.
# - ImageNet backbone weights; train the new head and ResNet layer4.
# - AdamW with weight decay 0.0001.
# - Head learning rate 0.001; layer4 learning rate 0.0001.
# - Polynomial learning rate decay, exponent 0.9.
# - Cross-entropy loss and random horizontal flips (50% chance).
# - ImageNet RGB normalisation.
# - Backbone BatchNorm statistics stay fixed; the head learns its own.
# - GPU if available (CUDA or Apple MPS), otherwise CPU.
#
# Save the best model based on validation IoU. Training does not use test scores.
# reference_splits.json lists the sample IDs for the group comparison.
#
# Results from the completed run (best model: epoch 11):
# Validation: mIoU 29.05%, pixel accuracy 61.92%.
# Test: mIoU 25.99%, pixel accuracy 57.35%, Dice/F1 38.99%.
# These are one-seed results at reduced resolution; FCN still needs comparing.
# Some test images were inspected in the earlier small checks.
#
# Outputs go to runs/rgb/<run-name>/<model>. Give each run a different name.
# Scores in the saved files are fractions: 0.2599 means 25.99%.
# Example images show RGB, correct labels, predictions and errors in red.
# Black pixels in the label image are ignored. Masks are evaluated at 120 x 160
# by default. Classes absent from labels and predictions are left out of mIoU.
# Keep best.pt with its run's splits.json to evaluate it again.
# --resume loads model weights into a new run with a fresh optimiser.
# Datasets and model weights are not included in this package.
#
# Sources:
# TorchVision: https://docs.pytorch.org/vision/stable/models/deeplabv3.html
# DeepLabV3 paper: https://arxiv.org/abs/1706.05587
# FCN paper: https://arxiv.org/abs/1411.4038
# NYU dataset: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

"""RGB-only DeepLabV3 / FCN training and evaluation for EGH444 Task 1."""

import argparse
import os
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF

# Keep downloaded weights together in a local cache.
os.environ.setdefault('TORCH_HOME', str(Path.cwd() / '.torch-cache'))
if Path('/etc/ssl/cert.pem').is_file():
    os.environ.setdefault('SSL_CERT_FILE', '/etc/ssl/cert.pem')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASSES, IGNORE = 40, 255
MODELS = {'deeplabv3': 'DeepLabV3', 'fcn': 'FCN'}


# Dataset and preprocessing

def read_split(root, name):
    """Read the image IDs for this split."""
    stems = [Path(line.split()[0]).stem for line in
             (root / f'{name}.txt').read_text().splitlines() if line.strip()]
    if len(stems) != len(set(stems)):
        raise ValueError(f'Duplicate samples in {name}.txt')
    return stems


class RGBDataset(Dataset):
    """Load matching photos and label masks."""
    def __init__(self, root, stems, size, label_scheme, training=False):
        self.root, self.stems, self.size = root, stems, size
        self.label_scheme, self.training = label_scheme, training
        self.paths = []
        for stem in stems:
            rgb = next((root / 'RGB' / f'{stem}{ext}' for ext in ('.jpg', '.png')
                        if (root / 'RGB' / f'{stem}{ext}').is_file()), None)
            mask = root / 'Label' / f'{stem}.png'
            if rgb is None or not mask.is_file():
                raise FileNotFoundError(f'Missing RGB or label for {stem}')
            self.paths.append((rgb, mask))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rp, lp = self.paths[index]
        with Image.open(rp) as image:
            rgb = image.convert('RGB').resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
        with Image.open(lp) as image:
            mask = np.array(image.resize((self.size[1], self.size[0]), Image.Resampling.NEAREST), dtype=np.int64)
        if mask.ndim != 2:
            raise ValueError(f'Expected a single-channel class-ID mask: {lp}')
        # Ignore unlabeled pixels and shift the class IDs to start at zero.
        if self.label_scheme == 'one-based':
            valid = (mask >= 1) & (mask <= CLASSES)
            mask = np.where(valid, mask - 1, IGNORE)
        else:
            if np.any(mask == 40):
                raise ValueError('Found ID 40 with zero-based labels. Confirm label convention.')
            mask = np.where((mask >= 0) & (mask < CLASSES), mask, IGNORE)
        # Flip the photo and mask together so they still line up.
        if self.training and random.random() < 0.5:
            rgb = TF.hflip(rgb)
            mask = np.flip(mask, axis=1).copy()
        return TF.normalize(TF.to_tensor(rgb), MEAN, STD), torch.from_numpy(mask.copy()), self.stems[index]


# Model and evaluation

def build_model(name, pretrained):
    """Set up DeepLabV3 or FCN (ResNet50) with 40 output classes."""
    if name not in MODELS:
        raise ValueError(f'Unknown model {name}; choose from {list(MODELS)}')
    constructor = getattr(torchvision.models.segmentation, f'{name}_resnet50')
    # Stop if the weights fail to download instead of using random weights.
    model = constructor(weights=None, weights_backbone='IMAGENET1K_V1' if pretrained else None,
                        num_classes=CLASSES, aux_loss=False)
    return model


def metrics_from_confusion(cm):
    """Work out IoU, Dice and pixel accuracy from the confusion matrix."""
    diagonal = cm.diagonal().astype(float)
    gt, pred = cm.sum(1), cm.sum(0)
    union = gt + pred - diagonal
    iou = np.divide(diagonal, union, out=np.full(CLASSES, np.nan), where=union > 0)
    dice = np.divide(2 * diagonal, gt + pred, out=np.full(CLASSES, np.nan), where=(gt + pred) > 0)
    return {'mIoU': float(np.nanmean(iou)) if np.any(union) else None,
            'PixelAcc': float(diagonal.sum() / cm.sum()) if cm.sum() else None,
            'mean_Dice_F1': float(np.nanmean(dice)) if np.any(union) else None,
            'IoU_per_class': [float(v) if np.isfinite(v) else None for v in iou],
            'Dice_per_class': [float(v) if np.isfinite(v) else None for v in dice],
            'valid_pixels_per_class': gt.tolist()}


def save_visual(rgb, target, prediction, path):
    # Keep class colours the same in the labels and predictions.
    palette = np.random.default_rng(0).integers(30, 256, (CLASSES, 3), dtype=np.uint8)
    def colour(mask):
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        valid = (mask >= 0) & (mask < CLASSES)
        out[valid] = palette[mask[valid]]
        return out
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    image = ((rgb.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    error = np.zeros_like(image)
    error[(target != IGNORE) & (target != prediction)] = (255, 0, 0)
    Image.fromarray(np.concatenate([image, colour(target), colour(prediction), error], axis=1)).save(path)


@torch.no_grad()
def evaluate(model, loader, device, output=None, visual_count=6):
    model.eval()
    cm = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    rows, elapsed, samples = [], 0.0, 0
    if output:
        output.mkdir(parents=True, exist_ok=True)
    for images, targets, ids in loader:
        images = images.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        predictions = model(images)['out'].argmax(1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        elapsed += time.perf_counter() - start
        predictions = predictions.cpu().numpy()
        for i, sid in enumerate(ids):
            target, prediction = targets[i].numpy(), predictions[i]
            # Only count pixels with a label.
            valid = target != IGNORE
            local = np.bincount(CLASSES * target[valid] + prediction[valid],
                                minlength=CLASSES ** 2).reshape(CLASSES, CLASSES)
            cm += local
            result = metrics_from_confusion(local)
            rows.append({'id': sid, 'mIoU': result['mIoU'], 'PixelAcc': result['PixelAcc']})
            if output and samples < visual_count:
                save_visual(images[i], target, prediction, output / f'{sid}.png')
            samples += 1
    result = metrics_from_confusion(cm)
    result['inference_ms_per_image'] = 1000 * elapsed / max(samples, 1)
    if output:
        np.save(output / 'confusion_matrix.npy', cm)
        write_csv(output / 'per_image.csv', rows)
        (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_results_text(run_dir, config, splits, results_dir, checkpoint, model_name, summary=None):
    """Put the settings and results together for the group to read."""
    def percentage(value):
        return 'not available' if value is None else f'{100 * value:.2f}%'

    label = MODELS[model_name]
    lines = [f'{label} RGB results', '',
             f'Run: {run_dir.name}', f'Checkpoint: {checkpoint}',
             f'Model: {label} with a ResNet50 backbone; RGB only, 40 classes.',
             f"Image size: {config['size'][0]} x {config['size'][1]} (height x width)",
             f"Seed: {config['seed']}; device: {config['device']}",
             'Split sizes: ' + ', '.join(f'{name}={len(ids)}' for name, ids in splits.items())]
    # Evaluation settings may differ from the settings used to train the weights.
    if config.get('evaluate_checkpoint'):
        original = Path(checkpoint).parent.parent / 'config.json'
        if original.is_file():
            training_config = json.loads(original.read_text())
        else:
            training_config = None
    else:
        training_config = config
    if training_config:
        lines += ['', 'Training settings:',
                  f"Epoch budget: {training_config.get('epochs')}; batch size: {training_config.get('batch_size')}",
                  f"Head learning rate: {training_config.get('lr')}; backbone multiplier: {training_config.get('backbone_lr_factor')}",
                  f"Trainable backbone: {training_config.get('trainable_backbone')}; schedule: {training_config.get('schedule')}",
                  f"Pretrained backbone: {training_config.get('pretrained')}",
                  'Loss: cross-entropy; augmentation: random horizontal flips (50%).']
    else:
        lines += ['', 'Original training config.json was unavailable; see the original run for training settings.']
    if summary:
        lines += [f"Best epoch: {summary['best_epoch']}",
                  f"Training time: {summary['training_seconds'] / 60:.2f} minutes"]
    lines += ['', 'Measured results:']
    for split in ('val', 'test'):
        folder = results_dir / split
        metrics_file = folder / 'metrics.json'
        if not metrics_file.is_file():
            lines.append(f'{split}: not evaluated in this run.')
            continue
        scores = json.loads(metrics_file.read_text())
        lines += [f"{split}: mIoU {percentage(scores['mIoU'])}; pixel accuracy {percentage(scores['PixelAcc'])}; Dice/F1 {percentage(scores['mean_Dice_F1'])}",
                  f"Approximate inference time: {scores['inference_ms_per_image']:.2f} ms/image (no dedicated warm-up).",
                  f'Prediction images and per-image scores: {folder.resolve()}',
                  'Class-wise IoU (mapped IDs 0-39):']
        lines += [f'  Class {i}: {percentage(value)}' for i, value in enumerate(scores['IoU_per_class'])]
    lines += ['', 'Reading the outputs:',
              'PNG panels: RGB | correct labels | prediction | errors in red.',
              'Images are initial examples, not a complete failure analysis. Use per_image.csv to find weaker cases.',
              'Unlabeled pixels are ignored. Metrics use the resized masks.',
              'A limited-image run checks the pipeline only; it is not a full baseline result.' if config.get('limit') else 'These scores are from one seed; compare methods using the same splits and evaluation size.',
              '', 'Ideas for Task 2 (hypotheses to test, not proven improvements):',
              '- Higher image resolution may help small objects; measure the extra time and memory.',
              '- Class weighting or a combined loss may help weaker classes; check class-wise scores.',
              'Test one change at a time against the baseline and select changes using validation results.']
    (run_dir / f'{model_name}_results.txt').write_text('\n'.join(lines) + '\n')


# Training

def run_model(name, args, splits, device, run_dir):
    """Train and keep the model with the best validation IoU."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    folder = run_dir / name
    folder.mkdir()
    loaders = {}
    for split, stems in splits.items():
        dataset = RGBDataset(args.data_root, stems, args.size, args.label_scheme, split == 'train')
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train',
                                    num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    model = build_model(name, args.pretrained).to(device)
    if args.trainable_backbone == 'frozen':
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    elif args.trainable_backbone == 'last-stage':
        for key, parameter in model.backbone.named_parameters():
            parameter.requires_grad_(key.startswith('layer4.'))
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.lr * args.backbone_lr_factor},
        {'params': model.classifier.parameters(), 'lr': args.lr}], weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / args.epochs) ** 0.9)
                 if args.schedule == 'poly' else None)
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=True)
        if previous['label_scheme'] != args.label_scheme:
            raise ValueError('Checkpoint label mapping differs.')
        model.load_state_dict(previous['model_state'])
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best, history, stale = -1.0, [], 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # Keep the pretrained backbone's BatchNorm statistics fixed.
        for layer in model.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        total_loss, batches = 0.0, 0
        for images, targets, _ in loaders['train']:
            if not (targets != IGNORE).any():
                continue
            # BatchNorm in the pooled branch needs at least two images.
            # For a single image, use its saved statistics.
            for layer in model.classifier.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.train(images.shape[0] > 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device))['out'], targets.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; stop rather than save misleading results.')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if not batches:
            raise ValueError('No valid training labels found.')
        validation = evaluate(model, loaders['val'], device)
        if validation['mIoU'] is None:
            raise ValueError('No valid validation labels found.')
        row = {'epoch': epoch, 'train_loss': total_loss / batches,
               'val_mIoU': validation['mIoU'], 'val_PixelAcc': validation['PixelAcc'],
               'epoch_seconds': time.perf_counter() - epoch_start, 'head_lr': optimizer.param_groups[1]['lr']}
        history.append(row)
        write_csv(folder / 'training.csv', history)
        print(f"{name}: epoch {epoch}/{args.epochs}, loss={row['train_loss']:.4f}, val mIoU={row['val_mIoU']:.4f}", flush=True)
        # Save a checkpoint when validation IoU improves.
        if validation['mIoU'] > best:
            stale = 0
            best = validation['mIoU']
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'model': name, 'num_classes': CLASSES,
                        'label_scheme': args.label_scheme, 'size': args.size,
                        'val_mIoU': best}, folder / 'best.pt')
        else:
            stale += 1
        if scheduler:
            scheduler.step()
        if args.patience and stale >= args.patience:
            print(f'Early stopping after {stale} epochs without validation improvement.', flush=True)
            break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    test = {}
    for split in (('val', 'test') if args.final_test else ('val',)):
        results = evaluate(model, loaders[split], device, folder / split)
        if split == 'test':
            test = results
    return {'model': name, 'best_epoch': checkpoint['epoch'], 'val_mIoU': best,
            'test_mIoU': test.get('mIoU'), 'test_PixelAcc': test.get('PixelAcc'),
            'test_Dice_F1': test.get('mean_Dice_F1'), 'training_seconds': training_seconds,
            'inference_ms_per_image': test.get('inference_ms_per_image'),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# Run settings

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True, help='Dataset folder containing RGB, Label, train.txt and test.txt')
    parser.add_argument('--model', choices=list(MODELS), default='deeplabv3',
                        help='Architecture to train (ignored with --evaluate-checkpoint; read from the checkpoint)')
    parser.add_argument('--output-root', type=Path, default=Path('runs/rgb'), help='Folder for generated runs')
    parser.add_argument('--label-scheme', default='one-based', choices=['one-based', 'zero-based'],
                        help='one-based: IDs 1..40, ignore 0; zero-based: IDs 0..39')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True, help='Use ImageNet backbone weights; --no-pretrained disables')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--size', type=int, nargs=2, default=[120, 160], metavar=('H', 'W'))
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--trainable-backbone', choices=['full', 'last-stage', 'frozen'], default='last-stage',
                        help='Choose how much of ImageNet backbone to fine-tune')
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1,
                        help='Multiplier on head LR for the pretrained backbone')
    parser.add_argument('--schedule', choices=['constant', 'poly'], default='poly')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience; 0 disables')
    parser.add_argument('--resume', type=Path, help='Fine-tune from these model weights; starts a new optimiser')
    parser.add_argument('--final-test', action='store_true', help='Evaluate test after training; omit during tuning')
    parser.add_argument('--validation-only', action='store_true', help='Skip test during checkpoint evaluation')
    parser.add_argument('--evaluate-checkpoint', type=Path, help='Evaluate selected checkpoint without training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--limit', type=int, help='Limit each split for pipeline checks only')
    parser.add_argument('--run-name', default=time.strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("Learning rate must be positive.")
    if args.backbone_lr_factor <= 0 or args.patience < 0:
        parser.error('Backbone LR factor must be positive and patience nonnegative.')
    if args.epochs < 1 or args.batch_size < 1 or min(args.size) < 32 or not 0 < args.val_fraction < 1:
        parser.error('Use positive epochs/batch size, dimensions >=32, and 0<val-fraction<1.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('--run-name must be a simple folder name.')
    training = read_split(args.data_root, 'train')
    test = read_split(args.data_root, 'test')
    if set(training) & set(test):
        raise ValueError('Training and test samples overlap.')
    if len(training) < 2 or not test:
        raise ValueError('Need at least two training samples and one test sample.')
    # Use the same split each time when the seed stays the same.
    random.Random(args.seed).shuffle(training)
    count = min(len(training) - 1, max(1, round(len(training) * args.val_fraction)))
    splits = {'train': training[count:], 'val': training[:count], 'test': test}
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    selected = args.device
    if selected == 'auto':
        selected = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(selected)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__),
                  batchnorm_running_statistics='backbone frozen; head trained except singleton batches', ignore_index=IGNORE,
                  metrics_note='Macro averages exclude classes with zero union; values are fractions.')
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    (run_dir / 'splits.json').write_text(json.dumps(splits, indent=2))
    # Save the code used for this run along with its settings.
    (run_dir / 'experiment_source.py').write_text(Path(__file__).read_text())
    (run_dir / 'README.txt').write_text(
        'Visual panels, left to right: RGB | ground truth | prediction | errors in red.\n'
        'Ignored pixels are black in ground truth and excluded from metrics.\n'
        'Class IDs in outputs are 0..39 after mapping.\n'
        'Inference timing includes model forward and argmax, excludes data loading.\n'
        'Timing has no dedicated warm-up; treat it as an approximate comparison.\n'
        'Inspect per_image.csv to find failures; initial images are examples, not curated failures.\n'
        'Use multiple seeds for stronger conclusions. Local masks use IDs 0..40; one-based mapping treats 0 as unlabeled, consistent with NYU documentation.\n')
    print(f'Device: {device}. Split sizes: { {k: len(v) for k, v in splits.items()} }', flush=True)
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=True)
        if checkpoint['label_scheme'] != args.label_scheme or list(checkpoint['size']) != list(args.size):
            raise ValueError('Evaluation size and label scheme must match selected checkpoint.')
        source_splits = args.evaluate_checkpoint.parent.parent / 'splits.json'
        if not source_splits.is_file():
            raise FileNotFoundError('Selected checkpoint must have its original splits.json alongside its run.')
        recorded_splits = json.loads(source_splits.read_text())
        if recorded_splits != splits:
            raise ValueError('Evaluation sample IDs differ from the selected training run. Use matching seed/data/limit.')
        model_name = checkpoint.get('model', 'deeplabv3')
        model = build_model(model_name, False).to(device)
        model.load_state_dict(checkpoint['model_state'])
        print(f'Evaluating {MODELS[model_name]} checkpoint (epoch {checkpoint.get("epoch")})', flush=True)
        for split in (('val',) if args.validation_only else ('val', 'test')):
            dataset = RGBDataset(args.data_root, splits[split], args.size, args.label_scheme)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            result = evaluate(model, loader, device, run_dir / split, visual_count=12)
            print(f"{split}: mIoU={result['mIoU']:.4f}, PixelAcc={result['PixelAcc']:.4f}", flush=True)
        print(f'Final evaluation saved: {run_dir}')
        write_results_text(run_dir, config, splits, run_dir, args.evaluate_checkpoint, model_name)
        return
    summary = run_model(args.model, args, splits, device, run_dir)
    write_csv(run_dir / 'summary.csv', [summary])
    write_results_text(run_dir, config, splits, run_dir / args.model,
                       run_dir / args.model / 'best.pt', args.model, summary)
    print(f'Finished. Results: {run_dir}')


if __name__ == '__main__':
    main()
