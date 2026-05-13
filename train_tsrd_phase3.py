"""
train_tsrd_phase3.py — TSRD Scratch Baseline Optimization (Phase 3)

Phase 3 目标：在 Phase 2 scratch baseline 基础上优化泛化能力。
- 不使用 transfer learning
- 不使用预训练权重 / checkpoint restore
- 保持模型整体架构不变

改进方向：
  1. 自适应 per-class 增强（非统一增强到 300）
  2. Class weight: inverse sqrt frequency（clip max=3.0）
  3. ReduceLROnPlateau（替代固定 milestone）
  4. 适度增加 dropout keep_prob（conv: 0.6->0.7）
  5. 适度增加 L2 beta（0.0001->0.0003, 全部层）
  6. Early stopping: patience=8, min_delta=1e-4
  7. 详细错误分析报告

输出:
  tsrd_runs/scratch_phase3_YYYYMMDD_HHMMSS/
  ├── checkpoints/{best, latest}/
  ├── logs/
  ├── plots/   (confusion_matrix, training_curves, class_f1_comparison, worst_classes)
  └── reports/ (metrics.json, classification_report.txt, phase3_error_analysis.txt,
                config.json, class_weights.txt, class_weights.npy, ...)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)

# ── Argument parser ──
parser = argparse.ArgumentParser(description='TSRD Scratch CNN Training — Phase 3')
parser.add_argument('--batch-size', type=int, default=64, choices=[32, 64],
                    help='Batch size (default: 64)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Max epochs (default: 50)')
parser.add_argument('--initial-lr', type=float, default=0.001,
                    help='Initial learning rate (default: 0.001)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='FC dropout keep_prob (default: 0.5)')
parser.add_argument('--l2', type=float, default=0.0003,
                    help='L2 regularization beta (default: 0.0003)')
parser.add_argument('--aug-target', type=int, default=300,
                    help='Max augmentation target per class (default: 300)')
parser.add_argument('--class-weight-mode', type=str, default='inv_sqrt',
                    choices=['inv_sqrt', 'effective_num'],
                    help='Class weight mode (default: inv_sqrt)')
parser.add_argument('--run-name', type=str, default=None,
                    help='Custom run name (default: auto timestamp)')
parser.add_argument('--smoke-test', action='store_true',
                    help='Run 1 epoch for pipeline verification')
args = parser.parse_args()

# ── 数据模块路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_TSRD = os.path.join(PROJECT_ROOT, 'data', 'TSRD')
sys.path.insert(0, DATA_TSRD)

from tsrd_loader import (
    load_tsrd_images,
    load_signnames_tsrd,
    stratified_split,
    tsrd_augment_single,
    TSRD_TRAIN_DIR,
    TSRD_TEST_DIR,
    SIGNNAMES_CSV,
    IMG_HEIGHT,
    IMG_WIDTH,
)

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
SEED = 2018
np.random.seed(SEED)
tf.set_random_seed(SEED)

NUM_CLASSES = 58
IMG_DEPTH = 1
EPOCHS = 1 if args.smoke_test else args.epochs
BATCH_SIZE = args.batch_size
INITIAL_LR = args.initial_lr
DROP_FC = args.dropout
DROP_CONV = 0.7          # Phase 2: 0.6, Phase 3: 0.7 (less regularization on conv)
L2_BETA = args.l2        # Phase 2: 0.0001 (FC only), Phase 3: all layers
AUG_TARGET_MAX = args.aug_target
CLASS_WEIGHT_MODE = args.class_weight_mode
EARLY_STOP_PATIENCE = 999 if args.smoke_test else 8
EARLY_STOP_MIN_DELTA = 1e-4

# Phase 2 确认的弱类别（用于增强加量 + 错误分析）
PHASE2_WEAK_CLASSES = {5, 15, 23, 34, 50}

# ── 时间戳输出目录 ──
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = args.run_name if args.run_name else f'scratch_phase3_{RUN_ID}'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', run_name)
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
CKPT_LATEST_DIR = os.path.join(CHECKPOINT_DIR, 'latest')
CKPT_BEST_DIR = os.path.join(CHECKPOINT_DIR, 'best')

for d in [LOG_DIR, PLOTS_DIR, REPORTS_DIR, CKPT_LATEST_DIR, CKPT_BEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 路径安全检查 ──
_RUN_DIR_ABS = os.path.normpath(os.path.abspath(OUTPUT_DIR))

def _assert_in_run_dir(path, purpose=''):
    resolved = os.path.normpath(os.path.abspath(path))
    if not resolved.startswith(_RUN_DIR_ABS):
        raise PermissionError(
            f'SAFETY: {purpose} path is outside run dir.\n'
            f'  Path: {resolved}\n'
            f'  Run dir: {_RUN_DIR_ABS}')

# ── Phase 2 结果路径（用于对比） ──
PHASE2_RUN_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', 'scratch_20260513_051658')
PHASE2_METRICS_PATH = os.path.join(PHASE2_RUN_DIR, 'reports', 'metrics.json')
PHASE2_CLASS_REPORT_PATH = os.path.join(PHASE2_RUN_DIR, 'reports', 'classification_report.txt')

print('=' * 60)
print('  Phase 3: TSRD Scratch Baseline Optimization')
print('  58-class scratch CNN — no transfer learning')
print(f'  Run: {run_name}')
if args.smoke_test:
    print('  [SMOKE TEST] 1 epoch pipeline verification')
print('=' * 60)


# ═══════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════
sign_names, _ = load_signnames_tsrd(SIGNNAMES_CSV)
print(f'\nLoaded {len(sign_names)} traffic sign classes.')

X_train_full, y_train_full = load_tsrd_images(TSRD_TRAIN_DIR)
X_test, y_test = load_tsrd_images(TSRD_TEST_DIR)
print(f'\nTraining set: {X_train_full.shape[0]} images')
print(f'Test set:     {X_test.shape[0]} images')
print(f'Label range:  {y_train_full.min()} ~ {y_train_full.max()}')

# ── 分层划分 train/val ──
X_train_raw, X_val_raw, y_train_raw, y_val_raw = stratified_split(
    X_train_full, y_train_full, test_size=0.2, seed=SEED, sign_names=sign_names
)
print(f'\nAfter split — Train: {len(y_train_raw)}  Val: {len(y_val_raw)}')


# ═══════════════════════════════════════════════════════════
# 2. 数据增强（Phase 3: per-class adaptive）
# ═══════════════════════════════════════════════════════════
print('\n--- Augmenting (per-class adaptive) ---')
counter = Counter(y_train_raw)

def per_class_aug_target(count, class_id):
    """
    计算 per-class 增强目标数。
    原则：少样本适当增强，多样本不增强，弱类别额外加量。
    """
    if count >= 200:
        target = count
    elif count >= 100:
        target = min(AUG_TARGET_MAX, int(count * 2.5))
    elif count >= 50:
        target = min(AUG_TARGET_MAX, int(count * 3))
    elif count >= 20:
        target = min(AUG_TARGET_MAX, int(count * 4))
    else:
        target = min(AUG_TARGET_MAX - 50, max(150, int(count * 5)))

    # 弱类别额外加量
    if class_id in PHASE2_WEAK_CLASSES:
        target = min(AUG_TARGET_MAX, target + 50)

    return max(target, count)


def augment_single_phase3(img, class_id):
    """
    对单张图片进行 Phase 3 增强。
    对弱类别使用稍强参数，class 23 使用基准参数（方向敏感）。
    """
    if class_id in PHASE2_WEAK_CLASSES and class_id != 23:
        # 弱类别：稍强范围（class 23 除外，方向敏感）
        aug = tsrd_augment_single(
            img,
            rotation_range=(-10, 10),
            scale_range=(0.85, 1.15),
            translation_range=(-4, 4),
            brightness_range=(0.6, 1.4),
            blur_prob=0.4,
        )
        # 额外高斯噪声（30%）
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32) * 255
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # 额外小剪切
        if np.random.random() < 0.3:
            import skimage.transform as transf
            shear_angle = np.deg2rad(np.random.uniform(-2, 2))
            h, w = aug.shape[:2]
            center = np.array([h, w]) / 2. - 0.5
            tform_center = transf.SimilarityTransform(translation=-center)
            tform_uncenter = transf.SimilarityTransform(translation=center)
            tform_shear = transf.AffineTransform(shear=shear_angle)
            full_tform = tform_center + tform_shear + tform_uncenter
            aug = transf.warp(aug, full_tform, preserve_range=True).astype(np.uint8)
    else:
        # 基准增强
        aug = tsrd_augment_single(img)
        # 额外高斯噪声（20%）
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32) * 255
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # 额外小剪切（20%）
        if np.random.random() < 0.2:
            import skimage.transform as transf
            shear_angle = np.deg2rad(np.random.uniform(-2, 2))
            h, w = aug.shape[:2]
            center = np.array([h, w]) / 2. - 0.5
            tform_center = transf.SimilarityTransform(translation=-center)
            tform_uncenter = transf.SimilarityTransform(translation=center)
            tform_shear = transf.AffineTransform(shear=shear_angle)
            full_tform = tform_center + tform_shear + tform_uncenter
            aug = transf.warp(aug, full_tform, preserve_range=True).astype(np.uint8)
    return aug


aug_X_list = [X_train_raw]
aug_y_list = [y_train_raw]

# 收集需要增强的类别和目标数量
needs_aug = []
for cid, cnt in counter.items():
    target = per_class_aug_target(cnt, cid)
    if cnt < target:
        needs_aug.append((cid, cnt, target - cnt))

total_aug = sum(n for _, _, n in needs_aug)
done = 0

print(f'  Per-class targets computed — total extra samples: {total_aug}')
print(f'  (Phase 2 used uniform AUGMENT_TARGET={AUG_TARGET_MAX})')
print()

for cid, cnt, n_extra in needs_aug:
    target = cnt + n_extra
    indices = np.where(y_train_raw == cid)[0]
    for i in range(n_extra):
        idx = np.random.choice(indices)
        aug_img = augment_single_phase3(X_train_raw[idx], cid)
        aug_X_list.append(aug_img[np.newaxis, ...])
        aug_y_list.append(np.array([cid], dtype=np.int32))
        done += 1
        if done % 500 == 0 or done == total_aug:
            print(f'  Augmenting... {done}/{total_aug}', flush=True)
    name = sign_names.get(cid, ('', ''))[0]
    print(f'  Class {cid:3d} ({name:<20s}): {cnt:4d} → {target:4d} (+{n_extra})', flush=True)

X_train_aug = np.concatenate(aug_X_list, axis=0).astype(np.uint8)
y_train_aug = np.concatenate(aug_y_list, axis=0).astype(np.int32)
X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=SEED)
print(f'\nAugmented training set: {len(y_train_aug)} images')
print(f'  (Phase 2: 17457 images — {"smaller" if len(y_train_aug) < 15000 else "comparable"})')


# ═══════════════════════════════════════════════════════════
# 3. Class Weights (Phase 3: inverse sqrt frequency)
# ═══════════════════════════════════════════════════════════
print('\n--- Computing class weights (inverse sqrt frequency) ---')
counter_aug = Counter(y_train_aug)
class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

if CLASS_WEIGHT_MODE == 'inv_sqrt':
    for cid in range(NUM_CLASSES):
        cnt = counter_aug.get(cid, 0)
        class_weights[cid] = 1.0 / np.sqrt(max(cnt, 1))
elif CLASS_WEIGHT_MODE == 'effective_num':
    # Effective number of samples (Cui et al. 2019)
    # w = (1 - beta) / (1 - beta^cnt), beta = 0.9999
    beta = 0.9999
    for cid in range(NUM_CLASSES):
        cnt = counter_aug.get(cid, 0)
        effective_num = (1.0 - beta) / (1.0 - beta ** max(cnt, 1))
        class_weights[cid] = effective_num

# 归一化到 mean=1
class_weights = class_weights / np.mean(class_weights)
# 裁剪最大值（Phase 3: clip=3.0，避免极少数类别权重过高）
MAX_WEIGHT_CLIP = 3.0
class_weights = np.clip(class_weights, None, MAX_WEIGHT_CLIP)

print(f'  Weight range: {class_weights.min():.3f} ~ {class_weights.max():.3f}')
print(f'  (Phase 2 range: 0.843 ~ 1.003)')

top_weighted = np.argsort(-class_weights)[:8]
print(f'\n  Top weighted classes:')
for cid in top_weighted:
    cnt = counter_aug.get(cid, 0)
    name = sign_names.get(cid, ('Unknown', ''))[0]
    print(f'    Class {cid:3d} (w={class_weights[cid]:.3f}): {cnt:5d} samples — {name}')

# 保存 class weights
np.save(os.path.join(REPORTS_DIR, 'class_weights.npy'), class_weights)

# 保存 class_weights.txt
with open(os.path.join(REPORTS_DIR, 'class_weights.txt'), 'w', encoding='utf-8') as f:
    f.write('Phase 3 Class Weights\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Mode: {CLASS_WEIGHT_MODE}\n')
    f.write(f'Clip max: {MAX_WEIGHT_CLIP}\n')
    f.write(f'Weight range: {class_weights.min():.3f} ~ {class_weights.max():.3f}\n\n')
    f.write(f'  {"Class":>5s}  {"Name":<22s}  {"Weight":>8s}  {"Count":>8s}\n')
    f.write('  ' + '-' * 50 + '\n')
    # Sort by weight descending
    sorted_cids = np.argsort(-class_weights)
    for cid in sorted_cids:
        cnt = counter_aug.get(cid, 0)
        name = sign_names.get(cid, ('', ''))[0]
        f.write(f'  {cid:>5d}  {name:<22s}  {class_weights[cid]:>8.3f}  {cnt:>8d}\n')
print('  class_weights.txt saved')


# ═══════════════════════════════════════════════════════════
# 4. 预处理
# ═══════════════════════════════════════════════════════════
def preprocess(X):
    n = X.shape[0]
    out = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for i in range(n):
        gray = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        out[i, :, :, 0] = gray / 255.0 - 0.5
    return out

X_train_proc = preprocess(X_train_aug)
X_val_proc   = preprocess(X_val_raw)
X_test_proc  = preprocess(X_test)

print(f'\nPreprocessed — Train: {X_train_proc.shape}  Val: {X_val_proc.shape}  Test: {X_test_proc.shape}')


# ═══════════════════════════════════════════════════════════
# 5. 模型定义（与 Phase 2 一致，保留原始 5Conv+3FC 架构）
# ═══════════════════════════════════════════════════════════
MU = 0.0
SIGMA = 0.05
BIAS_INIT = 0.05

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, mean=MU, stddev=SIGMA, seed=SEED), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(BIAS_INIT, shape=shape), name=name)

def conv2d(x, W, strides, padding, name):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)

def max_pool_2x2(x, padding, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name=name)

tf.reset_default_graph()

W = {
    'conv1': weight_variable([3, 3, IMG_DEPTH, 80], 'W_conv1'),
    'conv2': weight_variable([3, 3, 80, 120],       'W_conv2'),
    'conv3': weight_variable([4, 4, 120, 180],       'W_conv3'),
    'conv4': weight_variable([3, 3, 180, 200],       'W_conv4'),
    'conv5': weight_variable([3, 3, 200, 200],       'W_conv5'),
    'fc1':   weight_variable([800, 80],               'W_fc1'),
    'fc2':   weight_variable([80, 80],                'W_fc2'),
    'fc3':   weight_variable([80, NUM_CLASSES],       'W_fc3'),
}
b = {
    'conv1': bias_variable([80],          'b_conv1'),
    'conv2': bias_variable([120],         'b_conv2'),
    'conv3': bias_variable([180],         'b_conv3'),
    'conv4': bias_variable([200],         'b_conv4'),
    'conv5': bias_variable([200],         'b_conv5'),
    'fc1':   bias_variable([80],          'b_fc1'),
    'fc2':   bias_variable([80],          'b_fc2'),
    'fc3':   bias_variable([NUM_CLASSES], 'b_fc3'),
}


def cnn_model(x, keep_prob, keep_p_conv):
    """5Conv + 3FC, 输出 58 类 logits（与 Phase 2 架构完全一致）。"""
    # Block 1
    c1 = conv2d(x, W['conv1'], [1, 1, 1, 1], 'VALID', 'conv1')
    c1 = tf.nn.relu(c1 + b['conv1'], 'conv1_act')
    c1 = tf.nn.dropout(c1, keep_prob=keep_p_conv, name='conv1_drop')
    # Block 2
    c2 = conv2d(c1, W['conv2'], [1, 1, 1, 1], 'SAME', 'conv2')
    c2 = tf.nn.relu(c2 + b['conv2'], 'conv2_act')
    c2 = max_pool_2x2(c2, 'VALID', 'conv2_pool')
    c2 = tf.nn.dropout(c2, keep_prob=keep_p_conv, name='conv2_drop')
    # Block 3
    c3 = conv2d(c2, W['conv3'], [1, 1, 1, 1], 'VALID', 'conv3')
    c3 = tf.nn.relu(c3 + b['conv3'], 'conv3_act')
    c3 = tf.nn.dropout(c3, keep_prob=keep_p_conv, name='conv3_drop')
    # Block 4
    c4 = conv2d(c3, W['conv4'], [1, 1, 1, 1], 'SAME', 'conv4')
    c4 = tf.nn.relu(c4 + b['conv4'], 'conv4_act')
    c4 = max_pool_2x2(c4, 'VALID', 'conv4_pool')
    c4 = tf.nn.dropout(c4, keep_prob=keep_prob, name='conv4_drop')
    # Block 5
    c5 = conv2d(c4, W['conv5'], [1, 1, 1, 1], 'VALID', 'conv5')
    c5 = tf.nn.relu(c5 + b['conv5'], 'conv5_act')
    c5 = max_pool_2x2(c5, 'VALID', 'conv5_pool')
    c5 = tf.nn.dropout(c5, keep_prob=keep_prob, name='conv5_drop')
    # FC layers
    flat = tf.reshape(c5, [tf.shape(c5)[0], -1])
    fc1  = tf.nn.relu(tf.matmul(flat, W['fc1']) + b['fc1'], 'fc1')
    fc1  = tf.nn.dropout(fc1, keep_prob=keep_prob, name='fc1_drop')
    fc2  = tf.nn.relu(tf.matmul(fc1, W['fc2']) + b['fc2'], 'fc2')
    fc2  = tf.nn.dropout(fc2, keep_prob=keep_prob, name='fc2_drop')
    logits = tf.add(tf.matmul(fc2, W['fc3']), b['fc3'], name='logits')
    return logits


# ═══════════════════════════════════════════════════════════
# 6. 计算图
# ═══════════════════════════════════════════════════════════
x_ph   = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH], 'x')
y_ph   = tf.placeholder(tf.int32,   [None], 'y')
kp_ph  = tf.placeholder(tf.float32, name='keep_prob')
kpc_ph = tf.placeholder(tf.float32, name='keep_p_conv')
lr_ph  = tf.placeholder(tf.float32, name='lr')
cw_ph  = tf.placeholder(tf.float32, [NUM_CLASSES], 'class_weight')

logits      = cnn_model(x_ph, kp_ph, kpc_ph)
softmax_op  = tf.nn.softmax(logits, name='softmax')
pred_op     = tf.argmax(logits, 1, name='prediction')

# Loss: 带 class weight 的交叉熵 + L2 正则（Phase 3: 全部层，beta=0.0003）
one_hot_y = tf.one_hot(y_ph, NUM_CLASSES, dtype=tf.float32)
ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
weighted_ce = ce * tf.gather(cw_ph, y_ph)

# Phase 3: L2 on ALL layers, beta=0.0003 (Phase 2: FC only, beta=0.0001)
all_l2 = []
for key in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']:
    all_l2.append(tf.nn.l2_loss(W[key]))
l2_loss = L2_BETA * tf.add_n(all_l2)
loss_op = tf.reduce_mean(weighted_ce) + l2_loss

train_op = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_op)

# Accuracy
correct   = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

# Savers
latest_saver = tf.train.Saver(max_to_keep=20, name='latest')
best_saver = tf.train.Saver(max_to_keep=1, name='best')

print(f'\n  Graph — W_fc3: {W["fc3"].shape}  b_fc3: {b["fc3"].shape}')
print(f'  Dropout: conv={DROP_CONV}, fc={DROP_FC}')
print(f'  L2 beta: {L2_BETA} (all layers)')


# ═══════════════════════════════════════════════════════════
# 7. 评估函数
# ═══════════════════════════════════════════════════════════
def evaluate(sess, X_data, y_data):
    n = len(y_data)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for offset in range(0, n, BATCH_SIZE):
        bx = X_data[offset:offset + BATCH_SIZE]
        by = y_data[offset:offset + BATCH_SIZE]
        feed = {x_ph: bx, y_ph: by, kp_ph: 1.0, kpc_ph: 1.0,
                cw_ph: np.ones(NUM_CLASSES, dtype=np.float32)}
        l, acc, pred = sess.run([loss_op, accuracy_op, pred_op], feed_dict=feed)
        total_loss += l * len(bx)
        all_preds.extend(pred)
        all_labels.extend(by)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / n
    avg_acc    = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, avg_acc, macro_f1, all_preds, all_labels


# ═══════════════════════════════════════════════════════════
# 8. 训练循环
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Training')
print('=' * 60)

history = {
    'epoch': [], 'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': [],
}

best_val_f1  = 0.0
best_epoch   = 0
best_ckpt    = None
patience     = 0
current_lr   = INITIAL_LR

# ── ReduceLROnPlateau 状态 ──
lr_patience    = 5
lr_cooldown    = 2
lr_factor      = 0.5
lr_min_delta   = 1e-4
min_lr         = 1e-6
lr_plateau_cnt = 0
lr_cool_cnt    = 0
best_monitored_val_f1 = -1.0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── ReduceLROnPlateau (monitor val_macro_f1, mode='max') ──
        # 实际 LR 调整在 epoch 结束后进行，此处保持 current_lr

        # Shuffle
        X_shuf, y_shuf = shuffle(X_train_proc, y_train_aug,
                                 random_state=SEED + epoch)
        n_train = len(y_shuf)
        total_ce_loss = 0.0
        total_acc_val = 0.0
        n_batches = 0

        for offset in range(0, n_train, BATCH_SIZE):
            bx = X_shuf[offset:offset + BATCH_SIZE]
            by = y_shuf[offset:offset + BATCH_SIZE]
            feed = {x_ph: bx, y_ph: by, kp_ph: DROP_FC, kpc_ph: DROP_CONV,
                    lr_ph: current_lr, cw_ph: class_weights}
            _, bl, ba = sess.run([train_op, loss_op, accuracy_op],
                                 feed_dict=feed)
            total_ce_loss += bl
            total_acc_val += ba
            n_batches += 1

        avg_train_loss = total_ce_loss / n_batches
        avg_train_acc  = total_acc_val / n_batches

        # 验证
        val_loss, val_acc, val_f1, _, _ = evaluate(sess, X_val_proc, y_val_raw)

        # ── ReduceLROnPlateau: 根据 val_f1 决定是否降 LR ──
        if lr_cool_cnt > 0:
            lr_cool_cnt -= 1
        else:
            if val_f1 > best_monitored_val_f1 * (1.0 + lr_min_delta):
                best_monitored_val_f1 = val_f1
                lr_plateau_cnt = 0
            else:
                lr_plateau_cnt += 1
                if lr_plateau_cnt >= lr_patience and current_lr > min_lr:
                    old_lr = current_lr
                    current_lr = max(current_lr * lr_factor, min_lr)
                    lr_plateau_cnt = 0
                    lr_cool_cnt = lr_cooldown
                    print(f'  [LR] Reduced: {old_lr:.6f} -> {current_lr:.6f} '
                          f'(val_f1 plateaued at {val_f1:.4f})')

        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f'E{epoch:3d}/{EPOCHS}  '
            f'loss={avg_train_loss:.4f}  acc={avg_train_acc:.4f}  '
            f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  '
            f'val_f1={val_f1:.4f}  '
            f'lr={current_lr:.6f}  {elapsed:.0f}s'
        )

        # 保存 latest checkpoint
        _latest_path = os.path.join(CKPT_LATEST_DIR, 'tsrd_scratch_latest')
        _assert_in_run_dir(_latest_path, 'save latest checkpoint')
        latest_saver.save(sess, _latest_path,
                          global_step=epoch, write_meta_graph=False)

        # Early stopping with min_delta
        if val_f1 > best_val_f1 * (1.0 + EARLY_STOP_MIN_DELTA):
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience    = 0
            _best_path = os.path.join(CKPT_BEST_DIR, 'tsrd_scratch_best')
            _assert_in_run_dir(_best_path, 'save best checkpoint')
            best_ckpt = best_saver.save(sess, _best_path)
            print(f'  * New best val_f1={val_f1:.4f}, saved to {best_ckpt}')
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE and not args.smoke_test:
                print(f'\nEarly stop @ epoch {epoch}. Best epoch {best_epoch} '
                      f'(val_f1={best_val_f1:.4f})')
                break

    total_time = time.time() - train_start
    print(f'\nTraining done in {total_time:.0f}s ({total_time/60:.1f} min)')


# ═══════════════════════════════════════════════════════════
# 9. 最终评估 — 恢复到 best checkpoint
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Final Evaluation (best checkpoint)')
print('=' * 60)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    if best_ckpt is not None:
        try:
            best_saver.restore(sess, best_ckpt)
            print(f'Restored best checkpoint from epoch {best_epoch}')
        except Exception as e:
            print(f'Could not restore best checkpoint ({e}), using final model state.')
    else:
        print('Using final model state (no best checkpoint saved).')

    # ── Test evaluation ──
    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate(
        sess, X_test_proc, y_test)

    print(f'\n  Test Loss:      {test_loss:.4f}')
    print(f'  Test Accuracy:  {test_acc:.4f}')
    print(f'  Test Macro-F1:  {test_macro_f1:.4f}')

    # ── Per-class accuracy ──
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int32)
    class_total   = np.zeros(NUM_CLASSES, dtype=np.int32)
    for t, p in zip(test_labels, test_preds):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    class_acc = np.array([
        class_correct[i] / max(class_total[i], 1)
        for i in range(NUM_CLASSES)
    ])

    valid = [(i, class_acc[i]) for i in range(NUM_CLASSES) if class_total[i] > 0]
    valid.sort(key=lambda x: x[1])

    print('\n  Worst 5 classes (on test set):')
    for cid, acc in valid[:5]:
        n = class_total[cid]
        name = sign_names.get(cid, ('', ''))[0]
        print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

    print('\n  Best 5 classes (on test set):')
    for cid, acc in valid[-5:]:
        n = class_total[cid]
        name = sign_names.get(cid, ('', ''))[0]
        print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

    missing = [i for i in range(NUM_CLASSES) if class_total[i] == 0]
    if missing:
        print(f'\n  Test set missing classes (cannot evaluate): {missing}')


# ═══════════════════════════════════════════════════════════
# 10. 保存指标与可视化
# ═══════════════════════════════════════════════════════════
print('\n--- Saving metrics & visualizations ---')

try:
    # 10a. metrics.json
    metrics = {
        'phase': 'phase3',
        'best_epoch':        best_epoch,
        'best_val_f1':       float(best_val_f1),
        'test_loss':         float(test_loss),
        'test_accuracy':     float(test_acc),
        'test_macro_f1':     float(test_macro_f1),
        'total_train_sec':   float(total_time),
        'total_train_min':   total_time / 60.0,
        'n_train':           int(len(y_train_aug)),
        'n_val':             int(len(y_val_raw)),
        'n_test':            int(len(y_test)),
        'n_epochs_trained':  len(history['epoch']),
        'num_classes':       NUM_CLASSES,
        'batch_size':        BATCH_SIZE,
        'initial_lr':        INITIAL_LR,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'l2_beta':           L2_BETA,
        'dropout_fc':        DROP_FC,
        'dropout_conv':      DROP_CONV,
        'class_weight_mode': CLASS_WEIGHT_MODE,
        'class_weight_clip': MAX_WEIGHT_CLIP,
        'aug_target_max':    AUG_TARGET_MAX,
        'lr_schedule':       'ReduceLROnPlateau',
        'lr_factor':         lr_factor,
        'lr_patience':       lr_patience,
        'min_lr':            min_lr,
    }

    # 加入 Phase 2 对比
    metrics['phase2_test_accuracy'] = 0.9188
    metrics['phase2_test_macro_f1'] = 0.8629
    metrics['phase2_best_val_f1']   = 0.9989

    with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('  metrics.json saved')

    # 10b. Classification report
    all_target_names = [sign_names.get(i, (f'Class {i}', ''))[0]
                        for i in range(NUM_CLASSES)]
    present_labels = sorted(np.unique(test_labels))
    present_names = [all_target_names[i] for i in present_labels]
    report_str = classification_report(
        test_labels, test_preds, labels=present_labels,
        target_names=present_names, zero_division=0)

    with open(os.path.join(REPORTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write('TSRD Phase 3 — Classification Report\n')
        f.write('=' * 60 + '\n\n')
        f.write(report_str)
        f.write(f'\nTest Accuracy:  {test_acc:.4f}\n')
        f.write(f'Test Macro-F1:  {test_macro_f1:.4f}\n')
        f.write(f'Best Val F1:    {best_val_f1:.4f} (epoch {best_epoch})\n')
    print('  classification_report.txt saved')

    # 10c. Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=range(NUM_CLASSES))
    np.save(os.path.join(REPORTS_DIR, 'confusion_matrix.npy'), cm)

    fig, ax = plt.subplots(figsize=(22, 18))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('TSRD Phase 3 — Confusion Matrix')
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print('  confusion_matrix.png saved')

    # 10d. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Loss
    axes[0].plot(history['epoch'], history['train_loss'], 'b-', lw=1.5, label='Train')
    axes[0].plot(history['epoch'], history['val_loss'],   'r-', lw=1.5, label='Val')
    axes[0].axvline(best_epoch, color='gray', ls='--', alpha=0.5,
                    label=f'Best ep={best_epoch}')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    # Accuracy
    axes[1].plot(history['epoch'], history['train_acc'], 'b-', lw=1.5, label='Train')
    axes[1].plot(history['epoch'], history['val_acc'],   'r-', lw=1.5, label='Val')
    axes[1].axvline(best_epoch, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    # Macro-F1 + LR
    color_f1 = 'tab:green'
    color_lr = 'tab:orange'
    ax_f1 = axes[2]
    ax_f1.plot(history['epoch'], history['val_f1'], color=color_f1, lw=2,
               label='Val Macro-F1')
    ax_f1.axvline(best_epoch, color='gray', ls='--', alpha=0.5)
    ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('Macro-F1', color=color_f1)
    ax_f1.tick_params(axis='y', labelcolor=color_f1)
    ax_f1_lr = ax_f1.twinx()
    ax_f1_lr.plot(history['epoch'], history['lr'], color=color_lr, lw=1.5,
                  linestyle='--', label='LR', alpha=0.7)
    ax_f1_lr.set_ylabel('Learning Rate', color=color_lr)
    ax_f1_lr.tick_params(axis='y', labelcolor=color_lr)
    ax_f1_lr.set_yscale('log')
    lines1, labels1 = ax_f1.get_legend_handles_labels()
    lines2, labels2 = ax_f1_lr.get_legend_handles_labels()
    ax_f1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    ax_f1.grid(True, alpha=0.3)
    axes[2].set_title('Val F1 & LR Schedule')

    plt.suptitle('TSRD Phase 3 — Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print('  training_curves.png saved')

    # 10e. Training history
    np.save(os.path.join(REPORTS_DIR, 'training_history.npy'), history)
    print('  training_history.npy saved')

    # 10f. Config
    config_data = {
        'phase': 'phase3',
        'seed': SEED,
        'num_classes': NUM_CLASSES,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'initial_lr': INITIAL_LR,
        'dropout_fc': DROP_FC,
        'dropout_conv': DROP_CONV,
        'l2_beta': L2_BETA,
        'l2_layers': 'all',
        'class_weight_mode': CLASS_WEIGHT_MODE,
        'class_weight_clip': MAX_WEIGHT_CLIP,
        'aug_target_max': AUG_TARGET_MAX,
        'lr_schedule': 'ReduceLROnPlateau',
        'lr_factor': lr_factor,
        'lr_patience': lr_patience,
        'lr_cooldown': lr_cooldown,
        'lr_min_delta': lr_min_delta,
        'min_lr': min_lr,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'early_stop_monitor': 'val_macro_f1',
        'model_architecture': '5Conv+3FC',
        'weight_init': 'truncated_normal(std=0.05)',
        'weak_classes_boost': sorted(PHASE2_WEAK_CLASSES),
        'augmentation_base': 'tsrd_augment_single (rot±8, scale 0.85-1.15, trans±3, bright 0.7-1.3, blur 0.3)',
        'augmentation_added': 'gaussian_noise(0.02, 20%), shear(±2°, 20%)',
        'augmentation_weak': 'rot±10, trans±4, bright 0.6-1.4, blur 0.4, noise 0.3, shear 0.3',
    }
    with open(os.path.join(REPORTS_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print('  config.json saved')

except Exception as e:
    print(f'\n  [WARNING] Save section failed: {e}')
    print('  Training results may be incomplete.')


# ═══════════════════════════════════════════════════════════
# 11. Phase 3 Error Analysis Report
# ═══════════════════════════════════════════════════════════
print('\n--- Generating Phase 3 Error Analysis ---')
try:
    # Per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=range(NUM_CLASSES), zero_division=0)

    class_metrics = []
    for cid in range(NUM_CLASSES):
        if support[cid] > 0:
            class_metrics.append({
                'class_id': cid,
                'name': sign_names.get(cid, (f'Class {cid}', ''))[0],
                'support': int(support[cid]),
                'precision': prec[cid],
                'recall': rec[cid],
                'f1': f1[cid],
            })
    class_metrics.sort(key=lambda x: x['f1'])

    # Confused pairs
    confused_pairs = []
    for true_c in range(NUM_CLASSES):
        row_total = cm[true_c].sum()
        if row_total == 0:
            continue
        for pred_c in range(NUM_CLASSES):
            if true_c != pred_c and cm[true_c, pred_c] > 0:
                confused_pairs.append({
                    'true_class': true_c,
                    'true_name': sign_names.get(true_c, ('', ''))[0],
                    'pred_class': pred_c,
                    'pred_name': sign_names.get(pred_c, ('', ''))[0],
                    'count': int(cm[true_c, pred_c]),
                    'frac': cm[true_c, pred_c] / max(row_total, 1),
                })
    confused_pairs.sort(key=lambda x: -x['count'])

    error_report_path = os.path.join(REPORTS_DIR, 'phase3_error_analysis.txt')
    with open(error_report_path, 'w', encoding='utf-8') as f:
        f.write('TSRD Phase 3 — Error Analysis Report\n')
        f.write('=' * 65 + '\n\n')

        # Section 1: Worst 10 classes
        f.write('1. Worst 10 Classes (by test F1)\n')
        f.write('-' * 65 + '\n')
        f.write(f'  {"Class":>5s}  {"Name":<22s}  {"Prec":>6s}  '
                f'{"Rec":>6s}  {"F1":>6s}  {"Support":>8s}\n')
        f.write('  ' + '-' * 65 + '\n')
        for m in class_metrics[:10]:
            f.write(f'  {m["class_id"]:>5d}  {m["name"]:<22s}  '
                    f'{m["precision"]:>6.3f}  {m["recall"]:>6.3f}  '
                    f'{m["f1"]:>6.3f}  {m["support"]:>8d}\n')
        f.write('\n')

        # Section 2: Most confused pairs
        f.write('2. Top 15 Most Confused Class Pairs\n')
        f.write('-' * 65 + '\n')
        f.write(f'  {"True -> Predicted":<50s}  {"Count":>6s}  {"Frac":>6s}\n')
        f.write('  ' + '-' * 65 + '\n')
        for pair in confused_pairs[:15]:
            label = f'{pair["true_name"]} ({pair["true_class"]}) -> {pair["pred_name"]} ({pair["pred_class"]})'
            if len(label) > 49:
                label = label[:46] + '...'
            f.write(f'  {label:<50s}  {pair["count"]:>6d}  {pair["frac"]:>6.3f}\n')
        f.write('\n')

        # Section 3: Phase 2 weak classes deep-dive
        f.write('3. Phase 2 Weak Classes — Deep Dive\n')
        f.write('-' * 65 + '\n')
        for weak_cid in sorted(PHASE2_WEAK_CLASSES):
            weak_name = sign_names.get(weak_cid, ('', ''))[0]

            # Find Phase 3 metric for this class
            p3 = None
            for m in class_metrics:
                if m['class_id'] == weak_cid:
                    p3 = m
                    break

            if p3:
                f.write(f'\n  Class {weak_cid} ({weak_name}):\n')
                f.write(f'    F1:        {p3["f1"]:.3f}\n')
                f.write(f'    Precision: {p3["precision"]:.3f}\n')
                f.write(f'    Recall:    {p3["recall"]:.3f}\n')
                f.write(f'    Support:   {p3["support"]}\n')

                # Confusions for this class
                class_conf = [p for p in confused_pairs
                              if p['true_class'] == weak_cid][:5]
                if class_conf:
                    f.write(f'    Confused with:\n')
                    for p in class_conf:
                        f.write(f'      -> {p["pred_name"]} ({p["pred_class"]}): '
                                f'{p["count"]}/{p3["support"]} '
                                f'({p["frac"]*100:.1f}%)\n')
            else:
                f.write(f'\n  Class {weak_cid} ({weak_name}): '
                        f'no test samples — cannot evaluate\n')
        f.write('\n')

        # Section 4: Missing test classes
        f.write('4. Missing Test Classes (cannot evaluate generalization)\n')
        f.write('-' * 65 + '\n')
        missing_test = [i for i in range(NUM_CLASSES) if class_total[i] == 0]
        if missing_test:
            for cid in missing_test:
                name = sign_names.get(cid, ('', ''))[0]
                f.write(f'  Class {cid:3d} ({name})\n')
        else:
            f.write('  All 58 classes have test samples.\n')
        f.write('\n')

        # Section 5: Phase 3 results summary
        f.write('5. Results Summary\n')
        f.write('-' * 65 + '\n')
        f.write(f'  Test Accuracy:  {test_acc:.4f}\n')
        f.write(f'  Test Macro-F1:  {test_macro_f1:.4f}\n')
        f.write(f'  Best Val F1:    {best_val_f1:.4f} (epoch {best_epoch})\n')
        f.write(f'  Val/Test gap:   {best_val_f1 - test_macro_f1:.4f}\n')
        f.write(f'  Epochs trained: {len(history["epoch"])}\n')
        f.write(f'  Phase 2 Test Acc:    0.9188\n')
        f.write(f'  Phase 2 Test Macro-F1: 0.8629\n')
        delta_acc = test_acc - 0.9188
        delta_f1 = test_macro_f1 - 0.8629
        f.write(f'  Acc change:     {delta_acc:+.4f}\n')
        f.write(f'  F1 change:      {delta_f1:+.4f}\n')

    print(f'  phase3_error_analysis.txt saved to {error_report_path}')

    # ── 11b. class_f1_comparison.png (Phase 2 vs Phase 3 per-class F1) ──
    try:
        phase2_f1 = {}
        if os.path.exists(PHASE2_CLASS_REPORT_PATH):
            print('  Loading Phase 2 classification report for comparison...')
            # Parse Phase 2 report to extract per-class F1
            with open(PHASE2_CLASS_REPORT_PATH, 'r', encoding='utf-8') as pf:
                for line in pf:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            cid = int(parts[0])
                            f1_val = float(parts[4])
                            phase2_f1[cid] = f1_val
                        except ValueError:
                            pass

        cids = []
        p3_f1_vals = []
        p2_f1_vals = []

        for m in class_metrics:
            cid = m['class_id']
            cids.append(cid)
            p3_f1_vals.append(m['f1'])
            p2_f1_vals.append(phase2_f1.get(cid, 0.0))

        fig, ax = plt.subplots(figsize=(16, 7))
        x = np.arange(len(cids))
        width = 0.35

        has_p2 = any(v > 0 for v in p2_f1_vals)
        if has_p2:
            ax.bar(x - width/2, p3_f1_vals, width, label='Phase 3', alpha=0.85,
                   color='#5cb85c')
            ax.bar(x + width/2, p2_f1_vals, width, label='Phase 2', alpha=0.85,
                   color='#f0ad4e')
        else:
            ax.bar(x, p3_f1_vals, width, label='Phase 3', alpha=0.85,
                   color='#5cb85c')

        # Highlight weak classes
        weak_cids_plot = [i for i, c in enumerate(cids) if c in PHASE2_WEAK_CLASSES]
        for wi in weak_cids_plot:
            if has_p2:
                ax.get_children()[wi].set_color('#d9534f')
            else:
                ax.get_children()[wi].set_color('#d9534f')

        ax.set_xlabel('Class ID'); ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Comparison: Phase 2 vs Phase 3')
        ax.set_xticks(x)
        ax.set_xticklabels(cids, fontsize=7, rotation=90)
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add note about red bars = Phase 2 weak classes
        ax.text(0.99, 0.05, 'Red = Phase 2 weak classes',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'class_f1_comparison.png'), dpi=150)
        plt.close(fig)
        print('  class_f1_comparison.png saved')
    except Exception as e:
        print(f'  [WARNING] class_f1_comparison.png failed: {e}')

    # ── 11c. worst_classes.png ──
    try:
        worst = class_metrics[:10]
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(worst))
        colors_worst = ['#d9534f' if m['class_id'] in PHASE2_WEAK_CLASSES
                        else '#5bc0de' for m in worst]
        ax.barh(y_pos, [m['f1'] for m in worst], color=colors_worst, edgecolor='#333')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{m["class_id"]}: {m["name"]}' for m in worst], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('F1 Score')
        ax.set_title('Worst 10 Classes — Phase 3')
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate F1 values
        for i, m in enumerate(worst):
            ax.text(m['f1'] + 0.01, i, f'{m["f1"]:.3f}',
                    va='center', fontsize=8)

        ax.text(0.99, 0.02, 'Red = Phase 2 weak class',
                transform=ax.transAxes, ha='right', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'worst_classes.png'), dpi=150)
        plt.close(fig)
        print('  worst_classes.png saved')
    except Exception as e:
        print(f'  [WARNING] worst_classes.png failed: {e}')

except Exception as e:
    print(f'\n  [WARNING] Error analysis section failed: {e}')


# ═══════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Phase 3 Complete')
print('=' * 60)
print(f'  Run directory:    {OUTPUT_DIR}')
if best_ckpt:
    print(f'  Best checkpoint:  {best_ckpt}')
print(f'  Latest epochs:    {CKPT_LATEST_DIR}/')
print(f'  Best:             {CKPT_BEST_DIR}/')
print(f'  Metrics:          {REPORTS_DIR}/metrics.json')
print(f'  Report:           {REPORTS_DIR}/classification_report.txt')
print(f'  Error Analysis:   {REPORTS_DIR}/phase3_error_analysis.txt')
print(f'  Config:           {REPORTS_DIR}/config.json')
print(f'  Confusion:        {PLOTS_DIR}/confusion_matrix.png')
print(f'  Curves:           {PLOTS_DIR}/training_curves.png')
print(f'  F1 Comparison:    {PLOTS_DIR}/class_f1_comparison.png')
print(f'  Worst Classes:    {PLOTS_DIR}/worst_classes.png')
print(f'  Log:              {LOG_DIR}/')
print('=' * 60)

# Smoke test cleanup
if args.smoke_test:
    print('\n  [SMOKE TEST PASSED] Pipeline verification complete.')
    print(f'  All outputs generated in {OUTPUT_DIR}')
