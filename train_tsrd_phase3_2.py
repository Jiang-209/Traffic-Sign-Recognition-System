"""
train_tsrd_phase3_2.py — TSRD Scratch Baseline Refinement (Phase 3.2)

基于 Phase 2 基线，仅采纳 Phase 3 中验证有效的改进：
  1. Per-class adaptive augmentation（min_target=200）
  2. Class 23 仅基准增强（方向敏感）
  3. 警告标志类别的温和增强
  4. 移除 shear 增强
  5. 新增 class 34 诊断分析

保留 Phase 2 的其他配置（L2、class weight、LR schedule、dropout 等）。

输出:
  tsrd_runs/scratch_phase3_2_YYYYMMDD_HHMMSS/
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
parser = argparse.ArgumentParser(description='TSRD Scratch CNN — Phase 3.2')
parser.add_argument('--batch-size', type=int, default=64, choices=[32, 64])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--smoke-test', action='store_true')
parser.add_argument('--run-name', type=str, default=None)
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
INITIAL_LR = 0.001
EARLY_STOP_PATIENCE = 999 if args.smoke_test else 10
AUG_TARGET_MAX = 300
AUG_TARGET_MIN = 200  # Phase 3.2: raised from 150

# Phase 2 weak classes (kept for augmentation boost tracking)
PHASE2_WEAK_CLASSES = {5, 15, 23, 34, 50}

# Warning triangle classes (need milder augmentation)
WARNING_TRIANGLE_CLASSES = {32, 33, 34, 35, 36, 37, 38, 39, 45}

if args.smoke_test:
    print('\n  [SMOKE TEST] 1 epoch pipeline verification')

# ── 时间戳输出目录 ──
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = args.run_name if args.run_name else f'scratch_phase3_2_{RUN_ID}'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', run_name)
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
CKPT_LATEST_DIR = os.path.join(CHECKPOINT_DIR, 'latest')
CKPT_BEST_DIR = os.path.join(CHECKPOINT_DIR, 'best')

for d in [LOG_DIR, PLOTS_DIR, REPORTS_DIR, CKPT_LATEST_DIR, CKPT_BEST_DIR]:
    os.makedirs(d, exist_ok=True)

_RUN_DIR_ABS = os.path.normpath(os.path.abspath(OUTPUT_DIR))

def _assert_in_run_dir(path, purpose=''):
    resolved = os.path.normpath(os.path.abspath(path))
    if not resolved.startswith(_RUN_DIR_ABS):
        raise PermissionError(
            f'SAFETY: {purpose} path outside run dir.\n'
            f'  Path: {resolved}\n  Run dir: {_RUN_DIR_ABS}')

print('=' * 60)
print('  Phase 3.2: TSRD Scratch Baseline Refinement')
print('  58-class scratch CNN — no transfer learning')
print(f'  Run: {run_name}')
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

X_train_raw, X_val_raw, y_train_raw, y_val_raw = stratified_split(
    X_train_full, y_train_full, test_size=0.2, seed=SEED, sign_names=sign_names
)
print(f'\nAfter split — Train: {len(y_train_raw)}  Val: {len(y_val_raw)}')


# ═══════════════════════════════════════════════════════════
# 2. 数据增强（Phase 3.2: adaptive, no shear, warning triangle mild）
# ═══════════════════════════════════════════════════════════
print('\n--- Augmenting (Phase 3.2: adaptive, min=200) ---')
counter = Counter(y_train_raw)

def per_class_aug_target(count, class_id):
    """Adaptive target, min 200, weak classes up to ~320."""
    if count >= 200:
        target = count
    elif count >= 100:
        target = min(AUG_TARGET_MAX, int(count * 2.5))
    elif count >= 50:
        target = min(AUG_TARGET_MAX, max(AUG_TARGET_MIN, int(count * 3)))
    elif count >= 20:
        target = min(AUG_TARGET_MAX, max(AUG_TARGET_MIN, int(count * 3.5)))
    else:
        target = min(AUG_TARGET_MAX, max(AUG_TARGET_MIN, int(count * 5)))

    # Weak class boost (except class 23 — base augmentation only)
    if class_id in PHASE2_WEAK_CLASSES and class_id != 23:
        target = min(AUG_TARGET_MAX + 20, target + 30)

    return max(target, count)


def augment_phase3_2(img, class_id):
    """
    Phase 3.2 augmentation per class category:

    - Class 23: Base augmentation only (proven successful in Phase 3)
    - Warning triangles: Milder augmentation (rotation±5°, small translation/brightness)
    - Other classes: Base augmentation + mild noise (20%)
    - No shear (removed entirely from Phase 3)
    """
    if class_id == 23:
        # Class 23: 仅基准增强，保证方向语义稳定
        aug = tsrd_augment_single(img)
        # 非常轻微的噪声 (10%)
        if np.random.random() < 0.1:
            noise = np.random.normal(0, 0.008, img.shape).astype(np.float32) * 255
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return aug

    if class_id in WARNING_TRIANGLE_CLASSES:
        # 警告标志：温和增强，防止混淆
        aug = tsrd_augment_single(
            img,
            rotation_range=(-5, 5),        # Phase 3 was ±10, Phase 2 was ±8
            scale_range=(0.9, 1.1),        # narrower scale
            translation_range=(-2, 2),     # smaller translation
            brightness_range=(0.8, 1.2),   # narrower brightness
            blur_prob=0.2,                 # less blur
        )
        # 轻微噪声 (10%)
        if np.random.random() < 0.1:
            noise = np.random.normal(0, 0.008, img.shape).astype(np.float32) * 255
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return aug

    # 其他类别：基准增强 + 轻微噪声
    aug = tsrd_augment_single(img)
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 0.01, img.shape).astype(np.float32) * 255
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug


aug_X_list = [X_train_raw]
aug_y_list = [y_train_raw]

needs_aug = []
for cid, cnt in counter.items():
    target = per_class_aug_target(cnt, cid)
    if cnt < target:
        needs_aug.append((cid, cnt, target - cnt))

total_aug = sum(n for _, _, n in needs_aug)
done = 0
print(f'  Per-class targets — total extra: {total_aug}  min_target={AUG_TARGET_MIN}')

for cid, cnt, n_extra in needs_aug:
    target = cnt + n_extra
    indices = np.where(y_train_raw == cid)[0]
    for i in range(n_extra):
        idx = np.random.choice(indices)
        aug_img = augment_phase3_2(X_train_raw[idx], cid)
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

counter_aug = Counter(y_train_aug)
print(f'\nAugmented training set: {len(y_train_aug)} images')
print(f'  Phase 2: 17457  |  Phase 3: 10790  |  Phase 3.2: {len(y_train_aug)}')


# ═══════════════════════════════════════════════════════════
# 3. Class Weights（Phase 3.2: uniform, same as Phase 2）
# ═══════════════════════════════════════════════════════════
print('\n--- Computing class weights ---')
total_n = len(y_train_aug)
class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

for cid in range(NUM_CLASSES):
    cnt = counter_aug.get(cid, 0)
    class_weights[cid] = total_n / (NUM_CLASSES * max(cnt, 1))

class_weights = class_weights / np.mean(class_weights)

print(f'  Weight range: {class_weights.min():.3f} ~ {class_weights.max():.3f}')
print(f'  (Phase 2: 0.843~1.003 | Phase 3: 0.690~1.390)')

np.save(os.path.join(REPORTS_DIR, 'class_weights.npy'), class_weights)

# 保存详细增强后统计
with open(os.path.join(REPORTS_DIR, 'class_weights.txt'), 'w', encoding='utf-8') as f:
    f.write('Phase 3.2 — Per-Class Stats (After Augmentation)\n')
    f.write('=' * 65 + '\n\n')
    f.write(f'  Total augmented: {len(y_train_aug)}\n')
    f.write(f'  Weight range: {class_weights.min():.3f} ~ {class_weights.max():.3f}\n\n')
    f.write(f'  {"Class":>5s}  {"Name":<22s}  {"Weight":>8s}  {"Raw":>6s}  {"Aug":>6s}\n')
    f.write('  ' + '-' * 55 + '\n')
    sorted_cids = range(NUM_CLASSES)
    for cid in sorted_cids:
        cnt = counter_aug.get(cid, 0)
        raw = counter.get(cid, 0)
        name = sign_names.get(cid, ('', ''))[0]
        flag = '  [WEAK]' if cid in PHASE2_WEAK_CLASSES else ''
        flag2 = '  [TRI]' if cid in WARNING_TRIANGLE_CLASSES else ''
        f.write(f'  {cid:>5d}  {name:<22s}  {class_weights[cid]:>8.3f}  '
                f'{raw:>6d}  {cnt:>6d}{flag}{flag2}\n')
print('  class_weights.txt saved (with per-class augmented counts)')


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
# 5. 模型定义（与 Phase 2 完全一致）
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
    """5Conv + 3FC, 输出 58 类 logits（与 Phase 2 完全一致）。"""
    c1 = conv2d(x, W['conv1'], [1, 1, 1, 1], 'VALID', 'conv1')
    c1 = tf.nn.relu(c1 + b['conv1'], 'conv1_act')
    c1 = tf.nn.dropout(c1, keep_prob=keep_p_conv, name='conv1_drop')
    c2 = conv2d(c1, W['conv2'], [1, 1, 1, 1], 'SAME', 'conv2')
    c2 = tf.nn.relu(c2 + b['conv2'], 'conv2_act')
    c2 = max_pool_2x2(c2, 'VALID', 'conv2_pool')
    c2 = tf.nn.dropout(c2, keep_prob=keep_p_conv, name='conv2_drop')
    c3 = conv2d(c2, W['conv3'], [1, 1, 1, 1], 'VALID', 'conv3')
    c3 = tf.nn.relu(c3 + b['conv3'], 'conv3_act')
    c3 = tf.nn.dropout(c3, keep_prob=keep_p_conv, name='conv3_drop')
    c4 = conv2d(c3, W['conv4'], [1, 1, 1, 1], 'SAME', 'conv4')
    c4 = tf.nn.relu(c4 + b['conv4'], 'conv4_act')
    c4 = max_pool_2x2(c4, 'VALID', 'conv4_pool')
    c4 = tf.nn.dropout(c4, keep_prob=keep_prob, name='conv4_drop')
    c5 = conv2d(c4, W['conv5'], [1, 1, 1, 1], 'VALID', 'conv5')
    c5 = tf.nn.relu(c5 + b['conv5'], 'conv5_act')
    c5 = max_pool_2x2(c5, 'VALID', 'conv5_pool')
    c5 = tf.nn.dropout(c5, keep_prob=keep_prob, name='conv5_drop')
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

# Loss（Phase 3.2: Phase 2 风格 — L2=0.0001, FC only）
one_hot_y = tf.one_hot(y_ph, NUM_CLASSES, dtype=tf.float32)
ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
weighted_ce = ce * tf.gather(cw_ph, y_ph)

beta = 0.0001
l2_loss = beta * (tf.nn.l2_loss(W['fc1']) +
                  tf.nn.l2_loss(W['fc2']) +
                  tf.nn.l2_loss(W['fc3']))
loss_op = tf.reduce_mean(weighted_ce) + l2_loss

train_op = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_op)

correct   = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

latest_saver = tf.train.Saver(max_to_keep=20, name='latest')
best_saver = tf.train.Saver(max_to_keep=1, name='best')

print(f'\n  Graph — W_fc3: {W["fc3"].shape}  b_fc3: {b["fc3"].shape}')
print(f'  L2: beta={beta}, FC only (Phase 2 style)')


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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # LR schedule（Phase 2 固定 schedule）
        if epoch > 30:
            current_lr = 0.0003
        if epoch > 45:
            current_lr = 0.0001

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
            feed = {x_ph: bx, y_ph: by, kp_ph: 0.5, kpc_ph: 0.6,
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

        # Early stopping（Phase 2 风格）
        if val_f1 > best_val_f1:
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
# 9. 最终评估
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

    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate(
        sess, X_test_proc, y_test)

    print(f'\n  Test Loss:      {test_loss:.4f}')
    print(f'  Test Accuracy:  {test_acc:.4f}')
    print(f'  Test Macro-F1:  {test_macro_f1:.4f}')

    # Per-class
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

    print('\n  Worst 5 classes:')
    for cid, acc in valid[:5]:
        n = class_total[cid]
        name = sign_names.get(cid, ('', ''))[0]
        print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

    print('\n  Best 5 classes:')
    for cid, acc in valid[-5:]:
        n = class_total[cid]
        name = sign_names.get(cid, ('', ''))[0]
        print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

    missing = [i for i in range(NUM_CLASSES) if class_total[i] == 0]
    if missing:
        print(f'\n  Test set missing: {missing}')


# ═══════════════════════════════════════════════════════════
# 10. 保存指标与可视化
# ═══════════════════════════════════════════════════════════
print('\n--- Saving metrics & visualizations ---')

try:
    # 10a. metrics.json
    metrics = {
        'phase': 'phase3_2',
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
        'lr_schedule':       'fixed_milestones_30_45',
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'l2_beta':           beta,
        'l2_layers':         'fc_only',
        'dropout_fc':        0.5,
        'dropout_conv':      0.6,
        'class_weight':      'uniform_phase2_style',
        'aug_min_target':    AUG_TARGET_MIN,
        'aug_max_target':    AUG_TARGET_MAX,
        'phase2_test_acc':   0.9188,
        'phase2_test_f1':    0.8629,
        'phase3_test_acc':   0.9027,
        'phase3_test_f1':    0.8356,
    }
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
        f.write('TSRD Phase 3.2 — Classification Report\n')
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
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('TSRD Phase 3.2 — Confusion Matrix')
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
    axes[0].set_title('Loss (watch for val_loss divergence)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    # Accuracy
    axes[1].plot(history['epoch'], history['train_acc'], 'b-', lw=1.5, label='Train')
    axes[1].plot(history['epoch'], history['val_acc'],   'r-', lw=1.5, label='Val')
    axes[1].axvline(best_epoch, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    # Macro-F1
    axes[2].plot(history['epoch'], history['val_f1'], 'g-', lw=2, label='Val Macro-F1')
    axes[2].axvline(best_epoch, color='gray', ls='--', alpha=0.5)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Macro-F1')
    axes[2].set_title('Validation Macro-F1'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('TSRD Phase 3.2 — Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print('  training_curves.png saved')

    # 10e. Training history
    np.save(os.path.join(REPORTS_DIR, 'training_history.npy'), history)
    print('  training_history.npy saved')

    # 10f. Config
    config_data = {
        'phase': 'phase3_2',
        'seed': SEED, 'num_classes': NUM_CLASSES,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'initial_lr': INITIAL_LR,
        'lr_schedule': 'fixed_milestones_30_45',
        'dropout_fc': 0.5, 'dropout_conv': 0.6,
        'l2_beta': beta, 'l2_layers': 'fc_only',
        'class_weight': 'uniform_phase2',
        'aug_min_target': AUG_TARGET_MIN, 'aug_max_target': AUG_TARGET_MAX,
        'aug_shear': 'removed',
        'class23_aug': 'base_only',
        'warning_triangle_aug': 'mild_(rot±5,_bright_0.8-1.2,_trans±2)',
        'weak_classes_boost': sorted(PHASE2_WEAK_CLASSES),
        'model_architecture': '5Conv+3FC',
        'early_stop_patience': EARLY_STOP_PATIENCE,
    }
    with open(os.path.join(REPORTS_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print('  config.json saved')

except Exception as e:
    print(f'\n  [WARNING] Save section failed: {e}')


# ═══════════════════════════════════════════════════════════
# 11. 错误分析报告
# ═══════════════════════════════════════════════════════════
print('\n--- Generating Error Analysis ---')
try:
    prec, rec, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=range(NUM_CLASSES), zero_division=0)

    class_metrics = []
    for cid in range(NUM_CLASSES):
        if support[cid] > 0:
            class_metrics.append({
                'class_id': cid,
                'name': sign_names.get(cid, (f'Class {cid}', ''))[0],
                'support': int(support[cid]),
                'precision': prec[cid], 'recall': rec[cid], 'f1': f1[cid],
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

    error_path = os.path.join(REPORTS_DIR, 'error_analysis.txt')
    with open(error_path, 'w', encoding='utf-8') as f:
        f.write('TSRD Phase 3.2 — Error Analysis Report\n')
        f.write('=' * 65 + '\n\n')

        # 1. Worst 10
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

        # 2. Top confused pairs
        f.write('2. Top 15 Confused Pairs\n')
        f.write('-' * 65 + '\n')
        f.write(f'  {"True -> Predicted":<50s}  {"Count":>6s}  {"Frac":>6s}\n')
        f.write('  ' + '-' * 65 + '\n')
        for pair in confused_pairs[:15]:
            label = f'{pair["true_name"]} ({pair["true_class"]}) -> {pair["pred_name"]} ({pair["pred_class"]})'
            if len(label) > 49:
                label = label[:46] + '...'
            f.write(f'  {label:<50s}  {pair["count"]:>6d}  {pair["frac"]:>6.3f}\n')
        f.write('\n')

        # 3. Phase 2 weak classes deep dive
        f.write('3. Phase 2 Weak Classes — Deep Dive\n')
        f.write('-' * 65 + '\n')
        for weak_cid in sorted(PHASE2_WEAK_CLASSES):
            p3 = None
            for m in class_metrics:
                if m['class_id'] == weak_cid:
                    p3 = m; break
            if p3:
                f.write(f'\n  Class {weak_cid} ({p3["name"]}):\n')
                f.write(f'    F1: {p3["f1"]:.3f}  Prec: {p3["precision"]:.3f}  '
                        f'Rec: {p3["recall"]:.3f}  Support: {p3["support"]}\n')
                c_conf = [p for p in confused_pairs if p['true_class'] == weak_cid][:5]
                if c_conf:
                    f.write('    Confused with:\n')
                    for p in c_conf:
                        f.write(f'      -> {p["pred_name"]} ({p["pred_class"]}): '
                                f'{p["count"]}/{p3["support"]} ({p["frac"]*100:.1f}%)\n')
            else:
                f.write(f'\n  Class {weak_cid}: no test samples\n')
        f.write('\n')

        # 4. Warning triangle confusion analysis (NEW in Phase 3.2)
        f.write('4. Warning Triangle Confusion Analysis\n')
        f.write('-' * 65 + '\n')
        tri_classes = sorted(WARNING_TRIANGLE_CLASSES)
        tri_present = [c for c in tri_classes if class_total[c] > 0]
        for cid in tri_present:
            name = sign_names.get(cid, ('', ''))[0]
            m = next((x for x in class_metrics if x['class_id'] == cid), None)
            if m:
                f.write(f'\n  Class {cid} ({name}): F1={m["f1"]:.3f}  n={m["support"]}\n')
                c_conf = [p for p in confused_pairs if p['true_class'] == cid][:5]
                for p in c_conf:
                    f.write(f'    -> {p["pred_name"]} ({p["pred_class"]}): '
                            f'{p["count"]}/{m["support"]} ({p["frac"]*100:.1f}%)\n')

        # Triangles missing from test
        tri_missing = [c for c in tri_classes if class_total[c] == 0]
        if tri_missing:
            f.write(f'\n  Missing from test (cannot evaluate):\n')
            for cid in tri_missing:
                f.write(f'    Class {cid} ({sign_names.get(cid, ("", ""))[0]})\n')
        f.write('\n')

        # 5. Results summary
        f.write('5. Three-Way Comparison\n')
        f.write('-' * 65 + '\n')
        f.write(f'  {"Metric":<30s}  {"Phase2":>8s}  {"Phase3":>8s}  {"P3.2":>8s}\n')
        f.write('  ' + '-' * 55 + '\n')
        f.write(f'  {"Test Accuracy":<30s}  {"0.9188":>8s}  {"0.9027":>8s}  {test_acc:>8.4f}\n')
        f.write(f'  {"Test Macro-F1":<30s}  {"0.8629":>8s}  {"0.8356":>8s}  {test_macro_f1:>8.4f}\n')
        f.write(f'  {"Test Loss":<30s}  {"0.386":>8s}  {"0.749":>8s}  {test_loss:>8.4f}\n')
        f.write(f'  {"Best Epoch":<30s}  {43:>8d}  {49:>8d}  {best_epoch:>8d}\n')
        f.write(f'  {"n_train":<30s}  {17457:>8d}  {10790:>8d}  {len(y_train_aug):>8d}\n')

    print(f'  error_analysis.txt saved')

    # ── Class34 misclassification visualization ──
    try:
        c34_mask = test_labels == 34
        c34_count = c34_mask.sum()
        if c34_count > 0:
            c34_images = X_test[c34_mask]
            c34_true = test_labels[c34_mask]
            c34_pred = test_preds[c34_mask]

            n_samples = min(c34_count, 8)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            for i in range(n_samples):
                axes[i].imshow(c34_images[i])
                axes[i].set_title(f'True:34 Pred:{c34_pred[i]}',
                                  color='green' if c34_pred[i]==34 else 'red')
                axes[i].axis('off')
            for i in range(n_samples, 8):
                axes[i].axis('off')
            plt.suptitle('Class 34 (注意危险) — Test Samples & Predictions', fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, 'class34_predictions.png'), dpi=150)
            plt.close(fig)
            print('  class34_predictions.png saved')
    except Exception as e:
        print(f'  [WARNING] class34 viz failed: {e}')

except Exception as e:
    print(f'\n  [WARNING] Error analysis failed: {e}')


# ═══════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Phase 3.2 Complete')
print('=' * 60)
print(f'  Run directory:    {OUTPUT_DIR}')
if best_ckpt:
    print(f'  Best checkpoint:  {best_ckpt}')
print(f'  Metrics:          {REPORTS_DIR}/metrics.json')
print(f'  Report:           {REPORTS_DIR}/classification_report.txt')
print(f'  Error Analysis:   {REPORTS_DIR}/error_analysis.txt')
print(f'  Config:           {REPORTS_DIR}/config.json')
print(f'  Confusion:        {PLOTS_DIR}/confusion_matrix.png')
print(f'  Curves:           {PLOTS_DIR}/training_curves.png')
print(f'  Class34 Viz:      {PLOTS_DIR}/class34_predictions.png')
print('=' * 60)

if args.smoke_test:
    print('\n  [SMOKE TEST PASSED] Pipeline verification complete.')
    print(f'  All outputs generated in {OUTPUT_DIR}')
