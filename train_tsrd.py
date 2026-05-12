"""
train_tsrd.py - TSRD Scratch Baseline Training (Phase 2)

从零训练 58 类中国交通标志分类器。
无迁移学习，无预训练权重，无 checkpoint restore。

完整训练流程：数据加载 → 增强 → 训练 → 验证 → 测试 → 指标输出。

输出:
    tsrd_runs/scratch_YYYYMMDD_HHMMSS/
    ├── logs/
    ├── plots/
    ├── reports/
    └── checkpoints/
        ├── latest/
        └── best/
"""

import os
import sys
import time
import json
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
                             classification_report, confusion_matrix)

# ── 添加数据模块到路径 ──
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
IMG_DEPTH = 1          # grayscale
EPOCHS = 50
BATCH_SIZE = 64
INITIAL_LR = 0.001
EARLY_STOP_PATIENCE = 10
AUGMENT_TARGET = 300   # 少数类增强目标样本数

SMOKE_TEST = False     # True 时只跑 1 epoch 验证 pipeline

# ── 时间戳输出目录（每次运行独立，永不覆盖） ──
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR_NAME = f'scratch_{RUN_ID}'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', RUN_DIR_NAME)
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
CKPT_LATEST_DIR = os.path.join(CHECKPOINT_DIR, 'latest')
CKPT_BEST_DIR = os.path.join(CHECKPOINT_DIR, 'best')

for d in [LOG_DIR, PLOTS_DIR, REPORTS_DIR, CKPT_LATEST_DIR, CKPT_BEST_DIR]:
    os.makedirs(d, exist_ok=True)

if SMOKE_TEST:
    EPOCHS = 1
    EARLY_STOP_PATIENCE = 999
    print('\n  [SMOKE TEST] Running 1 epoch for verification')

# ── 路径安全检查：防止误写入旧目录 ──
_RUN_DIR_ABS = os.path.normpath(os.path.abspath(OUTPUT_DIR))

def _assert_in_run_dir(path, purpose=''):
    resolved = os.path.normpath(os.path.abspath(path))
    if not resolved.startswith(_RUN_DIR_ABS):
        raise PermissionError(
            f'SAFETY: {purpose} path is outside run dir.\n'
            f'  Path: {resolved}\n'
            f'  Run dir: {_RUN_DIR_ABS}')

print('=' * 60)
print('  Phase 2: TSRD Scratch Baseline Training')
print('  58-class scratch CNN — no transfer learning')
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
# 2. 数据增强（仅训练集）
# ═══════════════════════════════════════════════════════════
print('\n--- Augmenting minority classes ---')
counter = Counter(y_train_raw)

aug_X_list = [X_train_raw]
aug_y_list = [y_train_raw]

needs_aug = [(cid, cnt, AUGMENT_TARGET - cnt)
             for cid, cnt in counter.items()
             if 0 < cnt < AUGMENT_TARGET]
total_aug = sum(n for _, _, n in needs_aug)
done = 0

for cid, cnt, n_extra in needs_aug:
    indices = np.where(y_train_raw == cid)[0]
    for i in range(n_extra):
        idx = np.random.choice(indices)
        aug_X_list.append(tsrd_augment_single(X_train_raw[idx])[np.newaxis, ...])
        aug_y_list.append(np.array([cid], dtype=np.int32))
        done += 1
        if done % 500 == 0 or done == total_aug:
            print(f'  Augmenting... {done}/{total_aug}', flush=True)
    name = sign_names.get(cid, ('', ''))[0]
    print(f'  Class {cid:3d} ({name:<20s}): {cnt:4d} → {AUGMENT_TARGET:4d} (+{n_extra})', flush=True)

X_train_aug = np.concatenate(aug_X_list, axis=0).astype(np.uint8)
y_train_aug = np.concatenate(aug_y_list, axis=0).astype(np.int32)

X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=SEED)
print(f'\nAugmented training set: {len(y_train_aug)} images')


# ═══════════════════════════════════════════════════════════
# 3. Class Weights
# ═══════════════════════════════════════════════════════════
print('\n--- Computing class weights ---')
counter_aug = Counter(y_train_aug)
total_aug = len(y_train_aug)
class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

for cid in range(NUM_CLASSES):
    cnt = counter_aug.get(cid, 0)
    class_weights[cid] = total_aug / (NUM_CLASSES * max(cnt, 1))

# 归一化使权重均值为 1
class_weights = class_weights / np.mean(class_weights)

print(f'  Weight range: {class_weights.min():.3f} ~ {class_weights.max():.3f}')
for cid in np.argsort(-class_weights)[:5]:
    cnt = counter_aug.get(cid, 0)
    name = sign_names.get(cid, ('Unknown', ''))[0]
    print(f'    Class {cid:3d} (w={class_weights[cid]:.3f}): {cnt:5d} samples — {name}')

np.save(os.path.join(REPORTS_DIR, 'class_weights.npy'), class_weights)


# ═══════════════════════════════════════════════════════════
# 4. 预处理（RGB → 灰度 → [-0.5, 0.5]）
# ═══════════════════════════════════════════════════════════
def preprocess(X):
    """RGB uint8 → grayscale float32, normalized to [-0.5, 0.5]."""
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
# 5. 模型定义（与原始 CNN 架构一致，输出层 58 类）
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


# 重置图，避免 graph pollution
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
    """Conv1–5 + FC1–3, 输出 58 类 logits."""
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

# Loss: 带 class weight 的交叉熵 + L2 正则
one_hot_y = tf.one_hot(y_ph, NUM_CLASSES, dtype=tf.float32)
ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
weighted_ce = ce * tf.gather(cw_ph, y_ph)

beta = 0.0001
l2_loss = beta * (tf.nn.l2_loss(W['fc1']) +
                  tf.nn.l2_loss(W['fc2']) +
                  tf.nn.l2_loss(W['fc3']))
loss_op = tf.reduce_mean(weighted_ce) + l2_loss

# Optimizer
train_op = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_op)

# Accuracy
correct   = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

# Savers — latest 保留最近 20 轮，best 只保留最优
latest_saver = tf.train.Saver(max_to_keep=20, name='latest')
best_saver = tf.train.Saver(max_to_keep=1, name='best')

# Shape sanity check
print(f'\n  Graph — W_fc3: {W["fc3"].shape}  b_fc3: {b["fc3"].shape}')
print(f'  one_hot: (None, {NUM_CLASSES})  logits: (None, {NUM_CLASSES})')


# ═══════════════════════════════════════════════════════════
# 7. 评估函数
# ═══════════════════════════════════════════════════════════
def evaluate(sess, X_data, y_data):
    """
    在全量数据上计算 loss / accuracy / macro-F1 / 预测 / 标签。
    使用 uniform class weight（1.0）进行评估。
    """
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

        # LR schedule
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

        # 保存 latest checkpoint（保留最近 20 轮）
        _latest_path = os.path.join(CKPT_LATEST_DIR, 'tsrd_scratch_latest')
        _assert_in_run_dir(_latest_path, 'save latest checkpoint')
        latest_saver.save(sess, _latest_path,
                          global_step=epoch, write_meta_graph=False)

        # Early stopping: monitor val macro-F1
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
            if patience >= EARLY_STOP_PATIENCE:
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

    # 有效类别（测试集中有样本的）
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

    # ── 测试集缺失类别 ──
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
        f.write('TSRD Scratch Baseline — Classification Report\n')
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
    ax.set_title('TSRD Confusion Matrix — Scratch Baseline')
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

    # Macro-F1
    axes[2].plot(history['epoch'], history['val_f1'], 'g-', lw=2, label='Val Macro-F1')
    axes[2].axvline(best_epoch, color='gray', ls='--', alpha=0.5)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Macro-F1')
    axes[2].set_title('Validation Macro-F1'); axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('TSRD Scratch Baseline — Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print('  training_curves.png saved')

    # 10e. Training history (numpy)
    np.save(os.path.join(REPORTS_DIR, 'training_history.npy'), history)
    print('  training_history.npy saved')

except Exception as e:
    print(f'\n  [WARNING] Save section failed: {e}')
    print('  Training results may be incomplete.')


# ═══════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Phase 2 Complete')
print('=' * 60)
print(f'  Run directory:    {OUTPUT_DIR}')
if best_ckpt:
    print(f'  Best checkpoint:  {best_ckpt}')
print(f'  Latest epochs:    {CKPT_LATEST_DIR}/')
print(f'  Best:             {CKPT_BEST_DIR}/')
print(f'  Metrics:          {REPORTS_DIR}/metrics.json')
print(f'  Report:           {REPORTS_DIR}/classification_report.txt')
print(f'  Confusion:        {PLOTS_DIR}/confusion_matrix.png')
print(f'  Curves:           {PLOTS_DIR}/training_curves.png')
print(f'  Log:              {LOG_DIR}/')
print('=' * 60)
