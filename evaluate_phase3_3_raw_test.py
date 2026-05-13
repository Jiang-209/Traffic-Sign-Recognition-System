"""
evaluate_phase3_3_raw_test.py — Phase 3.3 RAW Test Evaluation

目的：测试 Phase 3.3 模型在不使用 ROI 裁剪的情况下，
直接对原始 TSRD 测试图片（全图 resize 到 32×32）的识别效果，
与 ROI-cropped 测试结果进行对比。

加载 Phase 3.3 已训练好的 checkpoint，不做任何重新训练。
"""
import os
import sys
import json
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── 路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_TSRD = os.path.join(PROJECT_ROOT, 'data', 'TSRD')
sys.path.insert(0, DATA_TSRD)

from tsrd_loader import (
    load_tsrd_images,
    load_signnames_tsrd,
    TSRD_TEST_DIR,
    SIGNNAMES_CSV,
    IMG_HEIGHT,
    IMG_WIDTH,
)

# Phase 3.3 最佳 checkpoint
PHASE3_3_RUN = 'scratch_phase3_3_20260514_024532'
CKPT_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', PHASE3_3_RUN, 'checkpoints', 'best')
CKPT_PATH = os.path.join(CKPT_DIR, 'tsrd_scratch_best')

# Phase 3.3 ROI-cropped test metrics 文件
PHASE3_3_METRICS = os.path.join(
    PROJECT_ROOT, 'tsrd_runs', PHASE3_3_RUN, 'reports', 'metrics.json')
PHASE3_3_CLASS_REPORT = os.path.join(
    PROJECT_ROOT, 'tsrd_runs', PHASE3_3_RUN, 'reports', 'classification_report.txt')
PHASE3_3_CM = os.path.join(
    PROJECT_ROOT, 'tsrd_runs', PHASE3_3_RUN, 'reports', 'confusion_matrix.npy')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'tsrd_runs', 'phase3_3_raw_eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 58

# ═══════════════════════════════════════════════════════════
# 1. 加载类别名称
# ═══════════════════════════════════════════════════════════
sign_names, _ = load_signnames_tsrd(SIGNNAMES_CSV)
print(f'Loaded {len(sign_names)} traffic sign classes.\n')

# ═══════════════════════════════════════════════════════════
# 2. 加载测试数据（原始全图，不做 ROI crop）
# ═══════════════════════════════════════════════════════════
print('Loading RAW test images (full-image, no ROI crop)...')
X_test, y_test = load_tsrd_images(TSRD_TEST_DIR)
print(f'  Test set: {X_test.shape[0]} images, shape={X_test.shape}, range=[{X_test.min()}, {X_test.max()}]\n')

# ═══════════════════════════════════════════════════════════
# 3. 预处理（与 Phase 3.3 训练完全一致）
# ═══════════════════════════════════════════════════════════
print('Preprocessing (RGB → Gray → [-0.5, 0.5])...')
def preprocess(X):
    n = X.shape[0]
    out = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for i in range(n):
        gray = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        out[i, :, :, 0] = gray / 255.0 - 0.5
    return out

X_test_proc = preprocess(X_test)
print(f'  Preprocessed: {X_test_proc.shape}\n')

# ═══════════════════════════════════════════════════════════
# 4. 加载 Phase 3.3 checkpoint 并推理
# ═══════════════════════════════════════════════════════════
meta_file = CKPT_PATH + '.meta'
if not os.path.exists(meta_file):
    raise FileNotFoundError(f'Checkpoint not found: {meta_file}')

print(f'Loading Phase 3.3 checkpoint from: {CKPT_PATH}')
tf.reset_default_graph()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, CKPT_PATH)

    graph = tf.get_default_graph()
    x_ph = graph.get_tensor_by_name('x:0')
    kp_ph = graph.get_tensor_by_name('keep_prob:0')
    kpc_ph = graph.get_tensor_by_name('keep_p_conv:0')
    logits = graph.get_tensor_by_name('logits:0')
    softmax_op = tf.nn.softmax(logits)

    print('\nRunning inference on RAW test images...')
    BATCH_SIZE = 64
    all_preds = []
    all_probas = []

    for offset in range(0, len(y_test), BATCH_SIZE):
        bx = X_test_proc[offset:offset + BATCH_SIZE]
        feed = {x_ph: bx, kp_ph: 1.0, kpc_ph: 1.0}
        proba, pred = sess.run([softmax_op, tf.argmax(logits, 1)], feed_dict=feed)
        all_preds.extend(pred)
        all_probas.extend(proba)

all_preds = np.array(all_preds)
all_probas = np.array(all_probas)

# ═══════════════════════════════════════════════════════════
# 5. 评估指标
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  RAW Test Results (no ROI crop)')
print('=' * 60)

raw_acc = accuracy_score(y_test, all_preds)
raw_f1 = f1_score(y_test, all_preds, average='macro', zero_division=0)

print(f'  Test Accuracy:  {raw_acc:.4f}')
print(f'  Test Macro-F1:  {raw_f1:.4f}')

# Per-class accuracy
class_correct = np.zeros(NUM_CLASSES, dtype=np.int32)
class_total = np.zeros(NUM_CLASSES, dtype=np.int32)
for t, p in zip(y_test, all_preds):
    class_total[t] += 1
    if t == p:
        class_correct[t] += 1

class_acc = np.array([
    class_correct[i] / max(class_total[i], 1)
    for i in range(NUM_CLASSES)
])

valid = [(i, class_acc[i]) for i in range(NUM_CLASSES) if class_total[i] > 0]
valid.sort(key=lambda x: x[1])

print('\n  Worst 5 classes (on RAW test):')
for cid, acc in valid[:5]:
    n = class_total[cid]
    name = sign_names.get(cid, ('', ''))[0]
    print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

print('\n  Best 5 classes (on RAW test):')
for cid, acc in valid[-5:]:
    n = class_total[cid]
    name = sign_names.get(cid, ('', ''))[0]
    print(f'    Class {cid:3d} ({name:<20s}): {acc:.4f} ({n} samples)')

# ═══════════════════════════════════════════════════════════
# 6. 加载 Phase 3.3 ROI 结果进行对比
# ═══════════════════════════════════════════════════════════
print('\n' + '-' * 60)
print('  ROI vs RAW Comparison')
print('-' * 60)

roi_metrics = json.load(open(PHASE3_3_METRICS))
roi_acc = roi_metrics['test_accuracy']
roi_f1 = roi_metrics['test_macro_f1']

print(f'  {"Metric":<25s} {"ROI-cropped":>12s} {"RAW (full)":>12s} {"Δ":>10s}')
print(f'  {"-"*59}')
print(f'  {"Test Accuracy":<25s} {roi_acc:>12.4f} {raw_acc:>12.4f} {raw_acc - roi_acc:>+10.4f}')
print(f'  {"Test Macro-F1":<25s} {roi_f1:>12.4f} {raw_f1:>12.4f} {raw_f1 - roi_f1:>+10.4f}')

# 加载 ROI 的 confusion matrix 做 per-class 对比
if os.path.exists(PHASE3_3_CM):
    roi_cm = np.load(PHASE3_3_CM)
    # Per-class accuracy comparison
    roi_class_acc = np.array([
        roi_cm[i, i] / max(roi_cm[i, :].sum(), 1)
        for i in range(NUM_CLASSES)
    ])
    raw_class_acc = class_acc

    print(f'\n  Per-class accuracy drop (worst 10):')
    drops = [(i, roi_class_acc[i] - raw_class_acc[i])
             for i in range(NUM_CLASSES) if class_total[i] > 0]
    drops.sort(key=lambda x: x[1])
    for cid, drop in drops[:10]:
        name = sign_names.get(cid, ('', ''))[0]
        n = class_total[cid]
        print(f'    Class {cid:3d} ({name:<20s}): '
              f'ROI={roi_class_acc[cid]:.4f} → RAW={raw_class_acc[cid]:.4f} '
              f'(Δ={drop:+.4f}, n={n})')

# ═══════════════════════════════════════════════════════════
# 7. 保存结果
# ═══════════════════════════════════════════════════════════
print('\n--- Saving results ---')

# 7a. metrics.json
metrics = {
    'phase': 'phase3_3_raw_eval',
    'roi_run': PHASE3_3_RUN,
    'test_accuracy_raw': float(raw_acc),
    'test_macro_f1_raw': float(raw_f1),
    'test_accuracy_roi': float(roi_acc),
    'test_macro_f1_roi': float(roi_f1),
    'acc_drop': float(raw_acc - roi_acc),
    'f1_drop': float(raw_f1 - roi_f1),
    'n_test': int(len(y_test)),
    'num_classes': NUM_CLASSES,
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print('  metrics.json saved')

# 7b. Classification report
all_target_names = [sign_names.get(i, (f'Class {i}', ''))[0]
                    for i in range(NUM_CLASSES)]
present_labels = sorted(np.unique(y_test))
present_names = [all_target_names[i] for i in present_labels]
report_str = classification_report(
    y_test, all_preds, labels=present_labels,
    target_names=present_names, zero_division=0)

with open(os.path.join(OUTPUT_DIR, 'classification_report_raw.txt'), 'w') as f:
    f.write('Phase 3.3 — RAW Test (no ROI crop)\n')
    f.write('=' * 60 + '\n\n')
    f.write(report_str)
    f.write(f'\nTest Accuracy:  {raw_acc:.4f}\n')
    f.write(f'Test Macro-F1:  {raw_f1:.4f}\n')
print('  classification_report_raw.txt saved')

# 7c. Confusion matrix
cm = confusion_matrix(y_test, all_preds, labels=range(NUM_CLASSES))
np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix_raw.npy'), cm)

fig, ax = plt.subplots(figsize=(22, 18))
im = ax.imshow(cm, cmap='Oranges', interpolation='nearest', aspect='auto')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Phase 3.3 — RAW Test Confusion (no ROI crop)')
plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_raw.png'), dpi=150)
plt.close(fig)
print('  confusion_matrix_raw.png saved')

# 7d. Per-class accuracy comparison chart
if os.path.exists(PHASE3_3_CM):
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(NUM_CLASSES)
    width = 0.35
    ax.bar(x - width/2, roi_class_acc, width, label='ROI-cropped',
           color='#5bc0de', edgecolor='#333', linewidth=0.3)
    ax.bar(x + width/2, raw_class_acc, width, label='RAW (full image)',
           color='#d9534f', edgecolor='#333', linewidth=0.3)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Accuracy')
    ax.set_title('Phase 3.3: ROI-cropped vs RAW Test Accuracy per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=5)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # 标注下降最大的类别
    drops_sorted = [(i, roi_class_acc[i] - raw_class_acc[i])
                    for i in range(NUM_CLASSES) if class_total[i] > 0]
    drops_sorted.sort(key=lambda x: x[1])
    for cid, drop in drops_sorted[:5]:
        ax.annotate(f'{cid}\n(Δ={drop:.2f})',
                    xy=(cid, raw_class_acc[cid]),
                    fontsize=7, color='red', ha='center',
                    xytext=(0, -20), textcoords='offset points')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'per_class_comparison.png'), dpi=150)
    plt.close(fig)
    print('  per_class_comparison.png saved')

# 7e. Confidence distribution comparison (optional)
raw_confidences = np.max(all_probas, axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(raw_confidences, bins=50, color='#d9534f', alpha=0.7, edgecolor='white')
ax.axvline(0.7, color='gray', linestyle='--', label='Threshold=0.7')
ax.set_xlabel('Confidence')
ax.set_ylabel('Count')
ax.set_title('Phase 3.3 — RAW Test Confidence Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'raw_confidence_dist.png'), dpi=150)
plt.close(fig)
print('  raw_confidence_dist.png saved')

print(f'\nAll outputs saved to: {OUTPUT_DIR}')

# ═══════════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Evaluation Complete')
print('=' * 60)
print(f'  ROI-cropped Test Accuracy:  {roi_acc:.4f}')
print(f'  RAW Test Accuracy:          {raw_acc:.4f}')
print(f'  Accuracy Drop:              {raw_acc - roi_acc:+.4f}')
print(f'  ROI-cropped Test Macro-F1:  {roi_f1:.4f}')
print(f'  RAW Test Macro-F1:          {raw_f1:.4f}')
print(f'  Macro-F1 Drop:              {raw_f1 - roi_f1:+.4f}')
print(f'\n  Output: {OUTPUT_DIR}')
print('=' * 60)
