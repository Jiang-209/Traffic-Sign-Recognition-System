"""
TSRD 数据集一键检查脚本

用法:
    python check_tsrd_dataset.py

功能:
    1. 加载训练集/测试集图片
    2. 统计类别分布
    3. 对比 train/test 覆盖情况
    4. 生成增强预览
    5. 验证 train/val 分层划分
    6. 输出报告和图表

所有输出保存到:
    tsrd_outputs/plots/      图表 (PNG)
    tsrd_outputs/reports/    报告 (TXT)
"""

import os
import sys
import numpy as np

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_TSRD = os.path.join(PROJECT_ROOT, 'data', 'TSRD')
sys.path.insert(0, DATA_TSRD)

from tsrd_loader import (
    load_tsrd_images,
    load_signnames_tsrd,
    analyze_class_distribution,
    compare_train_test_coverage,
    preview_augmentations,
    stratified_split,
    validate_dataset,
    TSRD_TRAIN_DIR,
    TSRD_TEST_DIR,
    SIGNNAMES_CSV,
)

OUTPUT_PLOTS = os.path.join(PROJECT_ROOT, 'tsrd_outputs', 'plots')
OUTPUT_REPORTS = os.path.join(PROJECT_ROOT, 'tsrd_outputs', 'reports')


def main():
    print('=' * 60)
    print('  TSRD 数据集完整性检查')
    print('=' * 60)
    print(f'  训练集路径: {TSRD_TRAIN_DIR}')
    print(f'  测试集路径: {TSRD_TEST_DIR}')
    print(f'  CSV 路径:   {SIGNNAMES_CSV}')
    print(f'  输出目录:   {OUTPUT_PLOTS}')
    print(f'  报告目录:   {OUTPUT_REPORTS}')

    os.makedirs(OUTPUT_PLOTS, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)

    # ---- 步骤 1: 完整验证 ----
    print(f'\n{"-"*60}')
    print('  步骤 1/5: 加载数据与完整验证')
    print(f'{"-"*60}')

    result = validate_dataset(
        train_dir=TSRD_TRAIN_DIR,
        test_dir=TSRD_TEST_DIR,
        csv_path=SIGNNAMES_CSV,
        output_plots_dir=OUTPUT_PLOTS,
        output_reports_dir=OUTPUT_REPORTS,
    )

    X_train, y_train = result['X_train'], result['y_train']
    X_test, y_test = result['X_test'], result['y_test']
    sign_names = result['sign_names']
    num_classes = result['num_classes']

    # ---- 步骤 2: 标签校验 ----
    print(f'\n{"-"*60}')
    print('  步骤 2/5: 标签校验')
    print(f'{"-"*60}')

    # 用 Annotation 文件验证文件名标签
    import csv

    def check_annotation_consistency(annotation_path, image_dir, split_name):
        errors = []
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                filename = parts[0]
                anno_label = int(parts[7]) if len(parts) > 7 else -1
                file_label = None
                try:
                    file_label = int(filename.split('_')[0])
                except ValueError:
                    errors.append(f'无法解析文件名标签: {filename}')
                    continue
                if anno_label != file_label:
                    errors.append(
                        f'{filename}: Annotation label={anno_label}, '
                        f'Filename label={file_label}'
                    )
        if errors:
            print(f'  [{split_name}] 发现 {len(errors)} 处不一致:')
            for e in errors[:10]:
                print(f'    {e}')
            if len(errors) > 10:
                print(f'    ... 共 {len(errors)} 处')
        else:
            print(f'  [{split_name}] Annotation 与文件名标签完全一致 (OK)')
        return errors

    train_anno = os.path.join(DATA_TSRD, 'TSRD-train', 'TsignRecgTrain4170Annotation.txt')
    test_anno = os.path.join(DATA_TSRD, 'TSRD-test', 'TsignRecgTest1994Annotation.txt')

    if os.path.exists(train_anno):
        check_annotation_consistency(train_anno, TSRD_TRAIN_DIR, '训练集')
    else:
        print('  训练集 Annotation 文件不存在，跳过校验')

    if os.path.exists(test_anno):
        check_annotation_consistency(test_anno, TSRD_TEST_DIR, '测试集')
    else:
        print('  测试集 Annotation 文件不存在，跳过校验')

    # ---- 步骤 3: 分层划分验证 ----
    print(f'\n{"-"*60}')
    print('  步骤 3/5: Train/Val 分层划分验证')
    print(f'{"-"*60}')

    X_tr, X_val, y_tr, y_val = stratified_split(
        X_train, y_train, test_size=0.2, seed=2018, sign_names=sign_names
    )

    # 检查划分后 val 集是否覆盖所有可能类别
    val_classes = set(np.unique(y_val))
    all_classes = set(range(num_classes))
    missing_in_val = all_classes - val_classes
    rare_in_val = {
        c: np.sum(y_val == c)
        for c in val_classes if np.sum(y_val == c) == 1
    }

    if missing_in_val:
        print(f'\n  [警告] 验证集缺失 {len(missing_in_val)} 个类别: {sorted(missing_in_val)}')
    if rare_in_val:
        print(f'  [警告] 验证集中有 {len(rare_in_val)} 个类别仅 1 个样本: '
              f'{sorted(rare_in_val.keys())}')

    print(f'  划分验证完成')

    # ---- 步骤 4: 针对少样本类别的增强比例建议 ----
    print(f'\n{"-"*60}')
    print('  步骤 4/5: 少样本类别增强建议')
    print(f'{"-"*60}')

    train_counter = {}
    for c in range(num_classes):
        train_counter[c] = np.sum(y_train == c)

    rare_threshold = 20
    rare_classes = [(c, cnt) for c, cnt in train_counter.items()
                    if cnt < rare_threshold and cnt > 0]
    rare_classes.sort(key=lambda x: x[1])

    if rare_classes:
        print(f'  以下 {len(rare_classes)} 个类别样本数 < {rare_threshold}，'
              f'训练时需要重点增强:')
        for cid, cnt in rare_classes:
            if cid in sign_names:
                name, stype = sign_names[cid]
                print(f'    class {cid:3d}: {cnt:3d} 张 -> {name} ({stype})')
        print(f'\n  建议增强倍数: 将少样本类别补到至少 200 张')
    else:
        print(f'  所有类别样本数 >= {rare_threshold}，无需特殊处理')

    # ---- 步骤 5: 生成总结 ----
    print(f'\n{"-"*60}')
    print('  步骤 5/5: 生成检查总结')
    print(f'{"-"*60}')

    summary_path = os.path.join(OUTPUT_REPORTS, 'check_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('TSRD 数据集检查总结\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'检查时间: {__import__("datetime").datetime.now()}\n\n')
        f.write(f'训练集: {len(y_train)} 张, {len(set(np.unique(y_train)))} '
                f'个类别\n')
        f.write(f'测试集: {len(y_test)} 张, {len(set(np.unique(y_test)))} '
                f'个类别\n')
        f.write(f'类别总数: {num_classes}\n')
        f.write(f'图片尺寸: 32x32x3 (RGB, resize后)\n\n')

        f.write('--- 测试集缺失类别 ---\n')
        train_only = set(np.unique(y_train)) - set(np.unique(y_test))
        for cid in sorted(train_only):
            if cid in sign_names:
                name, stype = sign_names[cid]
                f.write(f'  class {cid}: {name} ({stype})\n')
        f.write(f'\n共 {len(train_only)} 个类别无法在测试集评估。\n\n')

        f.write('--- 少样本类别 (< 20) ---\n')
        for cid, cnt in rare_classes:
            name, _ = sign_names.get(cid, ('Unknown', ''))
            f.write(f'  class {cid:3d}: {cnt:3d} -> {name}\n')
        f.write(f'\n共 {len(rare_classes)} 个少样本类别需要重点增强。\n\n')

        f.write('--- 输出文件清单 ---\n')
        f.write(f'  {os.path.join(OUTPUT_PLOTS, "class_distribution_train.png")}\n')
        f.write(f'  {os.path.join(OUTPUT_PLOTS, "class_distribution_test.png")}\n')
        f.write(f'  {os.path.join(OUTPUT_PLOTS, "train_test_coverage.png")}\n')
        f.write(f'  {os.path.join(OUTPUT_PLOTS, "augmentation_preview.png")}\n')
        f.write(f'  {os.path.join(OUTPUT_REPORTS, "dataset_report.txt")}\n')

    print(f'  总结已保存: {summary_path}')

    # ---- 完成 ----
    print(f'\n{"="*60}')
    print(f'  Phase 1 数据检查完成')
    print(f'{"="*60}')
    print(f'  输出图表: {OUTPUT_PLOTS}/')
    print(f'    - class_distribution_train.png')
    print(f'    - class_distribution_test.png')
    print(f'    - train_test_coverage.png')
    print(f'    - augmentation_preview.png')
    print(f'  输出报告: {OUTPUT_REPORTS}/')
    print(f'    - dataset_report.txt')
    print(f'    - check_summary.txt')

    return result


if __name__ == '__main__':
    main()
