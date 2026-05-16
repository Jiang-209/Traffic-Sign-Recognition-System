"""
TSRD 58 类交通标志名称映射
从 signnames_tsrd.csv 读取
"""
import csv
import os

# 内置的 TSRD 58 类名称（兜底，当 CSV 文件不可用时使用）
_BUILTIN_CLASS_NAMES = {
    0: "限制速度5km/h",
    1: "限制速度15km/h",
    2: "限制速度30km/h",
    3: "限制速度40km/h",
    4: "限制速度50km/h",
    5: "限制速度60km/h",
    6: "限制速度70km/h",
    7: "限制速度80km/h",
    8: "禁止直行和向左转弯",
    9: "禁止直行和向右转弯",
    10: "禁止直行",
    11: "禁止向左转弯",
    12: "禁止向左向右转弯",
    13: "禁止向右转弯",
    14: "禁止超车",
    15: "禁止掉头",
    16: "禁止机动车驶入",
    17: "禁止鸣喇叭",
    18: "解除限制速度40km/h",
    19: "解除限制速度50km/h",
    20: "直行和向右转弯",
    21: "直行",
    22: "向左转弯",
    23: "向左和向右转弯",
    24: "向右转弯",
    25: "靠左侧车道行驶",
    26: "靠右侧车道行驶",
    27: "环岛行驶",
    28: "机动车行驶",
    29: "鸣喇叭",
    30: "非机动车行驶",
    31: "允许掉头",
    32: "左右绕行",
    33: "注意信号灯",
    34: "注意危险",
    35: "注意行人",
    36: "注意非机动车",
    37: "注意儿童",
    38: "向右急弯路",
    39: "向左急弯路",
    40: "下陡坡",
    41: "上陡坡",
    42: "减速慢行",
    43: "右侧T形交叉",
    44: "左侧T形交叉",
    45: "村庄",
    46: "反向弯路",
    47: "无人看守铁路道口",
    48: "施工",
    49: "连续弯路",
    50: "有人看守铁路道口",
    51: "事故易发路段",
    52: "停车让行",
    53: "禁止通行",
    54: "禁止停车",
    55: "禁止驶入",
    56: "减速让行",
    57: "停车检查",
}


def load_class_names(csv_path: str) -> dict:
    """
    从 CSV 文件加载类别名称映射。
    CSV 格式：ClassId,SignName
    返回 dict: {class_id(int): class_name(str)}
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Class names CSV not found at '{csv_path}', "
              f"using built-in names.")
        return dict(_BUILTIN_CLASS_NAMES)

    class_names = {}
    with open(csv_path, "r", encoding="gbk") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                class_id = int(row[0].strip())
                class_name = row[1].strip()
                class_names[class_id] = class_name

    if len(class_names) != 58:
        print(f"[WARN] Expected 58 classes, got {len(class_names)}. "
              f"Using built-in names as fallback.")
        return dict(_BUILTIN_CLASS_NAMES)

    return class_names


def get_class_name(class_id: int, class_names: dict) -> str:
    """根据 class_id 获取类别名称，未找到则返回 'Unknown'"""
    return class_names.get(class_id, "Unknown")