"""
GTSRB 43 类交通标志名称映射
从 signnames.csv 读取，提供 class_id -> class_name 的查找能力
"""
import csv
import os

# 内置的 GTSRB 43 类名称（兜底，当 CSV 文件不可用时使用）
_BUILTIN_CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                class_id = int(row[0].strip())
                class_name = row[1].strip()
                class_names[class_id] = class_name

    if len(class_names) != 43:
        print(f"[WARN] Expected 43 classes, got {len(class_names)}. "
              f"Using built-in names as fallback.")
        return dict(_BUILTIN_CLASS_NAMES)

    return class_names


def get_class_name(class_id: int, class_names: dict) -> str:
    """根据 class_id 获取类别名称，未找到则返回 'Unknown'"""
    return class_names.get(class_id, "Unknown")
