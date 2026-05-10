"""
TensorFlow 模型加载与推理模块
支持 TF1 checkpoint 格式（.meta / .index / .data-*）
"""
import os
import numpy as np

# 使用 TF1 兼容模式，与训练代码保持一致
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ModelLoader:
    """TensorFlow 模型加载器，负责加载 checkpoint 并执行推理"""

    def __init__(self, model_path: str, img_height: int = 32,
                 img_width: int = 32, img_channels: int = 1):
        """
        参数:
            model_path: 模型文件路径（不含扩展名）
            img_height: 输入图像高度，默认 32
            img_width: 输入图像宽度，默认 32
            img_channels: 输入图像通道，默认 1（灰度）
        """
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        self.sess = None       # TF Session
        self.graph = None      # TF Graph
        self._x = None         # 输入占位符
        self._keep_prob = None # Dropout 保留概率（推理时 =1）
        self._k_p_conv = None  # 卷积 dropout 保留概率（推理时 =1）
        self._logits = None    # 输出 logits
        self._softmax = None   # softmax 概率
        self.loaded = False

    def load_model(self) -> None:
        """
        加载 TF1 checkpoint 模型。
        查找 .meta 文件并恢复计算图和权重。

        异常:
            FileNotFoundError: 模型文件不存在时抛出
        """
        meta_file = self.model_path + ".meta"

        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"模型文件未找到: {meta_file}\n"
                f"请确认以下文件存在:\n"
                f"  - {self.model_path}.meta\n"
                f"  - {self.model_path}.index\n"
                f"  - {self.model_path}.data-00000-of-00001\n"
                f"可通过环境变量 MODEL_PATH 指定模型路径"
            )

        print(f"[INFO] 正在加载模型: {self.model_path}")

        # 创建 TF Session
        self.sess = tf.Session()

        # 导入计算图结构并恢复权重
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(self.sess, self.model_path)

        # 获取计算图中的关键张量（名称必须与训练时定义的一致）
        self.graph = tf.get_default_graph()
        self._x = self.graph.get_tensor_by_name("x:0")
        self._keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self._k_p_conv = self.graph.get_tensor_by_name("k_p_conv:0")
        self._logits = self.graph.get_tensor_by_name("logits:0")
        self._softmax = tf.nn.softmax(self._logits)

        self.loaded = True
        print(f"[INFO] 模型加载成功 ✓")

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        对预处理后的图像执行推理。

        参数:
            image_array: shape=(1, 32, 32, 1) 的预处理图像

        返回:
            softmax 概率数组, shape=(1, 43)，每个元素为对应类别的概率
        """
        proba = self.sess.run(
            self._softmax,
            feed_dict={
                self._x: image_array,
                self._keep_prob: 1.0,   # 推理时关闭 dropout
                self._k_p_conv: 1.0,    # 推理时关闭 conv dropout
            },
        )
        return proba

    def close(self) -> None:
        """关闭 TF Session，释放资源"""
        if self.sess is not None:
            self.sess.close()
            print("[INFO] TF Session 已关闭")
