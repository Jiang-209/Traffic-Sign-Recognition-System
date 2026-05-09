import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import csv

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================
# 加载模型
# ==============================
sess = tf.Session()
saver = tf.train.import_meta_graph('traffic_modelora.meta')
saver.restore(sess, './traffic_modelora')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
k_p_conv = graph.get_tensor_by_name('k_p_conv:0')
logits = graph.get_tensor_by_name('logits:0')
softmax = tf.nn.softmax(logits)

# ==============================
# 读取类别名称
# ==============================
with open('data/signnames.csv', 'r') as f:
    reader = csv.reader(f)
    sign_names = list(reader)[1:]

# ==============================
# 预处理
# ==============================
IMG_HEIGHT = 32
IMG_WIDTH = 32

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_np = np.array(img)
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    gray = gray / 255.0 - 0.5
    return gray.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

# ==============================
# GUI逻辑
# ==============================
canvas = None  # 用于更新图表

def upload_image():
    global canvas

    file_path = filedialog.askopenfilename(filetypes=[("All Image Files", "*.*")])
    if not file_path:
        return

    img = Image.open(file_path)

    # 显示图片
    img_display = img.copy()
    img_display.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 预测
    processed = preprocess_image(img)
    pred_proba = sess.run(softmax, feed_dict={x: processed, keep_prob: 1, k_p_conv: 1})
    pred_class = np.argmax(pred_proba, axis=1)[0]

    result_label.config(text=f"预测结果: {sign_names[pred_class][1]}")

    # ==============================
    # 画概率柱状图（Top 5）
    # ==============================
    probs = pred_proba[0]
    top5_idx = probs.argsort()[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_labels = [sign_names[i][1] for i in top5_idx]

    fig = plt.figure(figsize=(7,5))
    plt.barh(top5_labels, top5_probs)
    plt.xlabel("Probability")
    plt.title("Top-5 Predictions")
    plt.tight_layout()

    # 清除旧图
    if canvas:
        canvas.get_tk_widget().destroy()

    # 嵌入Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# ==============================
# 界面
# ==============================
root = tk.Tk()
root.title("交通标志识别系统")

upload_btn = Button(root, text="上传图片", command=upload_image)
upload_btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="预测结果:", font=("Arial", 14))
result_label.pack(pady=10)

root.geometry("700x750")
root.mainloop()