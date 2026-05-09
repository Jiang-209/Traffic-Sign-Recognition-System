import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pickle
import cv2
import csv

# ==============================

# 1. 加载测试数据

# ==============================

testing_file = 'data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']

# ==============================

# 2. 读取类别名称

# ==============================

with open('data/signnames.csv', 'r') as f:
    reader = csv.reader(f)
    sign_names = list(reader)[1:]

# ==============================

# 3. 数据预处理（必须和训练一致）

# ==============================

IMG_HEIGHT = 32
IMG_WIDTH = 32

def preprocessed(dataset):
    n_imgs = dataset.shape[0]
    processed = np.zeros((n_imgs, IMG_HEIGHT, IMG_WIDTH, 1))

    for i in range(n_imgs):
        gray = cv2.cvtColor(dataset[i], cv2.COLOR_RGB2GRAY)
        processed[i,:,:,0] = gray / 255.0 - 0.5

    return processed

X_test_prep = preprocessed(X_test)
print("Test data shape:", X_test_prep.shape)

# ==============================

# 4.加载模型 + 推理

# ==============================

with tf.Session() as sess:
    print("Loading model...")
    # 加载图结构
    saver = tf.train.import_meta_graph('traffic_modelora.meta')
    # 加载权重
    saver.restore(sess, './traffic_modelora')

    print("Model loaded successfully!")

    # 获取计算图
    graph = tf.get_default_graph()

    # 从图中取变量（名字必须和训练时一致）
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    k_p_conv = graph.get_tensor_by_name('k_p_conv:0')
    logits = graph.get_tensor_by_name('logits:0')

    # softmax
    softmax = tf.nn.softmax(logits)

    # ==============================
    # 5.预测
    # ==============================
    BATCH_SIZE = 128

    num_examples = len(X_test_prep)
    predictions = []

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_test_prep[offset:offset + BATCH_SIZE]

        batch_pred = sess.run(softmax, feed_dict={
            x: batch_x,
            keep_prob: 1,
            k_p_conv: 1
        })

        predictions.append(batch_pred)

    # 拼接所有结果
    pred_proba = np.vstack(predictions)

    prediction = np.argmax(pred_proba, axis=1)
    # ==============================
    # 6.计算准确率
    # ==============================
    accuracy = np.mean(prediction == y_test)
    print("Test Accuracy = {:.4f}".format(accuracy))

    # ==============================
    # 7.显示前10个预测结果
    # ==============================
    print("\nSample Predictions:")
    for i in range(10):
        print("Pred:", sign_names[prediction[i]][1],
              "| True:", sign_names[y_test[i]][1])