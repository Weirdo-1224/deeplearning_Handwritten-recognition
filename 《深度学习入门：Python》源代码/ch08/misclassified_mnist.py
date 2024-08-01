# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


# 加载MNIST数据集，flatten=False表示不要展平图像
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 创建一个深度卷积神经网络
network = DeepConvNet()

# 从已保存的参数文件中加载神经网络的参数
network.load_params("deep_convnet_params.pkl")

# 打印提示信息
print("Calculating test accuracy...")

# 指定要测试的样本数量
# sampled = 1000
# x_test = x_test[:sampled]
# t_test = t_test[:sampled]

# 初始化一个空列表，用于存储分类结果
classified_ids = []

# 初始化准确率为0.0
acc = 0.0

# 指定批量大小
batch_size = 100

# 循环遍历测试数据集，并按批次进行分类和准确率计算
for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i * batch_size:(i + 1) * batch_size]  # 获取一个批次的测试数据
    tt = t_test[i * batch_size:(i + 1) * batch_size]  # 获取该批次的真实标签
    y = network.predict(tx, train_flg=False)  # 使用神经网络进行预测
    y = np.argmax(y, axis=1)  # 取预测结果中概率最高的类别作为预测结果
    classified_ids.append(y)  # 将预测结果添加到分类结果列表中
    acc += np.sum(y == tt)  # 计算该批次内的正确分类数量

# 计算整体的准确率
acc = acc / x_test.shape[0]

# 打印测试准确率
print("Test accuracy: " + str(acc))

# 将分类结果转换为NumPy数组
classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()

# 设置图像显示参数
# 设置最大视图数量
max_view = 20
# 设置当前视图的索引，初始为 1
current_view = 1
# 创建一个 Matplotlib 图形对象
fig = plt.figure()
# 调整子图布局，使子图占据整个图形，没有边距，同时子图之间有一些垂直和水平间隔
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)


# 创建一个字典，用于存储被错误分类的样本的真实标签和预测标签
mis_pairs = {}

# 遍历测试样本，找到并显示前20个被错误分类的样本
for i, val in enumerate(classified_ids == t_test):
    if not val:
        # 创建一个子图（Axes）并将其添加到图形（Figure）上
        # 参数 (4, 5, current_view) 指定子图的网格布局，4 行 5 列，current_view 是子图的序号
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])

        # 使用 imshow 函数在子图上显示图像
        # x_test[i] 是要显示的图像数据，这里假设 x_test 是一个图像数据集
        # 图像被调整为 28x28 像素并以灰度图像方式显示
        # cmap=plt.cm.gray_r 表示使用灰度颜色映射，interpolation='nearest' 表示最近邻插值
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')

        mis_pairs[current_view] = (t_test[i], classified_ids[i])
        current_view += 1
        if current_view > max_view:
            break

# 打印被错误分类的样本信息
print("======= Misclassified result =======")
print("{view index: (true label, predicted label), ...}")
print(mis_pairs)

# 显示图像
plt.show()
