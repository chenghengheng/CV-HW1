# CV-HW1
## 训练
调用`para()`函数，初始学习率为0.001，隐藏层大小为100，正则化强度0.01。
训练参数储存在相应txt文件中。

## 测试
应用`numpy.loadtxt()`读取网络参数，调用`yhat = predict(w, b, Xtest, [100], nLabels)`进行测试。
