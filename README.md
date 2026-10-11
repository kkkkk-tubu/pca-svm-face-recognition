# 基于 PCA 与 SVM 的开放集人脸识别系统

基于 PCA 特征子空间 + SVM 分类器的两阶段开放集人脸识别系统，支持已知身份识别与未知身份拒识。

> 《机器学习》课程设计 · 2 人协作项目

## 项目简介

人脸识别是人工智能领域的核心研究方向，广泛应用于安防监控、身份认证、智能交互等场景。传统方法在**封闭集假设**（所有测试身份均已知）下表现良好，但实际系统中常需面对**未知身份的拒绝识别**问题。

本项目实现了**开放集人脸识别**：既能准确识别已知身份，又能有效拒绝未知身份。

**核心指标**：

- 闭集识别准确率 **90.0%**（30 个已知类，每类取 14 张训练）
- 具备开放集拒绝能力 ✅
- 提供 GUI 交互界面 ✅

## 技术路线

```
输入图像 (24×24 灰度)
    ↓ 展平归一化
576 维特征向量
    ↓ PCA 投影
87 维特征子空间
    ↓ 计算重构误差
MSE 与阈值 τ 比较
    ↓
┌─────────────┴─────────────┐
ε ≤ τ  →  SVM 预测  →  输出已知类别
ε > τ  →  拒绝识别  →  输出未知类别
```

## 数据集

**GTdb_crop 人脸数据集**

| 项目 | 说明 |
|---|---|
| 图像规格 | 24×24 像素灰度图 |
| 特征维度 | 576 维（展平向量） |
| 类别分布 | s01 ~ s50，共 50 个身份 |
| 样本特点 | 每类含多张不同姿态/光照 |

**数据划分策略**：

- 已知类 **30 个**，每类取 14 张用于训练
- 剩余样本用于闭集测试
- 未知类 **20 个**完全不参与训练

**预处理**：灰度化 → 尺寸归一化(24×24) → 像素值归一化到 [0,1] → 展平为 576 维向量

## 方法设计

### 1. PCA 降维

**核心思想**：将 576 维人脸向量投影到低维子空间，寻找一组正交基使数据在新坐标轴上方差最大化。

**实现步骤**：

1. **数据中心化**——计算训练集均值脸 μ，所有样本减去 μ
2. **协方差矩阵**——sklearn 内部使用 SVD 分解替代显式计算（数值更稳定）
3. **特征分解**——按特征值降序排列，取前 k 个特征向量构成投影矩阵 W
4. **投影变换**——Z = X·W，从 576 维降至 87 维
5. **维度确定**——累计方差 ≥ 95% 自动截取

**结果**：维度从 576 → **87**，压缩率 **84.9%**，保留 95% 方差信息。

```python
pca = PCA(n_components=0.95, svd_solver="full")
z_train = pca.fit_transform(X_train)   # (n, 87)
```

### 2. SVM 分类器

采用 RBF 核，通过 OVO（One-vs-One）策略处理多分类：

```python
svm = SVC(C=20.0, gamma="scale", kernel="rbf",
          probability=True, class_weight="balanced")
```

- **OVO 策略**：训练 C(30,2) = **435 个二分类器**，测试时投票决定类别
- **gamma="scale"**：自动计算 γ = 1/(n_features × X.var())
- **probability=True**：对每个二分类器拟合 Platt sigmoid，输出概率

**闭集测试准确率：90.0%**

> 测试集是已知的 30 类里每类剩下的那 1 张，一共 30 张。

### 3. 开放集识别机制（核心创新点）

**问题**：传统 SVM 只能进行闭集分类，无法拒绝未知身份。

**解决方案**：基于 PCA 子空间重构误差的阈值判定法。

**核心思想**：

- PCA 子空间由**训练集特征**构建，已知类人脸在该子空间内分布
- 未知类人脸的特征分布与子空间不匹配，**投影后再重构损失大**
- MSE 量化了"子空间外信息"的强度，是天然的拒绝指标

**阈值选取**：取训练集重构误差的 **99.5 百分位数**

- 100% 会包含训练集中极少数异常样本（姿态极端、光照异常）
- 99.5% 留出 0.5% 裕量，使正常已知类样本都能通过

```python
# 阈值计算
reconstructed = pca.inverse_transform(z_train)
errors = np.mean((X_train - reconstructed) ** 2, axis=1)
threshold = float(np.percentile(errors, 99.5))

# 预测逻辑（双分支共享 PCA 投影，避免重复计算）
def predict(model, image_path):
    vector = read_image(image_path).reshape(1, -1)   # 576 维
    z = model.pca.transform(vector)                  # 投影 → 87 维
    pred = model.svm.predict(z)                      # SVM 投票
    proba = model.svm.predict_proba(z)               # Platt 概率
    reconstructed = model.pca.inverse_transform(z)   # 重构 → 576 维
    recon_error = np.mean((vector - reconstructed) ** 2)
    is_known = recon_error <= model.recon_threshold  # 拒绝判定
    return is_known, pred[0], recon_error, np.max(proba) * 100
```

## 方法优势

- **识别率高**——保留 87 维关键特征，去除 489 维噪声冗余
- **开放集能力**——利用 PCA 子空间几何特性，未知类偏离训练分布时重构误差天然偏大
- **计算高效**——训练后仅需矩阵乘法与逐元素 MSE，实时性好
- **参数少**——仅需调节 PCA 保留比例与 SVM 超参数，无需额外设计拒绝模块

## 系统架构

```
├── 数据预处理（灰度化 / 归一化 / 展平）
├── PCA + 开放集判定（核心算法）
├── SVM 分类器训练与预测
└── GUI 交互界面（Tkinter）
```

## 我负责的部分

**PCA 降维、SVM 训练、开放集识别算法实现**

- PCA 特征子空间构建：从 576 维降至 87 维，保留 95% 方差
- SVM 多分类器训练与超参数配置（RBF 核，C=20.0）
- **独立设计开放集识别机制**：基于重构误差百分位阈值的拒识方案
- 算法整体流程串联与预测函数实现

## 技术栈

Python · scikit-learn · NumPy · Pillow · joblib · Tkinter

## 参考

- Turk & Pentland, *Eigenfaces for Recognition*, Journal of Cognitive Neuroscience, 1991
- Platt, *Probabilistic Outputs for Support Vector Machines*, 1999
