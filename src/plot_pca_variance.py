import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from face_recognition_app import load_dataset
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
def plot_pca_curve():
    print("正在加载数据集...")
    X, y = load_dataset()  # 修复：只取2个返回值
    
    # 数据安全处理
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    
    print(f"Shape: {X.shape[0]}, {X.shape[1]}")
    
    print("正在计算PCA方差贡献率...")
    pca_full = PCA(svd_solver="full").fit(X)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 找到95%方差的维度
    target_variance = 0.95
    optimal_dim = np.argmax(cumulative_variance >= target_variance) + 1
    
    # 绘图
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-', linewidth=2, 
             label='累计方差贡献率')
    plt.axhline(y=target_variance, color='r', linestyle='--', label=f'95%阈值')
    plt.axvline(x=optimal_dim, color='g', linestyle=':', 
                label=f'最优维度={optimal_dim}')
    
    # 添加标注
    plt.scatter([optimal_dim], [cumulative_variance[optimal_dim-1]], 
                color='red', s=80)
    plt.text(optimal_dim+5, target_variance-0.05, 
             f'维度={optimal_dim}\n方差={cumulative_variance[optimal_dim-1]*100:.2f}%',
             fontsize=10, bbox=dict(alpha=0.3))
    
    # 图表美化
    plt.title('PCA 累计方差贡献率与最优维度选择', fontsize=16)
    plt.xlabel('主成分个数', fontsize=14)
    plt.ylabel('累计方差贡献率', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存结果
    save_path = "results/pca_variance_curve.png"
    Path("results").mkdir(exist_ok=True)  # 确保目录存在
    plt.savefig(save_path)
    print(f"\n✅ 图片已保存: {save_path}")
    print(f"📊 95%方差对应的维度: {optimal_dim}")
    plt.show()

if __name__ == "__main__":
    plot_pca_curve()