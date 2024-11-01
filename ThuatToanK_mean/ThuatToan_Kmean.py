from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
import numpy as np

# Tải bộ dữ liệu IRIS
data = load_iris()
X = data.data  # Các đặc trưng
y_true = data.target  # Nhãn thực tế

# Khởi tạo mô hình K-means với 3 cụm
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Lấy nhãn dự đoán từ mô hình K-means
y_pred = kmeans.labels_

# Ánh xạ các nhãn dự đoán để khớp với nhãn thực tế
predicted_labels = np.zeros_like(y_pred)
for i in range(3):
    mask = (y_pred == i)
    most_common = np.bincount(y_true[mask]).argmax()
    predicted_labels[mask] = most_common

# Tính F1-score (đo lường chất lượng phân cụm)
f1 = f1_score(y_true, predicted_labels, average='weighted')
print("F1 Score:", f1)

# Tính Adjusted Rand Index
rand_index = adjusted_rand_score(y_true, y_pred)
print("Adjusted Rand Index:", rand_index)

# Tính Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(y_true, y_pred)
print("Normalized Mutual Information (NMI):", nmi)

# Tính Davies-Bouldin Index
db_index = davies_bouldin_score(X, y_pred)
print("Davies-Bouldin Index:", db_index)
