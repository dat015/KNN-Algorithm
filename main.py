import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D  # Đã có trong import

# Bước 1: Tải và chuẩn bị dữ liệu
# Đọc dữ liệu từ file 'network_attack_data.csv'
data = pd.read_csv('network_attack_data.csv')

# Kiểm tra sự tồn tại của cột nhãn
label_col = 'label'
if label_col not in data.columns:
    raise ValueError("File CSV thiếu cột nhãn 'label'")

# Tách đặc trưng và nhãn
# Lấy tất cả cột trừ cột nhãn làm đặc trưng
feature_cols = [col for col in data.columns if col != label_col]
if not feature_cols:
    raise ValueError("File CSV không có cột đặc trưng nào")
X = data[feature_cols].values
y = data[label_col].values

# Bước 2: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Bước 3: Hàm chuẩn hóa dữ liệu
def standardize_data(X):
    X = X.copy()
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1
    X_standardized = (X - means) / stds
    return X_standardized, means, stds

# Chuẩn hóa dữ liệu
X_train_scaled, train_means, train_stds = standardize_data(X_train)
X_test_scaled = (X_test - train_means) / train_stds

# Bước 4: Hàm tính khoảng cách Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Bước 5: Hàm tìm k láng giềng gần nhất
def get_neighbors(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return [neighbor[1] for neighbor in neighbors]

# Bước 6: Hàm dự đoán
def predict(X_train, y_train, X_test, k):
    predictions = []
    for x_test in X_test:
        neighbors = get_neighbors(X_train, y_train, x_test, k)
        label_counts = {}
        for label in neighbors:
            label_counts[label] = label_counts.get(label, 0) + 1
        predicted_label = max(label_counts, key=label_counts.get)
        predictions.append(predicted_label)
    return np.array(predictions)

# Hàm tính xác suất lớp Attack
def predict_proba(X_train, y_train, X_test, k):
    probabilities = []
    for x_test in X_test:
        neighbors = get_neighbors(X_train, y_train, x_test, k)
        attack_count = sum(1 for label in neighbors if label == 1)
        prob = attack_count / k
        probabilities.append(prob)
    return np.array(probabilities)

# Bước 7: Hàm đánh giá
def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    cm = [[0, 0], [0, 0]]
    for true, pred in zip(y_true, y_pred):
        cm[int(true)][int(pred)] += 1
    return np.array(cm)

def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Bước 8: Dự đoán với k=3
k = 3
y_pred = predict(X_train_scaled, y_train, X_test_scaled, k)
y_prob = predict_proba(X_train_scaled, y_train, X_test_scaled, k)

# Bước 9: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1 = precision_recall_f1(y_test, y_pred)
print(f"Độ chính xác: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.savefig('confusion_matrix.png')
plt.close()

# Biểu đồ ROC
fpr = []
tpr = []
thresholds = np.linspace(0, 1, 100)
for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Đường cong ROC')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Biểu đồ độ chính xác theo k
k_values = range(1, min(6, len(X_train) + 1))
accuracies = []
for k in k_values:
    y_pred_k = predict(X_train_scaled, y_train, X_test_scaled, k)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
plt.figure(figsize=(6, 4))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Giá trị k')
plt.ylabel('Độ chính xác')
plt.title('Độ chính xác theo giá trị k')
plt.savefig('accuracy_vs_k.png')
plt.close()

# Biểu đồ phân phối đặc trưng
# Giới hạn tối đa 4 đặc trưng để tránh quá nhiều biểu đồ
n_features_to_plot = min(len(feature_cols), 4)
plt.figure(figsize=(10, 8))
for i, feature in enumerate(feature_cols[:n_features_to_plot]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=data, x=feature, hue='label', multiple='stack', palette=['blue', 'red'])
    plt.title(f'Phân phối {feature}')
plt.tight_layout()
plt.savefig('feature_distribution.png')
plt.close()

# Bước 10: Dự đoán dữ liệu mới
# Tạo dữ liệu mới với số đặc trưng bằng số cột đặc trưng trong dữ liệu
new_data = np.array([[0.5] * len(feature_cols)])  # Giá trị mẫu, có thể tùy chỉnh
new_data_scaled = (new_data - train_means) / train_stds
new_prediction = predict(X_train_scaled, y_train, new_data_scaled, k)
print("Dự đoán cho dữ liệu mới:", "Attack" if new_prediction[0] == 1 else "Normal")

# Bước 11: Trực quan hóa dữ liệu (2D)
# Chọn hai đặc trưng đầu tiên nếu có ít nhất 2 đặc trưng, nếu không thì bỏ qua
if len(feature_cols) >= 2:
    feature_pair = (0, 1)  # Chọn cặp đặc trưng đầu tiên
    X_train_2d = X_train_scaled[:, feature_pair]
    X_test_2d = X_test_scaled[:, feature_pair]
    y_pred_2d = predict(X_train_2d, y_train, X_test_2d, k)
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train_2d[y_train == 0][:, 0], X_train_2d[y_train == 0][:, 1], c='blue', label='Normal (Train)')
    plt.scatter(X_train_2d[y_train == 1][:, 0], X_train_2d[y_train == 1][:, 1], c='red', label='Attack (Train)')
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c='green', marker='x', s=100, label='Test')
    plt.xlabel(f'{feature_cols[feature_pair[0]]} (chuẩn hóa)')
    plt.ylabel(f'{feature_cols[feature_pair[1]]} (chuẩn hóa)')
    plt.title('Dữ liệu huấn luyện và kiểm tra (2D)')
    plt.legend()
    plt.savefig('data_visualization.png')
    plt.close()
else:
    print("Không đủ đặc trưng để vẽ biểu đồ phân tán 2D")

# Bước 12: Trực quan hóa dữ liệu (3D)
# Chỉ thực hiện nếu có ít nhất 3 đặc trưng
if len(feature_cols) >= 3:
    feature_triplet = (0, 1, 2)  # Chọn 3 đặc trưng đầu tiên
    X_train_3d = X_train_scaled[:, feature_triplet]
    X_test_3d = X_test_scaled[:, feature_triplet]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ dữ liệu huấn luyện
    ax.scatter(
        X_train_3d[y_train == 0][:, 0], 
        X_train_3d[y_train == 0][:, 1], 
        X_train_3d[y_train == 0][:, 2], 
        c='blue', label='Normal (Train)', alpha=0.6
    )
    ax.scatter(
        X_train_3d[y_train == 1][:, 0], 
        X_train_3d[y_train == 1][:, 1], 
        X_train_3d[y_train == 1][:, 2], 
        c='red', label='Attack (Train)', alpha=0.6
    )

    # Vẽ dữ liệu kiểm tra
    ax.scatter(
        X_test_3d[:, 0], 
        X_test_3d[:, 1], 
        X_test_3d[:, 2], 
        c='green', marker='x', s=60, label='Test'
    )

    ax.set_xlabel(f'{feature_cols[feature_triplet[0]]} (chuẩn hóa)')
    ax.set_ylabel(f'{feature_cols[feature_triplet[1]]} (chuẩn hóa)')
    ax.set_zlabel(f'{feature_cols[feature_triplet[2]]} (chuẩn hóa)')
    ax.set_title('Trực quan hóa dữ liệu (3D)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('data_visualization_3d.png')
    plt.close()
else:
    print("Không đủ đặc trưng để vẽ biểu đồ phân tán 3D")