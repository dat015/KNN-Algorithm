import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Tạo dataset giả lập nếu không có sẵn
try:
    df = pd.read_csv("network_attack_data.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Creating a sample dataset...")
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        "feature1": np.random.rand(data_size),
        "feature2": np.random.rand(data_size),
        "feature3": np.random.rand(data_size),
        "feature4": np.random.rand(data_size),
        "label": np.random.choice([0, 1], data_size)  # 0: Bình thường, 1: Tấn công
    })
    df.to_csv("network_attack_data.csv", index=False)
    print("Sample dataset created and saved as 'network_attack_data.csv'.")

# Hiển thị thông tin dữ liệu
print(df.head())

# Chia thành đặc trưng (X) và nhãn (y)
X = df.drop("label", axis=1)
y = df["label"]

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# In báo cáo phân loại
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix")
plt.show()

# Vẽ đồ thị chọn giá trị K tối ưu
error_rates = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    error_rates.append(np.mean(y_pred_k != y_test))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), error_rates, marker='o', linestyle='dashed', color='b')
plt.xlabel("Giá trị của K")
plt.ylabel("Tỉ lệ lỗi")
plt.title("Chọn K tốt nhất")
plt.show()
