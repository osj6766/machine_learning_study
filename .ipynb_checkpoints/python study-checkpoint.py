import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# "반가워요"

# 파일 경로
file_path = r"C:\Users\User\Desktop\Clio-NatCommData-main\simulated_optimization\bo\bo_ttei_uniform_1.pkl"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# 파일 로딩
with open(file_path, "rb") as file:
    data = pickle.load(file)

# 데이터 추출
points = np.array(data['points']).squeeze()
vals = np.array(data['vals']).squeeze()
true_vals = np.array(data['true_vals']).squeeze()

# 누적 최고값 계산
best_true_vals = np.maximum.accumulate(true_vals)
best_val = np.max(true_vals)
regret = best_val - best_true_vals
best_idx = np.argmax(true_vals == best_val)

# 1. Conductivity vs Iteration
plt.figure(figsize=(7, 5))
plt.plot(true_vals, label='True Values')
plt.plot(best_true_vals, label='Best So Far')
plt.scatter(best_idx, true_vals[best_idx], color='red', zorder=5, label='Best Point')
plt.xlabel('Iteration')
plt.ylabel('Ionic conductivity (mS/cm)')
plt.title('Conductivity vs Iteration')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Regret vs Iteration
plt.figure(figsize=(7, 5))
plt.plot(regret)
plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('Regret vs Iteration')
plt.tight_layout()
plt.show()
