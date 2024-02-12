import pandas as pd
import numpy as np

# โหลดข้อมูล
path = '/Users/natxpss/Downloads/Dtree/iris_modified.csv'
df = pd.read_csv(path)

# ฟังก์ชันคำนวณ entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([-counts[i] / np.sum(counts) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])

# ฟังก์ชันคำนวณ information gain
def InfoGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
        for i in range(len(vals))])
    return total_entropy - Weighted_Entropy

# คำนวณ IG สำหรับแต่ละ attribute ที่ไม่ใช่คลาส
IGs = {}
for column in df.columns[:-1]:  # ละเว้นคอลัมน์ class
    IG = InfoGain(df, column, 'class')
    IGs[column] = IG

# แสดงผล IG ของแต่ละ attribute
for attribute, ig in IGs.items():
    print(f"{attribute}: {ig}")

# หา attribute ที่มี IG สูงสุด
best_attribute = max(IGs, key=IGs.get)
print(f"Best attribute for root node: {best_attribute} with IG: {IGs[best_attribute]}")

