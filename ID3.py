import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# โหลดข้อมูล
path = '/Users/natxpss/Downloads/Dtree/iris_modified.csv'
df = pd.read_csv(path)

# แบ่งข้อมูลเป็น features และ target
X = df.drop('class', axis=1)
y = df['class']

# แบ่งข้อมูลเป็นชุดฝึกอบรมและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ฟังก์ชันคำนวณ entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([-counts[i] / np.sum(counts) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

# ฟังก์ชันคำนวณ information gain
def InfoGain(X, y, split_attribute_name):
    total_entropy = entropy(y)
    vals, counts = np.unique(X[split_attribute_name], return_counts=True)
    Weighted_Entropy = sum(
        (counts[i] / np.sum(counts)) * entropy(y[X[split_attribute_name] == vals[i]]) for i in range(len(vals)))
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

# ฟังก์ชัน ID3
def id3(X, y, attributes, tree=None, depth=0, max_depth=3):
    # ตรวจสอบเงื่อนไขการหยุด
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    elif len(attributes) == 0 or depth == max_depth:
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    # เลือก attribute ที่ดีที่สุด
    IGs = {attr: InfoGain(X, y, attr) for attr in attributes}
    for attr, ig in IGs.items():
        print(f"Attribute '{attr}' has Information Gain: {ig}")
    best_attr = max(IGs, key=IGs.get)
    print(f"\nBest attribute at depth {depth}: '{best_attr}' with IG: {IGs[best_attr]}\n")

    # สร้าง node และ subtree สำหรับ attribute ที่เลือก
    if tree is None:
        tree = {}
    tree[best_attr] = {}

    remaining_attributes = [i for i in attributes if i != best_attr]

    # สร้าง subtree สำหรับแต่ละค่าของ attribute
    for value in np.unique(X[best_attr]):
        sub_X = X[X[best_attr] == value]
        sub_y = y[X[best_attr] == value]
        subtree = id3(sub_X, sub_y, remaining_attributes, tree=None, depth=depth + 1, max_depth=max_depth)
        tree[best_attr][value] = subtree

    return tree



# แปลง X_train และ y_train เป็น DataFrame และ Series ตามลำดับ
X_train_df = pd.DataFrame(X_train, columns=df.columns[:-1])
y_train_series = pd.Series(y_train)

# ใช้ ID3 algorithm
attributes = list(X_train_df.columns)
tree = id3(X_train_df, y_train_series, attributes)

print(tree)

def predict(query, tree, default='unknown'):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


#ลองนำข้อมูลมาทดสอบ
def test(data, tree):
    # รีเซ็ต index ของ data ให้เริ่มต้นที่ 0 และเพิ่มขึ้นเรื่อยๆ
    data = data.reset_index(drop=True)
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

    # การคำนวณสำหรับแต่ละแถวในชุดข้อมูลทดสอบ
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 'unknown')

    # รีเซ็ต index ของ predicted เพื่อให้ตรงกับ data
    predicted = predicted.reset_index(drop=True)

    # คำนวณความแม่นยำ
    accuracy = np.sum(predicted["predicted"] == data["class"]) / len(data) * 100

    return accuracy

# แปลง X_test และ y_test เป็น DataFrame และเตรียมข้อมูลสำหรับการทดสอบ
test_data = X_test.copy()
test_data['class'] = y_test.values

# คำนวณความแม่นยำ
accuracy = test(test_data, tree)
print(f"Accuracy: {accuracy}%")
