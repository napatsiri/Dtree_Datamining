from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# สร้าง DataFrame จากข้อมูลที่ได้ทำการ preprocess มาแล้ว
data = """
sepal length,sepal width,petal length,petal width,class
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_small,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_normal,sw_big,pl_small,pw_small,Iris-setosa
sl_normal,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_normal,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_normal,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_normal,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_small,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_small,sw_big,pl_small,pw_small,Iris-setosa
sl_small,sw_normal,pl_small,pw_small,Iris-setosa
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_small,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_small,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_small,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_big,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_big,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_big,pw_normal,Iris-versicolor
sl_small,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_big,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_small,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_normal,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_small,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_normal,sw_small,pl_normal,pw_normal,Iris-versicolor
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_normal,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_small,sw_small,pl_normal,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_big,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_normal,sw_small,pl_normal,pw_big,Iris-virginica
sl_normal,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_big,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_normal,sw_small,pl_normal,pw_normal,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_normal,sw_small,pl_normal,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_normal,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_normal,pw_big,Iris-virginica
sl_normal,sw_normal,pl_normal,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_normal,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_big,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_big,pw_normal,Iris-virginica
sl_normal,sw_small,pl_big,pw_normal,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_big,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_normal,sw_normal,pl_normal,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_normal,sw_small,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_small,pl_normal,pw_big,Iris-virginica
sl_big,sw_normal,pl_big,pw_big,Iris-virginica
sl_big,sw_big,pl_big,pw_big,Iris-virginica
sl_normal,sw_normal,pl_big,pw_big,Iris-virginica

"""

# ในตัวอย่างนี้ ข้อมูลจะถูกตัดจบด้วย ... คุณต้องเติมข้อมูลที่เหลือให้ครบถ้วน

# เนื่องจากข้อมูลมีปริมาณมาก ในที่นี้เราจะจำลองข้อมูลแทนการใส่ทั้งหมด
# ในการใช้งานจริง คุณควรโหลดข้อมูลจากไฟล์ CSV หรือแหล่งข้อมูลอื่น

# สร้าง DataFrame จากข้อมูล
# ใช้ StringIO เนื่องจาก pd.read_csv() ปกติจะอ่านจากไฟล์
from io import StringIO
df = pd.read_csv(StringIO(data), skipinitialspace=True)

# แปลงข้อมูล categorical เป็นตัวเลข
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# แยก features และ target
X = df.drop('class', axis=1)
y = df['class']

# สร้างและฝึกโมเดล Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='entropy')  # ใช้ entropy เป็นเกณฑ์
dt_classifier.fit(X, y)

# แสดงโครงสร้างของ Decision Tree
# ด้วยการพิมพ์ หรือ คุณอาจใช้ฟังก์ชั่นอื่นๆ เพื่อ visualize tree ได้
from sklearn.tree import export_text
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print(tree_rules)

data = {
    'sepal length': ['low'] * 100 + ['medium'] * 100 + ['high'] * 50,
    'sepal width': ['very_high'] * 80 + ['medium'] * 90 + ['high'] * 80,
    'petal length': ['low'] * 50 + ['medium'] * 150 + ['high'] * 50,
    'petal width': ['low'] * 70 + ['medium'] * 120 + ['high'] * 60,
    'class': [0] * 150 + [1] * 50 + [2] * 50
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encoding string labels into integers
le = LabelEncoder()
df['sepal length'] = le.fit_transform(df['sepal length'])
df['sepal width'] = le.fit_transform(df['sepal width'])
df['petal length'] = le.fit_transform(df['petal length'])
df['petal width'] = le.fit_transform(df['petal width'])

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X, y)

# Generate a visual representation of the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'])
plt.show()