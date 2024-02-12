import pandas as pd
import plotly.express as px

# Load the modified dataset
file_path = '/Users/natxpss/Downloads/Dtree/iris_modified.csv'
df_modified = pd.read_csv(file_path)

# ตัวอย่างการนับจำนวนข้อมูลในแต่ละหมวดหมู่สำหรับคอลัมน์ 'sepal length'
count_sepal_length = df_modified.groupby(['class', 'sepal length']).size().unstack(fill_value=0)
print(count_sepal_length)

count_sepal_length = df_modified.groupby(['class', 'sepal width']).size().unstack(fill_value=0)
print(count_sepal_length)

count_sepal_length = df_modified.groupby(['class', 'petal length']).size().unstack(fill_value=0)
print(count_sepal_length)

count_sepal_length = df_modified.groupby(['class', 'petal width']).size().unstack(fill_value=0)
print(count_sepal_length)

# กรองข้อมูลเพื่อเลือกเฉพาะคลาส Iris-setosa
df_setosa = df_modified[df_modified['class'] == 'Iris-setosa']

# ตรวจสอบว่าคอลัมน์มีการเปลี่ยนแปลงเป็นหมวดหมู่หรือยังคงเป็นตัวเลข
# สมมติว่าคอลัมน์ยังคงเป็นตัวเลขสำหรับการสร้าง box plot

# สร้าง box plot สำหรับ df_setosa
fig = px.box(df_setosa,
             y=['sepal length', 'sepal width', 'petal length', 'petal width'],
             title="Box Plot of Iris-setosa Data",
             labels={'value': 'Measurement (cm)', 'variable': 'Feature'},
             points="all")

fig.show()

# กรองข้อมูลเพื่อเลือกเฉพาะคลาส Iris-versicolor
df_versicolor = df_modified[df_modified['class'] == 'Iris-versicolor']

# สร้าง box plot สำหรับ df_versicolor
fig = px.box(df_versicolor,
             y=['sepal length', 'sepal width', 'petal length', 'petal width'],
             title="Box Plot of Iris-versicolor Data",
             labels={'value': 'Measurement (cm)', 'variable': 'Feature'},
             points="all",
             color_discrete_sequence = ['red']
             )


fig.show()

# กรองข้อมูลเพื่อเลือกเฉพาะคลาส Iris-virginica
df_virginica = df_modified[df_modified['class'] == 'Iris-virginica']

# สร้าง box plot สำหรับ df_virginica
fig = px.box(df_virginica,
             y=['sepal length', 'sepal width', 'petal length', 'petal width'],
             title="Box Plot of Iris-virginica Data",
             labels={'value': 'Measurement (cm)', 'variable': 'Feature'},
             points="all",
             color_discrete_sequence=['green']
            )

fig.show()

fig = px.box(df_modified,
             y=['sepal length', 'sepal width', 'petal length', 'petal width'],
             color='class',  # ใช้ 'class' สำหรับแยกสีตามคลาสของดอกไม้
             title="Box Plot of Iris Data by Class",
             labels={'value': 'Measurement (cm)', 'variable': 'Feature'},
             points="all")

fig.show()
