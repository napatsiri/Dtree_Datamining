import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# โหลดข้อมูลและการเตรียมข้อมูล
file_path = '/Users/natxpss/Downloads/Dtree/iris.data'
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df = pd.read_csv(file_path, header=None, names=columns)

# ฟังก์ชั่นสำหรับการแสดงสถิติพื้นฐาน
def display_data_statistics(df, stage='Before'):
    print(f"{stage} preprocessing:")
    print(df.describe())

fig = px.box(df,
             y=['sepal length', 'sepal width', 'petal length', 'petal width'],
             color='class',  # Use the 'class' column for coloring
             labels={'value': 'Measurement (cm)', 'variable': 'Feature'},
             title="Box Plot of Iris Dataset Features by Class",
             orientation='v',
             points='all')

fig.show()

# ฟังก์ชั่นสำหรับการวาด Scatter Plot
def plot_scatter(df, stage='Before'):
    plt.figure(figsize=(10, 6))
    species = df['class'].unique()
    colors = ['red', 'green', 'blue']
    for i, specie in enumerate(species):
        subset = df[df['class'] == specie]
        plt.scatter(subset.iloc[:, 3], subset.iloc[:, 0], label=f"Class {specie}", color=colors[i])
    plt.xlabel('petal width')
    plt.ylabel('sepal length')
    plt.title(f'Scatter Plot of Iris Dataset by Sepal Dimensions ({stage} Preprocessing)')
    plt.legend()
    plt.show()

# แสดงสถิติและ Scatter Plot ก่อนการประมวลผล
display_data_statistics(df, 'Before')
plot_scatter(df, 'Before')

# การเปลี่ยนค่าหมวดหมู่เป็นตัวเลขและ Feature Scaling
encoder = LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])

# คำนวณค่าเฉลี่ยของ 'sepal length' สำหรับแต่ละคลาส
mean_sepal_length = df.groupby('class')['sepal length'].mean()
min_sepal_length = df.groupby('class')['sepal length'].min()
max_sepal_length = df.groupby('class')['sepal length'].max()

# วาดกราฟแสดงค่าเฉลี่ยของ 'sepal length' สำหรับแต่ละคลาส
plt.figure(figsize=(8, 5))
mean_sepal_length.plot(kind='bar')
plt.title('Mean sepal length for Each Class')
plt.xlabel('Class')
plt.ylabel('Mean sepal length')
plt.xticks(ticks=range(len(mean_sepal_length)), labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], rotation=45)
plt.show()

print("Statistics for Sepal Length by Class:")
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # Class names for printing

# Iterate over each class index to print statistics
for i in range(len(mean_sepal_length)):
    class_name = class_names[i]  # Get the class name based on index
    mean_value = mean_sepal_length.iloc[i]  # Access the mean value by index
    min_value = min_sepal_length.iloc[i]  # Access the min value by index
    max_value = max_sepal_length.iloc[i]  # Access the max value by index
    print(f"{class_name}: Mean = {mean_value:.2f}, Min = {min_value:.2f}, Max = {max_value:.2f}")


# คำนวณค่าเฉลี่ยของ 'sepal width' สำหรับแต่ละคลาส
mean_sepal_width = df.groupby('class')['sepal width'].mean()
min_sepal_width = df.groupby('class')['sepal width'].min()
max_sepal_width = df.groupby('class')['sepal width'].max()

# วาดกราฟแสดงค่าเฉลี่ยของ 'sepal width' สำหรับแต่ละคลาส
plt.figure(figsize=(8, 5))
mean_sepal_width.plot(kind='bar')
plt.title('Mean sepal width for Each Class')
plt.xlabel('Class')
plt.ylabel('Mean sepal width')
plt.xticks(ticks=range(len(mean_sepal_width)), labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], rotation=45)
plt.show()

print("Statistics for Sepal Width by Class:")
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # Class names for printing

# Iterate over each class index to print statistics
for i in range(len(mean_sepal_width)):
    class_name = class_names[i]  # Get the class name based on index
    mean_value = mean_sepal_width.iloc[i]  # Access the mean value by index
    min_value = min_sepal_width.iloc[i]  # Access the min value by index
    max_value = max_sepal_width.iloc[i]  # Access the max value by index
    print(f"{class_name}: Mean = {mean_value:.2f}, Min = {min_value:.2f}, Max = {max_value:.2f}")


# คำนวณค่าเฉลี่ยของ 'petal length' สำหรับแต่ละคลาส
mean_petal_length = df.groupby('class')['petal length'].mean()
min_petal_length = df.groupby('class')['petal length'].min()
max_petal_length = df.groupby('class')['petal length'].max()

# วาดกราฟแสดงค่าเฉลี่ยของ 'petal length' สำหรับแต่ละคลาส
plt.figure(figsize=(8, 5))
mean_petal_length.plot(kind='bar')
plt.title('Mean petal length for Each Class')
plt.xlabel('Class')
plt.ylabel('Mean petal length')
plt.xticks(ticks=range(len(mean_petal_length)), labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], rotation=45)
plt.show()

print("Statistics for Petal Length by Class:")
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # Class names for printing

# Iterate over each class index to print statistics
for i in range(len(mean_petal_length)):
    class_name = class_names[i]  # Get the class name based on index
    mean_value = mean_petal_length.iloc[i]  # Access the mean value by index
    min_value = min_petal_length.iloc[i]  # Access the min value by index
    max_value = max_petal_length.iloc[i]  # Access the max value by index
    print(f"{class_name}: Mean = {mean_value:.2f}, Min = {min_value:.2f}, Max = {max_value:.2f}")


# คำนวณค่าเฉลี่ยของ 'petal width' สำหรับแต่ละคลาส
mean_petal_width = df.groupby('class')['petal width'].mean()
min_petal_width = df.groupby('class')['petal width'].min()
max_petal_width = df.groupby('class')['petal width'].max()

# วาดกราฟแสดงค่าเฉลี่ยของ 'petal width' สำหรับแต่ละคลาส
plt.figure(figsize=(8, 5))
mean_petal_width.plot(kind='bar')
plt.title('Mean petal width for Each Class')
plt.xlabel('Class')
plt.ylabel('Mean petal width')
plt.xticks(ticks=range(len(mean_petal_width)), labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], rotation=45)
plt.show()

print("Statistics for Petal Width by Class:")
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # Class names for printing

# Iterate over each class index to print statistics
for i in range(len(mean_petal_width)):
    class_name = class_names[i]  # Get the class name based on index
    mean_value = mean_petal_width.iloc[i]  # Access the mean value by index
    min_value = min_petal_width.iloc[i]  # Access the min value by index
    max_value = max_petal_width.iloc[i]  # Access the max value by index
    print(f"{class_name}: Mean = {mean_value:.2f}, Min = {min_value:.2f}, Max = {max_value:.2f}")

import pandas as pd
import plotly.express as px

# Load the dataset
file_path = '/Users/natxpss/Downloads/Dtree/iris.data'
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df = pd.read_csv(file_path, header=None, names=columns)

# Define the categorization function
def categorize_sepal_length(length):
    if length <= 5.45:
        return 'sl_small'
    elif length <= 6.1:
        return 'sl_normal'
    else:  # length > 6.5
        return 'sl_big'

# Categorization6function for sepal width
def categorize_sepal_width(width):
    if width <= 2.9:
        return 'sw_small'
    elif width <= 3.3:
        return 'sw_normal'
    else:  # width > 3.3
        return 'sw_big'

# Categorization function for petal length
def categorize_petal_length(length):
    if length <= 2:
        return 'pl_small'
    elif length <= 5:
        return 'pl_normal'
    else:  # length > 5
        return 'pl_big'

# Categorization function for petal width
def categorize_petal_width(width):
    if width <= 0.6:
        return 'pw_small'
    elif width <= 1.6:
        return 'pw_normal'
    else:  # width > 1.6
        return 'pw_big'

# Apply the categorization functions to their respective columns
df['sepal length'] = df['sepal length'].apply(categorize_sepal_length)
df['sepal width'] = df['sepal width'].apply(categorize_sepal_width)
df['petal length'] = df['petal length'].apply(categorize_petal_length)
df['petal width'] = df['petal width'].apply(categorize_petal_width)