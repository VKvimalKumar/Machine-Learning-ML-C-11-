import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Create the Dataset (Only first 5 rows from the image)
data = {
    'Income':    [60,     75,       85.5,    52.8,      64.8],
    'Lawn_Size': [18.4,   19.6,     16.8,    20.8,      21.6],
    'Decision':  ['Owner','Nonowner','Owner','Nonowner','Owner']
}

# Convert to a DataFrame
df = pd.DataFrame(data)

print("Using this data for training:")
print(df)
print("-" * 30)

# 2. Preprocessing
# Convert 'Owner' to 1 and 'Nonowner' to 0
df['Target'] = df['Decision'].map({'Owner': 1, 'Nonowner': 0})

X = df[['Income', 'Lawn_Size']]
y = df['Target']

# 3. Train the Model
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 4. Plot the Tree
plt.figure(figsize=(8, 6), dpi=100)

plot_tree(clf, 
          feature_names=['Income', 'Lawn Size'],
          class_names=['Nonowner', 'Owner'], # 0=Nonowner, 1=Owner
          filled=True, 
          rounded=True)

plt.title("Decision Tree (First 5 Rows Only)")
plt.show()