import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Create the Dataset from the image
# I have manually transcribed the 24 rows visible in the slide.
data = {
    'Income': [
        60, 75, 85.5, 52.8, 64.8, 64.8, 61.5, 43.2, 87, 84, 
        110.1, 49.2, 108, 59.2, 82.8, 66, 69, 47.4, 93, 33, 
        51, 51, 81, 63
    ],
    'Lawn_Size': [
        18.4, 19.6, 16.8, 20.8, 21.6, 17.2, 20.8, 20.4, 23.6, 17.6, 
        19.2, 17.6, 17.6, 16.0, 22.4, 18.4, 20.0, 16.4, 20.0, 18.8, 
        22.0, 14.0, 20.0, 14.8
    ],
    'Decision': [
        'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner',
        'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner',
        'Owner', 'Nonowner', 'Owner', 'Nonowner'
    ]
}

df = pd.DataFrame(data)

# 2. Preprocessing
# Scikit-learn requires numbers, so we convert 'Owner'/'Nonowner' to 1/0
# Map: Owner = 1, Nonowner = 0
df['Target'] = df['Decision'].map({'Owner': 1, 'Nonowner': 0})

features = ['Income', 'Lawn_Size']
X = df[features]
y = df['Target']

# 3. Train the Model
# We train on the entire dataset to visualize the logic for these specific 24 records
clf = DecisionTreeClassifier(criterion='entropy', random_state=0) 
clf.fit(X, y)

# 4. Plot the Tree
plt.figure(figsize=(14, 10), dpi=100) # Large figure size for readability

plot_tree(clf, 
          feature_names=features,
          class_names=['Nonowner', 'Owner'], # 0 is Nonowner, 1 is Owner
          filled=True, 
          rounded=True,
          fontsize=10)

plt.title("Decision Tree for Lawn Mower Ownership")
plt.show()