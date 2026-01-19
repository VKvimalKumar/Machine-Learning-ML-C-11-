import math
import random

# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC DATA
# ---------------------------------------------------------
# We will create a list of data points.
# Each point represents a person: [Age, Loan Amount, Defaulter (0 or 1)]
# Defaulter: 0 = No, 1 = Yes

def generate_data(num_samples=20):
    data = []
    for _ in range(num_samples):
        # Random Age between 20 and 60
        age = random.randint(20, 60)
        
        # Random Loan between 10,000 and 100,000
        loan = random.randint(10000, 100000)
        
        # Simple rule to assign label for our synthetic data (to make it separable)
        # Check if Age < 40 and Loan > 50000 might be higher risk (just a hypothesis)
        if loan > 60000 or (age < 30 and loan > 40000):
            defaulter = 1 # Yes
        else:
            defaulter = 0 # No
            
        data.append([age, loan, defaulter])
    return data

# ---------------------------------------------------------
# 2. ALGORITHM FROM SCRATCH (K-Nearest Neighbors)
# ---------------------------------------------------------
# Since the goal is "Binary Classification" based on creating groups/proximity,
# K-Nearest Neighbors (KNN) is the standard "from scratch" approach.
# It finds the k closest data points to label the new one.

def euclidean_distance(point1, point2):
    # point1 = (Age, Loan), point2 = (Age, Loan)
    distance_squared = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    return math.sqrt(distance_squared)

def classify_person(dataset, new_age, new_loan, k=3):
    distances = []
    
    # Calculate distance from the new point to every point in the dataset
    for row in dataset:
        existing_age = row[0]
        existing_loan = row[1]
        label = row[2]
        
        dist = euclidean_distance((new_age, new_loan), (existing_age, existing_loan))
        distances.append((dist, label))
    
    # Sort by distance (smallest first)
    distances.sort(key=lambda x: x[0])
    
    # Get the top k nearest neighbors
    k_nearest = distances[:k]
    
    # Count votes
    votes_for_defaulter = 0
    votes_for_non_defaulter = 0
    
    print(f"\n--- Finding {k} Nearest Neighbors for Age: {new_age}, Loan: {new_loan} ---")
    for i, (dist, label) in enumerate(k_nearest):
        label_str = "Defaulter" if label == 1 else "Non-Defaulter"
        print(f"Neighbor {i+1}: Distance={dist:.2f}, Status={label_str}")
        
        if label == 1:
            votes_for_defaulter += 1
        else:
            votes_for_non_defaulter += 1
            
    # Majority Rule
    if votes_for_defaulter > votes_for_non_defaulter:
        return "Defaulter"
    else:
        return "Non-Defaulter"

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Generate Data
    dataset = generate_data(20)
    
    print("Synthetic Training Data (First 5 rows):")
    print(f"{'Age':<10} {'Loan':<15} {'Defaulter (1=Yes, 0=No)'}")
    print("-" * 45)
    for row in dataset[:5]:
        print(f"{row[0]:<10} {row[1]:<15} {row[2]}")
    print("... (total 20 rows)")

    # Input for Prediction
    test_age = 25
    test_loan = 75000
    
    # Perform Classification
    k_value = 3
    result = classify_person(dataset, test_age, test_loan, k=k_value)
    
    print("\n" + "="*40)
    print(f"PREDICTION RESULT: The person is likely a {result}")
    print("="*40)