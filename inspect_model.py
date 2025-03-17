import numpy as np

# Load the model file
data = np.load("cartpole_cmaes/model.npy", allow_pickle=True)

# Print information about the loaded data
print("Type:", type(data))
print("Shape:", data.shape if hasattr(data, 'shape') else "No shape")

# If it's a numpy array with a dictionary
if isinstance(data, np.ndarray) and data.size == 1:
    try:
        dict_data = data.item()
        print("\nDictionary contents:")
        for key, value in dict_data.items():
            print(f"Key: {key}")
            print(f"Value type: {type(value)}")
            print(f"Value shape: {value.shape if hasattr(value, 'shape') else 'No shape'}")
    except:
        print("\nCould not convert to dictionary")

# Print the actual data
print("\nData contents:")
print(data)