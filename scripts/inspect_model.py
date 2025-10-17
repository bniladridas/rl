import numpy as np

# Load the model file
data = np.load(
    "harpertoken-cartpole/model.npy",
    allow_pickle=True,
)

# Print information about the loaded data
print("Type:", type(data))
print("Shape:", data.shape if hasattr(data, "shape") else "No shape")

# Check if it's a dictionary (local save format)
if isinstance(data, dict):
    print("\nDictionary contents (local save format):")
    for key, value in data.items():
        print("Key:", key)
        print("Value type:", type(value))
        print("Value shape:", value.shape if hasattr(value, "shape") else "No shape")
        if key == "weights":
            print("Weights shape:", value.shape)
            print("Weights dtype:", value.dtype)
            print("Weights min:", value.min())
            print("Weights max:", value.max())

# If it's a numpy array (HF save format)
elif isinstance(data, np.ndarray):
    print("\nNumpy array contents (HF save format):")
    print("Array shape:", data.shape)
    print("Array dtype:", data.dtype)
    print("Array min:", data.min())
    print("Array max:", data.max())

# Print the actual data
print("\nData contents:")
print(data)
