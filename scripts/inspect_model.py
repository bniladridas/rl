import numpy as np

# Load the model file
data = np.load(
    "harpertoken-cartpole/model.npy",
    allow_pickle=True,
)

# Print information about the loaded data
print("Type:", type(data))
print(
    "Shape:",
    data.shape
    if hasattr(data, "shape")
    else "No shape",
)

# Check if it's a dictionary (local save format)
if isinstance(data, dict):
    print(
        "\nDictionary contents (local save format):"
    )
    for key, value in data.items():
        print(f"Key: {key}")
        print(
            f"Value type: {type(value)}"
        )
        print(
            f"Value shape: {value.shape if hasattr(value, 'shape') else 'No shape'}"
        )
        if key == "weights":
            print(
                f"Weights shape: {value.shape}"
            )
            print(
                f"Weights dtype: {value.dtype}"
            )
            print(
                f"Weights min: {value.min()}, max: {value.max()}"
            )

# If it's a numpy array (HF save format)
elif isinstance(data, np.ndarray):
    print(
        "\nNumpy array contents (HF save format):"
    )
    print(f"Array shape: {data.shape}")
    print(f"Array dtype: {data.dtype}")
    print(
        f"Array min: {data.min()}, max: {data.max()}"
    )

# Print the actual data
print("\nData contents:")
print(data)
