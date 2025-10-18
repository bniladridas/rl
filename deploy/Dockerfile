# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and install the package
COPY pyproject.toml .
COPY harpertoken/ harpertoken/
COPY README.md .

# Install the package and Flask
RUN pip install --no-cache-dir -e . flask

# Copy the app
COPY deploy/app.py .

# Default command: run the Flask app
CMD ["python", "app.py"]
