# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and install the package
COPY pyproject.toml .
COPY harpertoken/ harpertoken/
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Default command: run a simple test
CMD ["python", "-c", "from harpertoken.models.model import CMAESAgent; agent = CMAESAgent.from_pretrained('harpertoken/harpertoken-cartpole'); print('Model loaded successfully!')"]
