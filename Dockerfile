# Use official TensorFlow 2.15 (CPU) image
FROM tensorflow/tensorflow:2.15.0

# Set a working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt ./

# Ensure pip is up-to-date and install requirements
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy the rest of the application code
COPY . /app

# Expose a port if the repo contains a web app (change or remove if not needed)
EXPOSE 8000

# Default command — run the project's main script. Users can override.
CMD ["python", "main.py"]
