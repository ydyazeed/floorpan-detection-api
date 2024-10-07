# Use an official Python image from Docker Hub
FROM python:3.6.13

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uninstall numpy, opencv-python, and mkl
RUN pip uninstall -y numpy opencv-python mkl

# Reinstall specific versions of numpy and opencv-python
RUN pip install opencv-python==3.4.2.17 numpy==1.14.5

# Preload EasyOCR models to avoid downloading during runtime
RUN python -c "import easyocr; easyocr.Reader(['en'])"

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


