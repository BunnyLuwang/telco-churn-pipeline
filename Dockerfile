 
# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set system environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for compiling lightgbm/xgboost if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the Python dependencies directly inside the container environment
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all the remaining project folders into the container workdir
COPY src/ /app/src/
COPY config/ /app/config/

# Expose port 8501, which is the default port Streamlit uses
EXPOSE 8501

# Configure container healthchecks to monitor application stability
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the default command to launch your Streamlit application inside the container
ENTRYPOINT ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
