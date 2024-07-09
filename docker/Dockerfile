# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /opt

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /opt

WORKDIR /opt/app

# Expose the port that the app runs on
EXPOSE 3000

# Command to run the application
CMD ["python", "app.py"]
