# Use an official lightweight Python image
FROM python:3.12-slim


# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port the API will run on
EXPOSE 8080

# Command to run the API server
CMD ["python", "main.py"]
