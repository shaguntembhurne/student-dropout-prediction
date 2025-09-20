# Stage 1: Define the base environment
# Use an official Python runtime as a parent image. This gives us a clean OS with Python 3.12 pre-installed.
FROM python:3.12-slim

# Stage 2: Set up the working environment inside the container
WORKDIR /app

# Stage 3: Copy over ALL project files
# This is a more robust way to ensure all necessary files, including requirements.txt,
# are copied into the container's working directory.
COPY . .

# Stage 4: Install the dependencies using a more robust method
# Using 'python -m pip' ensures we use the pip associated with our main python executable,
# which helps avoid PATH issues inside the container.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Stage 5: Define the command to run the application
# This is the command that will be executed when the container starts.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
