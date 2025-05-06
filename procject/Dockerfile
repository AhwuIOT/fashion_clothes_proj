FROM python:3.10-slim

WORKDIR /app

# Copy app files
COPY . .

#Install system library
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
