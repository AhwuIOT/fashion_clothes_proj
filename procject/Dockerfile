FROM python:3.10-slim

WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
