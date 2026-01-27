# Stage 1: Builder
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

# Install dependencies efficiently (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runner (Slimmer final image, though PyTorch is dominant size)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder (optional if using same base, mainly for logical separation)
# In this simple case, we just installed in base. 
# But let's copy source code.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

# Optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "src/app.py"]