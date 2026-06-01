FROM python:3.11-slim

WORKDIR /app

# Install torch CPU-only first.
# The default PyPI torch wheel bundles CUDA libraries and is ~2 GB.
# The CPU-only build from pytorch.org is ~300 MB — a significant size saving
# for a server that has no GPU and will never use CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source after the dependency layer so that code-only
# changes don't invalidate the pip cache layer.
COPY . .

# Non-root user for security — create after COPY so /app can be chowned in one layer.
RUN adduser --disabled-password --gecos "" --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
