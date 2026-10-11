FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
COPY . .
RUN python -m pip install --no-cache-dir -r requirements-validation.txt \
    && python -m pip install --no-cache-dir --no-deps .
CMD ["python", "scripts/verify-example.py", "--output", "/tmp/risk-eval-validation"]
