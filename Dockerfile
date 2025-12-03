# Dockerfile mínimo
FROM python:3.9-slim
WORKDIR /app
COPY app_credit_scoring.py .
COPY requirements.txt .
COPY results/ results/
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app_credit_scoring.py"]