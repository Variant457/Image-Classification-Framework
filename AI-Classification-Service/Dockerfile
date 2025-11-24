FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY api_service.py ./
COPY model.py ./
COPY /models ./models

ENV PORT=8000
EXPOSE 8000

CMD ["python", "api_service.py"]