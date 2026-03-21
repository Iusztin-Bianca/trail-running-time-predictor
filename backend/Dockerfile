FROM python:3.11-slim

WORKDIR /project

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY backend/__init__.py ./backend/__init__.py
COPY backend/app ./backend/app

ENV PYTHONPATH=/project

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
