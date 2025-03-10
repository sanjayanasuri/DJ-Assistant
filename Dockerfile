FROM python:3.9
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
CMD ["python", "flask_api.py"]
