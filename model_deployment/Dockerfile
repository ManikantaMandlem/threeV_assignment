FROM python:3.11.6-slim

ENV PORT=5050

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

ENTRYPOINT ["python", "app.py"]
CMD ["--port", "5050"]
