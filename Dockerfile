FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY ml ./ml
COPY datasets ./datasets
COPY --from=frontend /build/dist ./frontend/dist
RUN useradd -m app && mkdir /app/data /app/artifacts && chown -R app:app /app/data /app/artifacts
USER app
EXPOSE 8940
CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","8940"]
