FROM python:3.10.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/logs /app/plots /app/saved_models

RUN echo '#!/bin/bash\n\
echo "Starting MTechCreditScoreVFL services..."\n\
\n\
echo "Starting Flask API server on port 5001..."\n\
python VFLClientModels/models/apis/app.py &\n\
API_PID=$!\n\
\n\
# Wait for API server to start up\n\
sleep 5\n\
\n\
echo "Starting Streamlit UI..."\n\
streamlit run VFLClientModels/models/UI/credit_score_ui.py --server.port 8501 --server.address 0.0.0.0\n\
\n\
# Clean up API process on exit\n\
kill $API_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 5001 8501
CMD ["/app/start.sh"]