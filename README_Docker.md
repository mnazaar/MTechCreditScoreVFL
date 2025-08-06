# MTechCreditScoreVFL - Docker Setup

This document provides instructions for running the MTechCreditScoreVFL project using Docker.

## 🐳 Docker Setup

### Prerequisites

- Docker installed on your system
- Docker Compose installed
- At least 4GB of available RAM (for TensorFlow models)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd MTechCreditScoreVFL
   ```

2. **Set up environment variables** (optional):
   Create a `.env` file in the root directory:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
   Note: The OpenAI API key is only needed if you want natural language explanations.

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the applications**:
   - **Streamlit UI**: http://localhost:8501
   - **Flask API**: http://localhost:5001

### Alternative: Direct Docker Build

If you prefer to build the Docker image directly:

```bash
# Build the image
docker build -t mtech-credit-score-vfl .

# Run the container
docker run -p 5001:5001 -p 8501:8501 mtech-credit-score-vfl
```

## 🏗️ Architecture

The Docker setup runs two services:

1. **Flask API Server** (`app.py`)
   - Port: 5001
   - Provides credit score prediction endpoints
   - Handles feature explanations and model predictions

2. **Streamlit UI** (`credit_score_ui.py`)
   - Port: 8501
   - Provides a web interface for credit score predictions
   - Connects to the Flask API for data

## 📁 Volume Mounts

The Docker setup includes the following volume mounts for data persistence:

- `./logs` → `/app/logs` - Application logs
- `./plots` → `/app/plots` - Generated plots and visualizations
- `./saved_models` → `/app/saved_models` - Trained models
- `./data` → `/app/data` - Data files

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, for natural language explanations)
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered

### Ports

- **5001**: Flask API server
- **8501**: Streamlit web interface

## 🚀 Usage

1. **Start the services**:
   ```bash
   docker-compose up
   ```

2. **Access the Streamlit UI**:
   - Open your browser and go to http://localhost:8501
   - Enter a customer ID (e.g., "100-16-1590")
   - Click "Predict Credit Score"

3. **API Endpoints**:
   - Health check: `GET http://localhost:5001/health`
   - Credit score insights: `POST http://localhost:5001/credit-score/customer-insights`
   - Auto loan features: `POST http://localhost:5001/auto-loan/predict`
   - Credit card features: `POST http://localhost:5001/credit-card/predict`
   - Home loan features: `POST http://localhost:5001/home-loan/predict`
   - Digital savings features: `POST http://localhost:5001/digital-savings/predict`

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :5001
   netstat -tulpn | grep :8501
   
   # Kill the processes or change ports in docker-compose.yml
   ```

2. **Out of memory**:
   - Ensure you have at least 4GB RAM available
   - Increase Docker memory limit in Docker Desktop settings

3. **Model loading errors**:
   - Check that all model files are present in `VFLClientModels/saved_models/`
   - Ensure the data files are in the correct locations

4. **API connection errors**:
   - Wait for the Flask API to fully start (check logs)
   - Ensure the API is accessible at http://localhost:5001/health

### Viewing Logs

```bash
# View all logs
docker-compose logs

# View logs for specific service
docker-compose logs mtech-credit-score

# Follow logs in real-time
docker-compose logs -f
```

### Stopping Services

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v

# Stop and remove containers + images
docker-compose down --rmi all
```

## 🔄 Development

For development, you can mount the source code as a volume:

```yaml
# In docker-compose.yml, add:
volumes:
  - .:/app
```

This allows you to make changes to the code without rebuilding the container.

## 📊 Health Monitoring

The Docker setup includes health checks:

- **API Health**: `GET http://localhost:5001/health`
- **Container Health**: Docker will automatically restart the container if it becomes unhealthy

## 🎯 Performance Tips

1. **Use SSD storage** for better I/O performance
2. **Allocate sufficient RAM** (4GB+ recommended)
3. **Use Docker volumes** for persistent data
4. **Monitor resource usage** with `docker stats`

## 🔒 Security Notes

- The API is exposed on all interfaces (0.0.0.0)
- Consider using a reverse proxy for production deployments
- Keep your OpenAI API key secure
- Regularly update dependencies

## 📝 License

This project is part of the MTech Credit Score VFL research project. 