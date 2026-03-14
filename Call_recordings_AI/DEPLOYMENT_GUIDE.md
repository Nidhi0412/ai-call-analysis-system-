# Deployment Guide - AI Call Analysis System

## 📋 Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Production Deployment Options](#production-deployment-options)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment (AWS)](#cloud-deployment-aws)
5. [Environment Configuration](#environment-configuration)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)

---

## 🖥️ Local Development Setup

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed
- 4GB+ RAM
- OpenAI API key

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Call_recordings_AI
```

#### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y ffmpeg python3-pip python3-venv
```

**macOS:**
```bash
brew install ffmpeg python3
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html
- Add FFmpeg to system PATH

#### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file
nano .env
```

Add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
NODE_ENV=Development
APP_PORT=8000
```

#### 6. Run the Application

```bash
# Option 1: Using the startup script
chmod +x start_local.sh
./start_local.sh

# Option 2: Manual start
python3 run_local.py
```

#### 7. Verify Installation

Open your browser and navigate to:
- **Web UI**: http://localhost:8000/innex/transcribe-ui
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

You should see the transcription interface.

---

## 🚀 Production Deployment Options

### Option 1: Traditional Server (Linux VM)

**Best for**: Small to medium deployments (100-500 calls/day)

**Pros**:
- Simple setup
- Full control
- Cost-effective

**Cons**:
- Manual scaling
- Requires server management

### Option 2: Docker Container

**Best for**: Consistent deployments across environments

**Pros**:
- Reproducible builds
- Easy deployment
- Isolated environment

**Cons**:
- Requires Docker knowledge
- Slightly more complex

### Option 3: Cloud Platform (AWS/GCP/Azure)

**Best for**: Large-scale deployments (1000+ calls/day)

**Pros**:
- Auto-scaling
- High availability
- Managed services

**Cons**:
- Higher cost
- Cloud platform lock-in

---

## 🐳 Docker Deployment

### Create Dockerfile

Create a file named `Dockerfile` in the project root:

```dockerfile
# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads cache results logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=Production

# Run the application
CMD ["python3", "run_local.py"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  call-analysis:
    build: .
    container_name: call-analysis-system
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NODE_ENV=Production
    volumes:
      - ./uploads:/app/uploads
      - ./cache:/app/cache
      - ./results:/app/results
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
# Build the Docker image
docker build -t call-analysis-system .

# Run using docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Docker Deployment Best Practices

1. **Use .dockerignore**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
.env
.cache/
uploads/
results/
logs/
*.log
```

2. **Multi-stage builds** (for smaller images):
```dockerfile
# Build stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python3", "run_local.py"]
```

---

## ☁️ Cloud Deployment (AWS)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   AWS Cloud                          │
│                                                      │
│  ┌──────────────┐      ┌──────────────┐            │
│  │ Application  │      │  Application │            │
│  │ Load Balancer│─────▶│  Load Balancer│           │
│  │   (ALB)      │      │   (ALB)       │           │
│  └──────┬───────┘      └──────┬────────┘           │
│         │                     │                     │
│         ▼                     ▼                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │   ECS Task   │      │   ECS Task   │            │
│  │  (Container) │      │  (Container) │            │
│  └──────┬───────┘      └──────┬────────┘           │
│         │                     │                     │
│         ▼                     ▼                     │
│  ┌─────────────────────────────────┐               │
│  │         S3 Bucket                │               │
│  │  (Audio Files & Results)         │               │
│  └─────────────────────────────────┘               │
│                                                      │
│  ┌─────────────────────────────────┐               │
│  │      CloudWatch Logs             │               │
│  │  (Application Logs & Metrics)    │               │
│  └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### AWS ECS Deployment

#### 1. Create ECR Repository

```bash
# Create ECR repository
aws ecr create-repository --repository-name call-analysis-system

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t call-analysis-system .
docker tag call-analysis-system:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/call-analysis-system:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/call-analysis-system:latest
```

#### 2. Create ECS Task Definition

Create `task-definition.json`:

```json
{
  "family": "call-analysis-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "call-analysis-container",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/call-analysis-system:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NODE_ENV",
          "value": "Production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/call-analysis",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 3. Create ECS Service

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS cluster
aws ecs create-cluster --cluster-name call-analysis-cluster

# Create ECS service
aws ecs create-service \
  --cluster call-analysis-cluster \
  --service-name call-analysis-service \
  --task-definition call-analysis-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### 4. Configure Auto-scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/call-analysis-cluster/call-analysis-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/call-analysis-cluster/call-analysis-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

`scaling-policy.json`:
```json
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleInCooldown": 300,
  "ScaleOutCooldown": 60
}
```

### AWS Lambda Deployment (Serverless)

**Note**: Lambda has 15-minute timeout limit, suitable for short calls only.

#### 1. Create Lambda Function

```python
# lambda_handler.py
import json
import base64
from web_ui import process_audio

def lambda_handler(event, context):
    # Decode audio file from base64
    audio_data = base64.b64decode(event['body'])
    
    # Process audio
    result = process_audio(audio_data)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

#### 2. Package and Deploy

```bash
# Install dependencies in a folder
pip install -r requirements.txt -t package/

# Copy application code
cp -r *.py package/

# Create deployment package
cd package
zip -r ../lambda-deployment.zip .
cd ..

# Deploy to Lambda
aws lambda create-function \
  --function-name call-analysis-function \
  --runtime python3.10 \
  --role arn:aws:iam::<account-id>:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda-deployment.zip \
  --timeout 900 \
  --memory-size 2048 \
  --environment Variables={OPENAI_API_KEY=your-key}
```

---

## ⚙️ Environment Configuration

### Development Environment

```env
# .env.development
NODE_ENV=Development
APP_HOST=0.0.0.0
APP_PORT=8000
WORKERS=1
DEBUG=True
LOG_LEVEL=debug

# API Keys
OPENAI_API_KEY=sk-your-dev-key

# Optional
GROQ_API_KEY=
DEEPGRAM_API_KEY=
```

### Production Environment

```env
# .env.production
NODE_ENV=Production
APP_HOST=0.0.0.0
APP_PORT=8000
WORKERS=4
DEBUG=False
LOG_LEVEL=info

# API Keys (use secrets manager in production)
OPENAI_API_KEY=sk-your-prod-key

# Security
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CORS_ORIGINS=https://yourdomain.com

# Performance
MAX_UPLOAD_SIZE=100MB
CACHE_TTL=86400
REQUEST_TIMEOUT=300
```

### Environment Variable Management

#### AWS Secrets Manager

```bash
# Store secret
aws secretsmanager create-secret \
  --name openai-api-key \
  --secret-string "sk-your-api-key"

# Retrieve secret in application
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

OPENAI_API_KEY = get_secret('openai-api-key')
```

#### Docker Secrets

```bash
# Create secret
echo "sk-your-api-key" | docker secret create openai_api_key -

# Use in docker-compose.yml
services:
  app:
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    external: true
```

---

## 📊 Monitoring & Logging

### Application Logging

```python
# Add to run_local.py
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
```

### Prometheus Metrics

```python
# Add to web_ui.py
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
transcription_requests = Counter(
    'transcription_requests_total',
    'Total transcription requests'
)

transcription_duration = Histogram(
    'transcription_duration_seconds',
    'Transcription duration in seconds'
)

@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### CloudWatch Logs (AWS)

```python
# Add CloudWatch logging
import watchtower
import logging

logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler(
    log_group='/aws/ecs/call-analysis',
    stream_name='application'
))
```

### Health Check Endpoint

```python
# Add to web_ui.py
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {
            "openai_api": check_openai_connection(),
            "disk_space": check_disk_space(),
            "memory": check_memory_usage()
        }
    }
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or change port in run_local.py
```

#### 2. FFmpeg Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

#### 3. OpenAI API Key Error

**Error**: `AuthenticationError: Incorrect API key provided`

**Solution**:
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Ensure no extra spaces or quotes
OPENAI_API_KEY=sk-your-key-here

# Restart application
```

#### 4. Out of Memory

**Error**: `MemoryError` or container killed

**Solution**:
```bash
# Reduce max_workers in optimized_transcription_service.py
max_workers = 2  # Instead of 4

# Increase Docker memory limit
docker run -m 4g call-analysis-system

# Or increase ECS task memory
"memory": "4096"  # 4GB
```

#### 5. Slow Processing

**Issue**: Processing takes too long

**Solution**:
```python
# Enable caching
clear_cache = False

# Use parallel processing
use_advanced_diarization = False  # Faster but less accurate

# Reduce audio quality
sample_rate = 8000  # Instead of 16000 (lower quality)
```

### Debugging Tips

#### Enable Debug Logging

```python
# In run_local.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check API Usage

```bash
# Monitor OpenAI API usage
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### Test Components Individually

```python
# Test transcription only
from transcription_with_speakers import TranscriptionWithSpeakersService

service = TranscriptionWithSpeakersService()
result = service.transcribe_audio("test.wav")
print(result)
```

---

## 📈 Performance Tuning

### Optimize for Speed

```python
# In optimized_transcription_service.py
max_workers = 8  # Increase parallel workers
cache_ttl = 3600  # 1 hour cache

# Use smaller Whisper model
model = "base"  # Instead of "large-v2"
```

### Optimize for Accuracy

```python
# Use larger Whisper model
model = "large-v3"

# Enable advanced diarization
use_advanced_diarization = True

# Increase preprocessing
noise_reduction_strength = 0.9
```

### Optimize for Cost

```python
# Enable aggressive caching
cache_ttl = 86400  # 24 hours

# Use smaller models
whisper_model = "base"
gpt_model = "gpt-3.5-turbo"

# Batch processing
batch_size = 10
```

---

## 🔒 Security Checklist

- [ ] API keys stored in environment variables or secrets manager
- [ ] HTTPS enabled for production
- [ ] CORS configured properly
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] File upload size limits enforced
- [ ] Automatic cleanup of uploaded files
- [ ] Logging of all API requests
- [ ] Regular security updates
- [ ] Firewall rules configured

---

## 📝 Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] API keys validated
- [ ] Dependencies up to date
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Documentation updated

### Deployment

- [ ] Backup current version
- [ ] Deploy new version
- [ ] Run health checks
- [ ] Monitor logs for errors
- [ ] Test critical functionality
- [ ] Update DNS if needed
- [ ] Notify team of deployment

### Post-Deployment

- [ ] Monitor metrics for 24 hours
- [ ] Check error rates
- [ ] Verify performance
- [ ] Collect user feedback
- [ ] Document any issues
- [ ] Plan rollback if needed

---

## 🆘 Support & Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS ECS Guide](https://docs.aws.amazon.com/ecs/)

### Community
- GitHub Issues: [Your Repo Issues]
- Stack Overflow: Tag `call-analysis-system`
- Discord/Slack: [Your Community Link]

### Contact
- Email: [your-email@example.com]
- GitHub: [your-github-username]

---

**Last Updated**: March 2026  
**Version**: 1.0.0
