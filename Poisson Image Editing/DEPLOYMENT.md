# 🚀 Deployment Guide - Enhanced Poisson Image Editor

This guide provides comprehensive instructions for deploying the Enhanced Poisson Image Editor to various platforms.

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized deployment)

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended for GitHub)

**Streamlit Cloud** provides the easiest deployment directly from GitHub.

#### Steps:
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Enhanced Poisson Image Editor"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Connect your GitHub repository
   - Select this repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration**:
   - Streamlit will automatically install dependencies from `requirements.txt` and `streamlit_requirements.txt`
   - Your app will be available at: `https://[your-app-name].streamlit.app/`

#### Streamlit Cloud Features:
- ✅ Automatic HTTPS
- ✅ Automatic updates from GitHub
- ✅ Free tier available
- ✅ Built-in secrets management
- ✅ Easy custom domain setup

---

### 2. Heroku Deployment

**Heroku** provides robust cloud hosting with easy scaling.

#### Setup Files Required:

**Create `Procfile`**:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**Create `runtime.txt`**:
```
python-3.9.16
```

**Create `setup.sh`**:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

#### Deployment Steps:
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create new app
heroku create your-poisson-editor

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

---

### 3. Docker Deployment

**Docker** provides consistent deployment across any platform.

#### Build and Run:
```bash
# Build Docker image
docker build -t poisson-editor:latest .

# Run container
docker run -p 8501:8501 poisson-editor:latest

# Access app at http://localhost:8501
```

#### Docker Compose (with volume mounts):
**Create `docker-compose.yml`**:
```yaml
version: '3.8'
services:
  poisson-editor:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./pics:/app/pics
      - ./uploads:/app/uploads
      - ./results:/app/results
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

```bash
# Run with Docker Compose
docker-compose up -d
```

#### Deploy to Docker Hub:
```bash
# Tag image
docker tag poisson-editor:latest yourusername/poisson-editor:latest

# Push to Docker Hub
docker push yourusername/poisson-editor:latest
```

---

### 4. Google Cloud Platform (GCP)

#### Cloud Run Deployment:
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy poisson-editor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### App Engine Deployment:
**Create `app.yaml`**:
```yaml
runtime: python39

entrypoint: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

automatic_scaling:
  max_instances: 5
  min_instances: 0
  target_cpu_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 4

env_variables:
  STREAMLIT_SERVER_PORT: 8080
```

```bash
# Deploy to App Engine
gcloud app deploy
```

---

### 5. AWS Deployment

#### EC2 Instance:
```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-instance

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Clone repository
git clone https://github.com/yourusername/poisson-image-editor.git
cd poisson-image-editor

# Install dependencies
pip3 install -r requirements.txt
pip3 install -r streamlit_requirements.txt

# Run with screen (persistent session)
screen -S poisson-editor
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Detach from screen: Ctrl+A, then D
# Reattach to screen: screen -r poisson-editor
```

#### Elastic Beanstalk:
**Create `.ebextensions/01_packages.config`**:
```yaml
packages:
  yum:
    git: []
    gcc: []
    
option_settings:
  aws:elasticbeanstalk:application:environment:
    STREAMLIT_SERVER_PORT: 8080
  aws:elasticbeanstalk:container:python:
    WSGIPath: application.py
```

**Create `application.py`** (EB entry point):
```python
import subprocess
import os

def run_streamlit():
    port = int(os.environ.get('PORT', 8080))
    subprocess.run([
        'streamlit', 'run', 'app.py', 
        '--server.port', str(port),
        '--server.address', '0.0.0.0'
    ])

if __name__ == '__main__':
    run_streamlit()
```

---

### 6. Azure Container Instances

```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name poisson-editor-rg --location eastus

# Create container instance
az container create \
  --resource-group poisson-editor-rg \
  --name poisson-editor \
  --image yourusername/poisson-editor:latest \
  --ports 8501 \
  --ip-address public \
  --memory 4 \
  --cpu 2
```

---

## 🔧 Configuration Options

### Environment Variables

Set these environment variables for different deployment scenarios:

```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Application settings
MAX_UPLOAD_SIZE=200
ENABLE_CUDA=false
DEBUG_MODE=false

# Performance settings
MAX_WORKERS=4
MEMORY_LIMIT=4096
PROCESSING_TIMEOUT=300
```

### Custom Domain Setup

#### Streamlit Cloud:
1. Go to your app settings
2. Add custom domain in "Custom subdomains"
3. Update DNS CNAME record

#### Heroku:
```bash
# Add custom domain
heroku domains:add yourdomain.com

# Configure DNS
# Add CNAME record pointing to your-app.herokuapp.com
```

#### Other Platforms:
- Configure reverse proxy (nginx) for custom domains
- Set up SSL certificates (Let's Encrypt recommended)

---

## 📊 Performance Optimization

### For Production Deployment:

1. **Enable Caching**:
   ```python
   @st.cache_data
   def load_large_data():
       # Cache expensive operations
       pass
   ```

2. **Resource Limits**:
   - Set appropriate memory limits
   - Configure CPU limits
   - Enable auto-scaling

3. **CDN Setup**:
   - Use CloudFlare or similar for static assets
   - Enable image optimization

4. **Monitoring**:
   - Set up health checks
   - Configure logging
   - Monitor resource usage

---

## 🛡️ Security Considerations

### Production Security:

1. **File Upload Security**:
   - Validate file types
   - Limit file sizes
   - Scan for malware

2. **Input Validation**:
   - Sanitize user inputs
   - Validate image formats
   - Check parameter ranges

3. **Rate Limiting**:
   - Implement request rate limiting
   - Add processing timeouts
   - Monitor resource usage

4. **HTTPS**:
   - Always use HTTPS in production
   - Set up proper SSL certificates
   - Configure security headers

---

## 🔍 Monitoring & Maintenance

### Health Checks:
```python
# Add to app.py for health check endpoint
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

### Logging Configuration:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Backup Strategy:
- Regular backups of user-generated content
- Version control for code updates
- Database backups (if applicable)

---

## 🚀 Quick Start Commands

Choose your preferred deployment method:

```bash
# Streamlit Cloud (GitHub)
git push origin main
# Then visit share.streamlit.io

# Local Development
streamlit run app.py

# Docker Quick Start
docker build -t poisson-editor . && docker run -p 8501:8501 poisson-editor

# Heroku Quick Deploy
heroku create your-app-name && git push heroku main
```

---

## 📞 Support & Troubleshooting

### Common Issues:

1. **Memory Errors**:
   - Increase container memory limits
   - Enable processing optimization
   - Use smaller test images

2. **Import Errors**:
   - Verify all dependencies in requirements.txt
   - Check Python version compatibility
   - Ensure proper virtual environment

3. **Performance Issues**:
   - Enable fast mode for testing
   - Reduce image sizes
   - Use appropriate instance types

### Getting Help:
- Check application logs
- Review deployment platform documentation
- Test locally before deploying
- Monitor resource usage

---

**🎉 Congratulations! Your Enhanced Poisson Image Editor is now ready for professional deployment!**
