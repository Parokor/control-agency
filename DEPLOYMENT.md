# Deployment Guide

This document provides detailed instructions for deploying the Federated AI System, including both frontend and backend components.

## Frontend Deployment

### Option 1: GitHub Pages

```bash
# Clone the repository if you haven't already
git clone https://github.com/Parokor/control-agency.git
cd control-agency

# Install dependencies
cd frontend
npm install

# Build the application
npm run build

# Deploy to GitHub Pages
# First, make sure you have gh-pages package installed
npm install --save-dev gh-pages

# Add these scripts to your package.json:
# "predeploy": "npm run build",
# "deploy": "gh-pages -d dist"

# Then run the deploy command
npm run deploy
```

### Option 2: Cloudflare Pages

```bash
# Clone the repository if you haven't already
git clone https://github.com/Parokor/control-agency.git
cd control-agency

# Install dependencies
cd frontend
npm install

# Build the application
npm run build

# Install Wrangler CLI if you haven't already
npm install -g wrangler

# Login to Cloudflare (you'll need a Cloudflare account)
wrangler login

# Deploy to Cloudflare Pages
wrangler pages publish dist
```

## Backend Deployment

### Option 1: Replit Deployment

1. Create a new Replit project
   - Go to [Replit](https://replit.com) and sign up/login
   - Click "+ Create Repl"
   - Choose "Import from GitHub"
   - Enter the repository URL: https://github.com/Parokor/control-agency.git
   - Select "Python" as the language

2. Configure the Replit project
   ```bash
   # In the Replit shell, navigate to the backend directory
   cd backend
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables in the Replit secrets panel
   # Go to Tools > Secrets and add the following:
   # - SUPABASE_URL: Your Supabase URL
   # - SUPABASE_KEY: Your Supabase API key
   # - JWT_SECRET: A secure random string for JWT signing
   ```

3. Configure the run command
   - Open the `.replit` file and set the run command to:
   ```
   run = "cd backend && uvicorn main:app --host 0.0.0.0 --port 8080"
   ```

4. Enable Always On feature (optional, requires Replit Hacker Plan)
   - Go to your Repl settings
   - Find the "Always On" toggle and enable it

### Option 2: Local Deployment

```bash
# Clone the repository if you haven't already
git clone https://github.com/Parokor/control-agency.git
cd control-agency

# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your environment variables
echo "SUPABASE_URL=your_supabase_url" > .env
echo "SUPABASE_KEY=your_supabase_key" >> .env
echo "JWT_SECRET=your_jwt_secret" >> .env

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Database Setup

1. Create a Supabase account at [Supabase](https://supabase.com)
2. Create a new project
3. Get your API URL and anon key from the project settings
4. Run the database initialization script:

```bash
# Navigate to the scripts directory
cd control-agency/scripts

# Run the database initialization script
python init_database.py --url YOUR_SUPABASE_URL --key YOUR_SUPABASE_KEY
```

## Specialized Containers Deployment

### Chat Container

```bash
# Navigate to the chat container directory
cd control-agency/containers/chat_container

# Build the Docker image
docker build -t control-agency/chat-container .

# Run the container
docker run -d -p 8000:8000 --env-file ../../.env control-agency/chat-container
```

### Development Container

```bash
# Navigate to the development container directory
cd control-agency/containers/dev_container

# Build the Docker image
docker build -t control-agency/dev-container .

# Run the container
docker run -d -p 8001:8000 --env-file ../../.env control-agency/dev-container
```

### Media Container

```bash
# Navigate to the media container directory
cd control-agency/containers/media_container

# Build the Docker image
docker build -t control-agency/media-container .

# Run the container
# Note: This requires NVIDIA Docker for GPU support
docker run -d -p 8188:8188 --gpus all control-agency/media-container
```

## Verifying Deployment

After deployment, verify that all components are working correctly:

1. Frontend: Visit your deployed frontend URL (GitHub Pages or Cloudflare Pages)
2. Backend API: Test the API endpoints
   ```bash
   curl https://your-backend-url/health
   ```
3. Containers: Check that all containers are running
   ```bash
   docker ps
   ```

## Troubleshooting

### Frontend Build Issues
- Check Node.js version (v16+ recommended)
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

### Backend Deployment Issues
- Check Python version (3.8+ recommended)
- Verify all environment variables are set correctly
- Check Replit logs for any errors

### Container Issues
- Verify Docker is installed and running
- Check container logs: `docker logs [container_id]`
- For GPU containers, verify NVIDIA Docker is installed correctly
