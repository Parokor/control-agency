# Deployment Guide

This document provides comprehensive, step-by-step instructions for deploying the Federated AI System, including both frontend and backend components. Follow these instructions carefully to ensure a successful deployment on your first attempt.

## Prerequisites

Before beginning deployment, ensure you have the following installed and configured:

1. **Git** - [Download and install Git](https://git-scm.com/downloads)
2. **Node.js** (v16.0.0 or higher) - [Download and install Node.js](https://nodejs.org/)
3. **Python** (v3.8.0 or higher) - [Download and install Python](https://www.python.org/downloads/)
4. **Docker** (for container deployment) - [Download and install Docker](https://www.docker.com/products/docker-desktop)
5. **NVIDIA Docker** (for GPU containers) - [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify installations with the following commands:
```bash
# Check Git version
git --version

# Check Node.js version
node --version
npm --version

# Check Python version
python --version
pip --version

# Check Docker version
docker --version

# Check NVIDIA Docker (if applicable)
nvidia-smi
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Deployment Checklist

Use this checklist to track your progress through the deployment process:

- [ ] Clone the repository
- [ ] Set up frontend
- [ ] Deploy frontend
- [ ] Set up backend
- [ ] Configure environment variables
- [ ] Deploy backend
- [ ] Set up database
- [ ] Deploy specialized containers
- [ ] Verify deployment
- [ ] Configure system integration

## Frontend Deployment

The frontend can be deployed using either GitHub Pages (recommended for simplicity) or Cloudflare Pages (recommended for performance). Choose the option that best suits your needs.

### Option 1: GitHub Pages

GitHub Pages provides free hosting directly from your GitHub repository. Follow these steps carefully:

#### Step 1: Clone the Repository

```bash
# Open your terminal and navigate to your preferred directory
cd ~/projects  # or any directory of your choice

# Clone the repository
git clone https://github.com/Parokor/control-agency.git

# Navigate to the project directory
cd control-agency
```

#### Step 2: Set Up the Frontend Environment

```bash
# Navigate to the frontend directory
cd frontend

# Install all dependencies (this may take a few minutes)
npm install

# If you encounter any errors, try with legacy peer dependencies
# npm install --legacy-peer-deps
```

#### Step 3: Configure GitHub Pages Deployment

```bash
# Install the gh-pages package for deployment
npm install --save-dev gh-pages
```

Now, open the `package.json` file in the frontend directory and add the following scripts:

```json
"scripts": {
  // ... existing scripts
  "predeploy": "npm run build",
  "deploy": "gh-pages -d dist"
}
```

Also, add the homepage field to your package.json (replace USERNAME with your GitHub username):

```json
"homepage": "https://Parokor.github.io/control-agency"
```

#### Step 4: Build and Deploy

```bash
# Build the application for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

#### Step 5: Configure GitHub Repository Settings

1. Go to your GitHub repository at https://github.com/Parokor/control-agency
2. Click on "Settings" tab
3. Navigate to "Pages" in the left sidebar
4. Under "Source", select "gh-pages" branch
5. Click "Save"

#### Step 6: Verify Deployment

After a few minutes, your site will be available at:
https://Parokor.github.io/control-agency

If you encounter a 404 error, wait a few more minutes as GitHub Pages deployment can take up to 10 minutes to propagate.

### Option 2: Cloudflare Pages

Cloudflare Pages offers enhanced performance with a global CDN and automatic HTTPS. Follow these steps for deployment:

#### Step 1: Clone the Repository

```bash
# Open your terminal and navigate to your preferred directory
cd ~/projects  # or any directory of your choice

# Clone the repository
git clone https://github.com/Parokor/control-agency.git

# Navigate to the project directory
cd control-agency
```

#### Step 2: Set Up the Frontend Environment

```bash
# Navigate to the frontend directory
cd frontend

# Install all dependencies (this may take a few minutes)
npm install

# If you encounter any errors, try with legacy peer dependencies
# npm install --legacy-peer-deps
```

#### Step 3: Build the Application

```bash
# Build the application for production
npm run build
```

This will create a `dist` directory with the built application.

#### Step 4: Install and Configure Wrangler CLI

```bash
# Install Wrangler CLI globally
npm install -g wrangler

# Verify installation
wrangler --version
```

#### Step 5: Create a Cloudflare Account

1. Go to [Cloudflare](https://dash.cloudflare.com/sign-up) and create an account if you don't have one
2. Verify your email address
3. Navigate to the Cloudflare Dashboard

#### Step 6: Authenticate with Cloudflare

```bash
# Login to Cloudflare (this will open a browser window)
wrangler login
```

Follow the prompts in the browser to authenticate.

#### Step 7: Deploy to Cloudflare Pages

```bash
# Deploy the application
wrangler pages publish dist
```

During the deployment process, you'll be prompted to:
1. Name your project (e.g., "control-agency")
2. Select a production branch (typically "main")

#### Step 8: Configure Build Settings (Alternative Method)

Alternatively, you can deploy directly from GitHub:

1. Go to the [Cloudflare Pages dashboard](https://dash.cloudflare.com/?to=/:account/pages)
2. Click "Create a project"
3. Connect your GitHub account and select the repository
4. Configure the build settings:
   - Build command: `npm run build`
   - Build output directory: `dist`
   - Root directory: `/frontend`
5. Click "Save and Deploy"

#### Step 9: Verify Deployment

After deployment completes, Cloudflare will provide a URL where your site is accessible (typically `https://your-project-name.pages.dev`).

You can also set up a custom domain in the Cloudflare Pages dashboard if desired.

## Backend Deployment

The backend can be deployed using either Replit (recommended for simplicity and free tier) or locally on your own server. Choose the option that best suits your needs.

### Option 1: Replit Deployment

Replit provides a simple, free hosting solution for the backend with minimal configuration. Follow these steps carefully:

#### Step 1: Create a Replit Account

1. Go to [Replit](https://replit.com) and sign up for an account if you don't have one
2. Verify your email address
3. Complete the onboarding process

#### Step 2: Create a New Repl

1. From the Replit dashboard, click the "+ Create" button
2. Select the "Import from GitHub" option
3. Enter the repository URL: `https://github.com/Parokor/control-agency.git`
4. In the "Language" dropdown, select "Python"
5. Give your Repl a name (e.g., "control-agency-backend")
6. Click "Import from GitHub"

#### Step 3: Configure the Backend

Once the repository is imported, you'll need to configure the backend:

```bash
# In the Replit shell, navigate to the backend directory
cd backend

# Install dependencies (this may take a few minutes)
pip install -r requirements.txt
```

#### Step 4: Set Up Environment Variables

Replit provides a secure way to store sensitive information like API keys:

1. In the left sidebar, click on "Tools"
2. Select "Secrets"
3. Add the following secrets one by one:
   - Key: `SUPABASE_URL` | Value: Your Supabase URL (from Supabase dashboard)
   - Key: `SUPABASE_KEY` | Value: Your Supabase API key (from Supabase dashboard)
   - Key: `JWT_SECRET` | Value: A secure random string (you can generate one at [randomkeygen.com](https://randomkeygen.com/))
4. Click "Add new secret" after each entry

#### Step 5: Configure the Run Command

1. In the Replit file explorer, locate the `.replit` file at the root of the project
2. Click on it to open it in the editor
3. Replace its contents with the following:

```
entrypoint = "backend/main.py"
modules = ["python-3.10:v18-20230807-322e88b"]

run = "cd backend && uvicorn main:app --host 0.0.0.0 --port 8080"

[nix]
channel = "stable-23_05"

[env]
PYTHONPATH = "${REPL_HOME}/backend"

[deployment]
run = ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port 8080"]
```

4. Save the file

#### Step 6: Run the Backend

1. Click the "Run" button at the top of the Replit interface
2. Wait for the backend to start (you should see output indicating the server is running)
3. Replit will display a URL where your backend is accessible (typically something like `https://control-agency-backend.yourusername.repl.co`)

#### Step 7: Enable Always On (Optional)

By default, Replit will shut down your backend after a period of inactivity. To keep it running continuously:

1. Click on the project name in the top-left corner
2. Select "Deploy" from the dropdown menu
3. Toggle "Always On" to enable it (note: this requires a Replit Hacker Plan)

#### Step 8: Verify Deployment

Test your backend API by visiting the health endpoint in your browser:
```
https://your-repl-url/health
```

You should see a JSON response indicating the backend is healthy.

### Option 2: Local Deployment

Deploying the backend locally gives you more control and is ideal for development or self-hosting. Follow these steps carefully:

#### Step 1: Clone the Repository

```bash
# Open your terminal and navigate to your preferred directory
cd ~/projects  # or any directory of your choice

# Clone the repository
git clone https://github.com/Parokor/control-agency.git

# Navigate to the project directory
cd control-agency
```

#### Step 2: Set Up Python Virtual Environment

A virtual environment keeps your dependencies isolated from other Python projects:

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# You should see (venv) at the beginning of your command prompt
# indicating the virtual environment is active
```

#### Step 3: Install Dependencies

```bash
# Make sure you're in the backend directory with the virtual environment activated

# Install all required packages
pip install -r requirements.txt

# If you encounter any errors, try updating pip first
# pip install --upgrade pip
```

#### Step 4: Configure Environment Variables

Create a `.env` file to store your configuration securely:

```bash
# Create and open .env file
# On Windows:
echo SUPABASE_URL=your_supabase_url > .env
echo SUPABASE_KEY=your_supabase_key >> .env
echo JWT_SECRET=your_secure_random_string >> .env

# On macOS/Linux:
cat > .env << EOL
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
JWT_SECRET=your_secure_random_string
EOL
```

Replace the placeholder values with your actual credentials:
- `your_supabase_url`: URL from your Supabase project dashboard
- `your_supabase_key`: API key from your Supabase project dashboard
- `your_secure_random_string`: A secure random string for JWT token signing (generate at [randomkeygen.com](https://randomkeygen.com/))

#### Step 5: Start the Backend Server

```bash
# Make sure you're in the backend directory with the virtual environment activated

# Start the server in development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# For production, remove the --reload flag
# uvicorn main:app --host 0.0.0.0 --port 8080
```

You should see output indicating the server is running, typically something like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

#### Step 6: Verify Deployment

Test your backend API by visiting the health endpoint in your browser:
```
http://localhost:8080/health
```

You should see a JSON response indicating the backend is healthy.

#### Step 7: Running as a Service (Optional)

For production deployments, you may want to run the backend as a service that starts automatically:

**On Linux (using systemd):**

1. Create a service file:
```bash
sudo nano /etc/systemd/system/control-agency.service
```

2. Add the following content (adjust paths as needed):
```
[Unit]
Description=Control Agency Backend
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/control-agency/backend
EnvironmentFile=/path/to/control-agency/backend/.env
ExecStart=/path/to/control-agency/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl enable control-agency
sudo systemctl start control-agency
```

4. Check status:
```bash
sudo systemctl status control-agency
```

## Database Setup

The Federated AI System uses Supabase as its database provider, which offers a generous free tier. Follow these steps to set up your database:

#### Step 1: Create a Supabase Account

1. Go to [Supabase](https://supabase.com) and click "Start your project"
2. Sign up using your GitHub account or email
3. Verify your email address if required

#### Step 2: Create a New Project

1. From the Supabase dashboard, click "New Project"
2. Select an organization (create one if needed)
3. Enter a name for your project (e.g., "control-agency")
4. Set a secure database password (save this somewhere safe)
5. Choose a region closest to your users
6. Click "Create new project"

#### Step 3: Wait for Project Setup

Supabase will now create your project. This typically takes 1-2 minutes. You'll see a progress indicator while it sets up your database.

#### Step 4: Get API Credentials

Once your project is created:

1. Go to the project dashboard
2. In the left sidebar, click on "Project Settings"
3. Click on "API" in the submenu
4. You'll find your API URL and anon key (public API key) here
5. Copy these values as you'll need them for your backend configuration

#### Step 5: Initialize the Database Schema

Now you need to set up the database schema using the provided initialization script:

```bash
# Make sure you've cloned the repository
git clone https://github.com/Parokor/control-agency.git
cd control-agency

# Navigate to the scripts directory
cd scripts

# Create and activate a virtual environment (if not already done)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the database initialization script
python init_database.py --url "YOUR_SUPABASE_URL" --key "YOUR_SUPABASE_KEY"
```

Replace `YOUR_SUPABASE_URL` and `YOUR_SUPABASE_KEY` with the values you copied in Step 4.

#### Step 6: Verify Database Setup

1. Go back to your Supabase project dashboard
2. In the left sidebar, click on "Table Editor"
3. You should see several tables created by the initialization script
4. Click on a few tables to verify they have the correct structure

#### Step 7: Database Backup (Optional but Recommended)

It's good practice to set up regular backups of your database:

1. In the Supabase dashboard, go to "Project Settings"
2. Click on "Database"
3. Scroll down to the "Backups" section
4. Enable "Pitr" (Point-in-Time Recovery) if available on your plan
5. Alternatively, you can set up a scheduled SQL dump using the Supabase API

## Specialized Containers Deployment

The Federated AI System uses three specialized containers for different functionalities. Each container can be deployed independently based on your needs.

### Prerequisites for Container Deployment

Before deploying any containers, ensure you have Docker installed and configured:

```bash
# Install Docker (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Log out and log back in for the group changes to take effect

# Verify Docker installation
docker --version
docker-compose --version
```

For Windows and macOS, download and install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).

### Chat Container

The Chat Container provides the chat interface using Dolphin 3.0 R1 with Groq Cloud LPU.

#### Step 1: Set Up Environment Variables

Create a `.env` file in the project root directory with your API keys:

```bash
# Create .env file in the project root
cat > .env << EOL
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
EOL
```

You can obtain API keys from:
- [Groq Cloud](https://console.groq.com/) - Sign up and get your API key
- [OpenRouter](https://openrouter.ai/) - Sign up and get your API key (used as fallback)

#### Step 2: Build and Deploy the Chat Container

```bash
# Make sure you've cloned the repository
git clone https://github.com/Parokor/control-agency.git
cd control-agency

# Navigate to the chat container directory
cd containers/chat_container

# Build the Docker image
docker build -t control-agency/chat-container .

# Run the container
docker run -d \
  --name chat-container \
  -p 8000:8000 \
  --env-file ../../.env \
  --restart unless-stopped \
  control-agency/chat-container
```

#### Step 3: Verify Chat Container Deployment

1. Check if the container is running:
```bash
docker ps | grep chat-container
```

2. View container logs:
```bash
docker logs chat-container
```

3. Access the chat interface in your browser:
```
http://localhost:8000
```

### Development Container

The Development Container provides GitHub integration and code development capabilities.

#### Step 1: Build and Deploy the Development Container

```bash
# Navigate to the development container directory
cd control-agency/containers/dev_container

# Build the Docker image
docker build -t control-agency/dev-container .

# Run the container
docker run -d \
  --name dev-container \
  -p 8001:8000 \
  --env-file ../../.env \
  -v dev-container-data:/app/repositories \
  --restart unless-stopped \
  control-agency/dev-container
```

The `-v dev-container-data:/app/repositories` flag creates a persistent volume for repositories, ensuring your code is preserved even if the container is restarted.

#### Step 2: Verify Development Container Deployment

1. Check if the container is running:
```bash
docker ps | grep dev-container
```

2. View container logs:
```bash
docker logs dev-container
```

3. Access the development interface in your browser:
```
http://localhost:8001
```

### Media Container

The Media Container provides ComfyUI for media generation. This container requires GPU support for optimal performance.

#### Step 1: Install NVIDIA Docker Support (for GPU containers)

```bash
# For Ubuntu/Debian
# Install NVIDIA drivers if not already installed
sudo apt-get install nvidia-driver-535  # Use the latest version available

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify NVIDIA Docker installation
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

For Windows and macOS, follow the [official NVIDIA Docker installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### Step 2: Build and Deploy the Media Container

```bash
# Navigate to the media container directory
cd control-agency/containers/media_container

# Build the Docker image
docker build -t control-agency/media-container .

# Run the container with GPU support
docker run -d \
  --name media-container \
  -p 8188:8188 \
  --gpus all \
  -v media-container-data:/app/output \
  --restart unless-stopped \
  control-agency/media-container
```

The `-v media-container-data:/app/output` flag creates a persistent volume for generated media, ensuring your outputs are preserved even if the container is restarted.

#### Step 3: Verify Media Container Deployment

1. Check if the container is running:
```bash
docker ps | grep media-container
```

2. View container logs:
```bash
docker logs media-container
```

3. Access the ComfyUI interface in your browser:
```
http://localhost:8188
```

### Using Docker Compose (Alternative Method)

For easier management of all containers, you can use Docker Compose:

1. Create a `docker-compose.yml` file in the project root:

```bash
cat > docker-compose.yml << EOL
version: '3'

services:
  chat-container:
    build: ./containers/chat_container
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped

  dev-container:
    build: ./containers/dev_container
    ports:
      - "8001:8000"
    env_file:
      - .env
    volumes:
      - dev-container-data:/app/repositories
    restart: unless-stopped

  media-container:
    build: ./containers/media_container
    ports:
      - "8188:8188"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - media-container-data:/app/output
    restart: unless-stopped

volumes:
  dev-container-data:
  media-container-data:
EOL
```

2. Start all containers:

```bash
docker-compose up -d
```

3. Stop all containers:

```bash
docker-compose down
```

## System Integration and Verification

After deploying all components, you need to integrate them and verify that the entire system is working correctly.

### Step 1: System Integration

1. Update the frontend configuration to point to your backend:
   - Navigate to the frontend directory
   - Open `src/config.js` (or similar configuration file)
   - Update the API endpoint to point to your deployed backend URL
   - Rebuild and redeploy the frontend if necessary

2. Configure cross-origin resource sharing (CORS) on the backend:
   - For Replit deployment, add your frontend domain to the allowed origins
   - For local deployment, update the CORS settings in the backend code

3. Connect specialized containers to the backend:
   - Update the backend configuration to point to the container URLs
   - Ensure all containers are accessible from the backend

### Step 2: Comprehensive Verification

Verify that all components are working correctly and communicating with each other:

#### Frontend Verification

1. Visit your deployed frontend URL (GitHub Pages or Cloudflare Pages)
2. Check that the UI loads correctly without console errors
3. Verify that all UI elements are displayed properly
4. Test navigation between different sections

#### Backend Verification

1. Test the health endpoint:
   ```bash
   curl https://your-backend-url/health
   ```
   You should receive a JSON response with status "healthy"

2. Test authentication (if applicable):
   ```bash
   curl -X POST https://your-backend-url/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"test","password":"test"}'
   ```

3. Test API endpoints:
   ```bash
   # Get a list of resources (replace with actual endpoint)
   curl https://your-backend-url/api/resources \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

#### Container Verification

1. Check that all containers are running:
   ```bash
   docker ps
   ```
   You should see all three containers (chat, development, media) in the list

2. Verify each container individually:
   - Chat Container: Visit http://localhost:8000
   - Development Container: Visit http://localhost:8001
   - Media Container: Visit http://localhost:8188

3. Test container functionality:
   - Chat Container: Send a test message and verify response
   - Development Container: Create a test repository
   - Media Container: Generate a test image

#### Database Verification

1. Check database connection from the backend:
   ```bash
   curl https://your-backend-url/api/database/status
   ```

2. Verify data persistence:
   - Create a test record through the API
   - Verify it appears in the Supabase dashboard

### Step 3: End-to-End Testing

Perform end-to-end testing to ensure all components work together:

1. Create a new user account through the frontend
2. Log in with the new account
3. Create a new project or resource
4. Verify the resource is stored in the database
5. Test integration with each specialized container

## Troubleshooting Guide

If you encounter issues during deployment or verification, use this troubleshooting guide to resolve them.

### Frontend Issues

#### Build Failures

**Symptoms**: npm build fails with errors

**Solutions**:
1. Check Node.js version (v16+ recommended):
   ```bash
   node --version
   ```
   If outdated, install the latest LTS version from [nodejs.org](https://nodejs.org/)

2. Clear npm cache and reinstall dependencies:
   ```bash
   npm cache clean --force
   rm -rf node_modules
   npm install
   ```

3. Check for syntax errors in your code:
   ```bash
   npm run lint
   ```

#### Deployment Issues

**Symptoms**: Frontend deploys but shows blank page or errors

**Solutions**:
1. Check browser console for errors (F12 in most browsers)

2. Verify the build output:
   ```bash
   ls -la dist
   ```
   Ensure index.html and JS/CSS files are present

3. For GitHub Pages, check that the repository settings are correct:
   - Go to repository Settings > Pages
   - Ensure source is set to "gh-pages" branch

4. For Cloudflare Pages, check build logs in the Cloudflare dashboard

### Backend Issues

#### Deployment Failures

**Symptoms**: Backend fails to start or crashes

**Solutions**:
1. Check Python version (3.8+ recommended):
   ```bash
   python --version
   ```

2. Verify all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Check for syntax errors:
   ```bash
   python -m py_compile main.py
   ```

4. Verify environment variables:
   ```bash
   # For local deployment
   cat .env

   # For Replit
   # Check Secrets in the Replit dashboard
   ```

#### API Errors

**Symptoms**: API returns errors or unexpected responses

**Solutions**:
1. Check backend logs:
   ```bash
   # For local deployment
   tail -f backend.log

   # For Replit
   # Check the Console tab in Replit
   ```

2. Verify database connection:
   ```bash
   # Test database connection manually
   python -c "import os, supabase; print(supabase.create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY']).table('your_table').select('*').execute())"
   ```

3. Check API endpoint implementation for logical errors

### Database Issues

**Symptoms**: Database queries fail or return unexpected results

**Solutions**:
1. Verify Supabase credentials:
   - Check that URL and API key are correct
   - Ensure the API key has the necessary permissions

2. Check database schema:
   - Verify tables exist with correct structure
   - Check for missing columns or constraints

3. Run database initialization script again:
   ```bash
   python init_database.py --url "YOUR_SUPABASE_URL" --key "YOUR_SUPABASE_KEY"
   ```

### Container Issues

#### Docker Problems

**Symptoms**: Containers fail to build or start

**Solutions**:
1. Verify Docker is installed and running:
   ```bash
   docker --version
   systemctl status docker
   ```

2. Check Docker logs:
   ```bash
   docker logs [container_name]
   ```

3. Rebuild the container with verbose output:
   ```bash
   docker build -t control-agency/[container-name] . --progress=plain
   ```

#### GPU Container Issues

**Symptoms**: Media container starts but GPU is not detected

**Solutions**:
1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Docker installation:
   ```bash
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. Verify Docker has GPU access:
   ```bash
   docker info | grep -i runtime
   ```
   Should show nvidia as an available runtime

### Networking Issues

**Symptoms**: Components cannot communicate with each other

**Solutions**:
1. Check firewall settings:
   ```bash
   sudo ufw status
   ```
   Ensure ports 8000, 8001, and 8188 are open

2. Verify container network:
   ```bash
   docker network ls
   docker network inspect bridge
   ```

3. Test connectivity between components:
   ```bash
   curl -v http://localhost:8000/health
   curl -v http://localhost:8001/health
   curl -v http://localhost:8188/health
   ```

### Common Error Messages and Solutions

#### "Module not found" errors

**Solution**: Install the missing module:
```bash
pip install [module_name]
```

#### "Permission denied" errors

**Solution**: Check file permissions and ownership:
```bash
chmod +x [script_name]
# or
sudo chown -R $USER:$USER [directory]
```

#### "Address already in use" errors

**Solution**: Find and stop the process using the port:
```bash
sudo lsof -i :[port_number]
sudo kill [process_id]
```

#### "Connection refused" errors

**Solution**: Verify the service is running and accessible:
```bash
telnet localhost [port_number]
# or
nc -zv localhost [port_number]
```

## Getting Help

If you continue to experience issues after trying the troubleshooting steps above:

1. Check the [GitHub Issues](https://github.com/Parokor/control-agency/issues) for similar problems and solutions

2. Create a new issue with detailed information:
   - Exact error messages
   - Steps to reproduce
   - Environment details (OS, Docker version, etc.)
   - Logs from the component having issues

3. Join the community discussion on [GitHub Discussions](https://github.com/Parokor/control-agency/discussions) for additional support

## Conclusion

Congratulations! You have successfully deployed the Federated AI System. Here's a summary of what you've accomplished:

1. **Frontend Deployment**: Set up and deployed the user interface using either GitHub Pages or Cloudflare Pages
2. **Backend Deployment**: Deployed the backend API using either Replit or local deployment
3. **Database Setup**: Configured Supabase as the database provider and initialized the schema
4. **Specialized Containers**: Deployed three containers for chat, development, and media generation
5. **System Integration**: Connected all components to work together as a unified system
6. **Verification**: Tested all components to ensure they're working correctly

### Next Steps

Now that your system is deployed, consider the following next steps:

1. **User Management**: Set up user accounts and access controls
2. **Monitoring**: Implement monitoring and alerting for system health
3. **Backup Strategy**: Establish regular backups of your database and configurations
4. **Performance Optimization**: Fine-tune the system for better performance
5. **Feature Expansion**: Explore adding new features or integrating with additional services

### Maintenance Best Practices

1. **Regular Updates**: Keep all components updated with security patches
2. **Periodic Testing**: Regularly test all functionality to catch issues early
3. **Resource Monitoring**: Keep an eye on resource usage, especially for free tier services
4. **Documentation**: Keep your deployment documentation updated as you make changes

Thank you for using the Federated AI System! If you have any questions or need further assistance, please refer to the documentation or reach out to the community.
