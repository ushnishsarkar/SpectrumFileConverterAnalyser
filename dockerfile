# Dockerfile
# Multi-stage Docker build for Spectrum Converter Pro

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # For GUI applications (if needed)
    libgtk-3-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    # For plotting
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
RUN chown app:app /app

# Copy application code
COPY --chown=app:app . .

# Install application
RUN pip install -e .

# Switch to non-root user
USER app

# Set up data directory
RUN mkdir -p /app/data /app/output

# Environment variables
ENV PYTHONPATH=/app
ENV DISPLAY=:0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from spectrum_converter.main import main; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "spectrum_converter.main"]

# For CLI usage:
# docker run -it --rm -v $(pwd)/data:/app/data spectrum-converter-pro spectrum-cli --help

# docker-compose.yml
version: '3.8'

services:
  spectrum-converter:
    build: .
    image: spectrum-converter-pro:latest
    container_name: spectrum-converter
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - DISPLAY=${DISPLAY:-:0}
    networks:
      - spectrum-net
    restart: unless-stopped
    # For GUI applications, you might need:
    # volumes:
    #   - /tmp/.X11-unix:/tmp/.X11-unix:rw
    # environment:
    #   - DISPLAY=${DISPLAY}
    # network_mode: host

  # Development service with live reload
  spectrum-converter-dev:
    build:
      context: .
      target: builder
    image: spectrum-converter-pro:dev
    container_name: spectrum-converter-dev
    volumes:
      - .:/app
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
      - DISPLAY=${DISPLAY:-:0}
    working_dir: /app
    command: python -m spectrum_converter.main
    networks:
      - spectrum-net
    profiles:
      - dev

networks:
  spectrum-net:
    driver: bridge

# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
      fail-fast: false

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .

    - name: Lint with flake8
      run: |
        flake8 src/spectrum_converter tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/spectrum_converter tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Type check with mypy
      run: |
        mypy src/spectrum_converter

    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=spectrum_converter --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety

    - name: Security check with bandit
      run: |
        bandit -r src/spectrum_converter -f json -o bandit-report.json
      continue-on-error: true

    - name: Safety check
      run: |
        safety check --json --output safety-report.json
      continue-on-error: true

    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: |
        python -m build

    - name: Check package
      run: |
        twine check dist/*

    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist-packages
        path: dist/

  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Docker Hub
      if: github.ref == 'refs/heads/main'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: spectrumanalysis/spectrum-converter-pro
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.ref == 'refs/heads/main' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    uses: ./.github/workflows/ci.yml

  release:
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: |
        python -m build

    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/*

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: |
          spectrumanalysis/spectrum-converter-pro:${{ github.ref_name }}
          spectrumanalysis/spectrum-converter-pro:latest

# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
        pip install -e .

    - name: Build documentation
      run: |
        cd docs
        make html

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html

# .dockerignore
# Git
.git
.gitignore

# Documentation
docs/
*.md

# Tests
tests/
.pytest_cache/
htmlcov/
.coverage

# Development
.vscode/
.idea/
*.swp
*.swo

# Virtual environments
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# Application specific
spectrum_converter_config.json
spectrum_file_history.json
*.spe
*.cnf
*.z1d
temp_*

# OS
.DS_Store
Thumbs.db

# scripts/build.sh
#!/bin/bash
# Build script for Spectrum Converter Pro

set -e

echo "Building Spectrum Converter Pro..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip build twine

# Build package
echo "Building package..."
python -m build

# Check package
echo "Checking package..."
twine check dist/*

# Build Docker image
echo "Building Docker image..."
docker build -t spectrum-converter-pro:latest .

echo "Build completed successfully!"
echo "Package files:"
ls -la dist/

# scripts/test.sh
#!/bin/bash
# Test script for Spectrum Converter Pro

set -e

echo "Running tests for Spectrum Converter Pro..."

# Install test dependencies
pip install -e ".[dev,plotting]"

# Run linting
echo "Running linting..."
flake8 src/spectrum_converter tests/
mypy src/spectrum_converter

# Run tests with coverage
echo "Running tests..."
pytest tests/ -v --cov=spectrum_converter --cov-report=html --cov-report=term

# Security checks
echo "Running security checks..."
bandit -r src/spectrum_converter
safety check

echo "All tests passed!"

# scripts/deploy.sh
#!/bin/bash
# Deployment script for Spectrum Converter Pro

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 2.0.0"
    exit 1
fi

VERSION=$1

echo "Deploying Spectrum Converter Pro v$VERSION..."

# Run tests first
./scripts/test.sh

# Build package
./scripts/build.sh

# Tag release
git tag -a "v$VERSION" -m "Release version $VERSION"
git push origin "v$VERSION"

# Build and push Docker image
docker tag spectrum-converter-pro:latest spectrum-converter-pro:$VERSION
docker tag spectrum-converter-pro:latest spectrumanalysis/spectrum-converter-pro:$VERSION
docker tag spectrum-converter-pro:latest spectrumanalysis/spectrum-converter-pro:latest

# Push to registry (if credentials are set)
if [ ! -z "$DOCKER_USERNAME" ]; then
    echo "Pushing Docker images..."
    docker push spectrumanalysis/spectrum-converter-pro:$VERSION
    docker push spectrumanalysis/spectrum-converter-pro:latest
fi

echo "Deployment completed!"
echo "Don't forget to:"
echo "1. Upload to PyPI: twine upload dist/*"
echo "2. Create GitHub release"
echo "3. Update documentation"
