pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                    python3 --version
                    python3 -m venv .venv
                    .venv/bin/python -m pip install --upgrade pip
                    .venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Prepare Dataset') {
            steps {
                sh '''
                    .venv/bin/python download_data.py
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    .venv/bin/python app/train.py
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    .venv/bin/python -m pytest -v
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t aquaguard:latest .
                '''
            }
        }
         
        stage('Deploy AquaGuard') {
            steps {
                sh '''
                    docker rm -f aquaguard-app || true

                    docker run -d \
                        --name aquaguard-app \
                        --network aqua_project_default \
                        --restart unless-stopped \
                        -p 5001:5000 \
                        aquaguard:latest
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                sh '''
                    for i in $(seq 1 15); do
                        if docker exec aquaguard-app python -c \
                            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)" \
                            2>/dev/null; then
                            echo "AquaGuard deployment is healthy!"
                            exit 0
                        fi
                        sleep 2
                    done

                    echo "Deployment health check failed!"
                    docker logs --tail 50 aquaguard-app
                    exit 1
                '''
            }
        }

    }
}