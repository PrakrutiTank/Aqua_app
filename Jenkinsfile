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
    }
}