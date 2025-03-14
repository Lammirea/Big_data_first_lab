pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'derelia/lab_itmo_bigdata:latest'
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}", ".")
                }
            }
        }

        stage('Run Tests') {
            steps {
                sh 'docker-compose run --rm app'
            }
        }

        stage('Publish Coverage Report') {
            steps {
                cobertura(
                    coberturaReportFile: '**/coverage.xml'
                )
            }
        }

        stage('Push Docker Image') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_IMAGE}").push()
                    }
                }
            }
        }
    }
}