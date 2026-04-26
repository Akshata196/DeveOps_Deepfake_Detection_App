# Deepfake Detection App using DevOps Pipeline

## Project Overview
This project demonstrates a complete DevOps pipeline for deploying a deep learning-based deepfake detection system. A hybrid model using spatial and frequency features is integrated into a web application and deployed using Docker, Jenkins, and Kubernetes.

---

## Features
- Deepfake detection using hybrid CNN model
- Image upload and prediction via web interface
- Confidence score display
- Automated CI/CD pipeline using Jenkins
- Containerized deployment using Docker
- Kubernetes-based orchestration (Minikube)

---

## Tech Stack
- Python
- PyTorch
- FastAPI
- HTML, CSS, JavaScript
- Docker
- Jenkins
- Kubernetes (Minikube)
- GitHub

---

## Project Architecture
Developer → GitHub → Jenkins → Docker → Kubernetes → Web Application


---

## Model Description
The model is a hybrid architecture combining:

- ResNet18 → Spatial feature extraction  
- MobileNetV2 → Frequency feature extraction (FFT-based)  
- Feature fusion → Concatenation  
- Final classification → Fake / Real  

---

## Folder Structure
deepfake-app/
│
├──> app.py
├──> index.html
├── Dockerfile
├── hybrid_model.pth
├── deployment.yaml
├── service.yaml
└── README.md


---

## How to Run Locally

### 1. Clone Repository
git clone https://github.com/Akshata196/DeveOps_Deepfake_Detection_App.git

cd deepfake-app


---

### 2. Run using Docker

docker build -t deepfake-app .
docker run -d -p 8000:8000 deepfake-app


Open: http://localhost:8000


---

## CI/CD Pipeline (Jenkins)

Pipeline stages:
- Clone repository from GitHub
- Build Docker image
- Run container

---

## Kubernetes Deployment

### Start Minikube
minikube start


### Use Minikube Docker
eval $(minikube docker-env)


### Build Image
docker build -t deepfake-app .


### Deploy
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml


### Access App
minikube service deepfake-service


---

## API Endpoint

### POST /predict
- Input: Image file
- Output:
{
"prediction": "fake",
"confidence": 92.34
}


---

## Output
- Upload image
- Prediction displayed (Fake / Real)
- Confidence score shown

---

## Advantages
- Automated deployment using CI/CD
- Scalable architecture with Kubernetes
- Easy to deploy and maintain
- Real-world DevOps integration

---

## Future Scope
- Extend to video deepfake detection
- Deploy on cloud platforms (AWS/GCP)
- Add monitoring (Nagios/Prometheus)
- Improve UI/UX

---

## Author
Akshata Jadhav

---

## License
This project is for academic and educational purposes.
