# Energy Consumption Prediction System

## Introduction
This project implements a machine learning system for predicting energy consumption across different types of buildings (Residential, Commercial, and Industrial). The system uses a Flask-based web application containerized with Docker and deployed using Kubernetes, following modern DevOps practices with comprehensive CI/CD pipelines.

## Artifact Details

### Technology Stack
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: Flask
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Testing**: pytest

### Project Structure
```
├── ml_model/
│   ├── model.py           # ML model training
│   ├── preprocessing.py   # Data preprocessing
│   ├── flask_app.py       # Web application
│   ├── requirements.txt   # Dependencies
│   └── Dockerfile        # Container configuration
├── test/
│   ├── model_test.py     # Model tests
│   └── Unit_test.py      # Unit tests
├── k8s/
│   └── deployment.yml    # Kubernetes configuration
└── .github/workflows/    # CI/CD pipelines
```

## Branching Strategy
The project follows a feature-branch workflow with the following branches:
- `feature`: Development branch for new features
- `Dev`: Integration branch for testing
- `Prod`: Production deployment branch

### Branch Flow
1. Feature development in feature branches
2. Merge to Dev for integration testing
3. Pull requests to Prod for production deployment

## Testing Strategy

### Test Categories
1. **Unit Tests**
   - Data preprocessing validation
   - Model utility functions testing
   - String handling and data type conversions

2. **Model Tests**
   - Model prediction validation
   - Input data validation
   - Model loading/saving verification

3. **Integration Tests**
   - Flask API endpoint testing
   - Health check validation
   - End-to-end prediction flow

### Test Cases
```python
# Example test cases
def test_func_digit():
    # Validates numeric conversion
    assert con_str_num('9000') == 9000

def test_func_null():
    # Validates null handling
    assert con_str_num('') is pd.NA

def test_home():
    # Validates API endpoint
    response = app.test_client().get('/')
    assert response.status_code == 200
```

## Deployment Strategy

### Container Strategy
- Docker images tagged with:
  - Version number
  - Git commit hash
  - Latest tag
- Images pushed to Docker Hub

### Kubernetes Deployment
- 3 replicas for high availability
- LoadBalancer service type
- Health checks implemented
- Rolling updates for zero-downtime deployments

## Model Retraining Strategy

### Trigger Conditions
- Automated retraining on new data file updates (CSV changes)
- Manual trigger available through workflow_dispatch

### Retraining Process
1. Data validation
2. Model training
3. Performance evaluation
4. Model artifact generation
5. Test execution
6. Deployment if tests pass

## CI/CD Workflows

### 1. Build and Push Workflow (runner.yml)
- Triggers: Push to feature branch
- Steps:
  - Code checkout
  - Docker Hub authentication
  - Version tagging
  - Image building and pushing

### 2. Deployment Workflow (deploy.yml)
- Triggers: 
  - Pull requests to Prod
  - Completion of image build workflow
- Steps:
  - Environment setup
  - Dependency installation
  - Model training
  - Test execution
  - Kubernetes deployment
  - Rollout status verification

### 3. Retraining Workflow (retrain.yml)
- Triggers: CSV file changes
- Purpose: Automated model retraining

## Continuous Testing/Monitoring

### Testing Automation
- Pre-deployment testing
- Post-deployment health checks
- API endpoint monitoring

### Monitoring
- Kubernetes health probes
- Container status monitoring
- API endpoint availability checks

## References
1. Flask Documentation: https://flask.palletsprojects.com/
2. scikit-learn Documentation: https://scikit-learn.org/
3. Kubernetes Documentation: https://kubernetes.io/docs/
4. GitHub Actions Documentation: https://docs.github.com/en/actions
5. Docker Documentation: https://docs.docker.com/