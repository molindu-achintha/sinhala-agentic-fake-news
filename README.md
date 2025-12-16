# Sinhala Agentic Fake News Detection

This project implements an Agentic AI system for detecting fake news in Sinhala. It uses a modular agent architecture covering Claim Extraction, Language Processing, Retrieval Augmented Generation (RAG), Reasoning, and Verdict generation.

## Project Structure
- `backend/`: FastAPI backend with agent logic.
- `data/`: Datasets and preprocessing scripts.
- `training/`: Model training and evaluation scripts.
- `infra/`: deployment configurations.
- `tests/`: Project tests.

## Setup & Running

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- **HuggingFace API Key** (Required for Embeddings and Classification)

### Environment Setup
1. Create a `.env` file in the root directory (see `.env.example`).
2. Add your key: `HF_API_KEY=hf_...`

### Running with Docker
```bash
cd backend
docker-compose up --build
```
The API will be available at `http://localhost:8080/v1/predict`.

### Local Development
1. **Install Dependencies**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2. **Preprocessing**:
    ```bash
    # Prepare data
    python data/preprocessing/preprocess.py
    # Build embeddings
    python data/preprocessing/build_embeddings.py
    ```
3. **Train Classifier**:
    ```bash
    python training/train_classifier.py --data data/dataset/processed.jsonl
    ```
4. **Run API**:
    ```bash
    cd backend/app
    uvicorn main:app --reload --port 8080
    ```

### Testing
```bash
pytest
```
