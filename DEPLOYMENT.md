# Deployment Guide

This guide explains how to deploy the Sinhala Fake News Detector.

## Architecture

```
GitHub Pages (Frontend)     ─────────────>     Render (Backend)
     │                                              │
     │                                              ├── FastAPI Server
     │                                              ├── Redis (Cache)
     │                                              └── PostgreSQL (Memory)
     │
     └── Static HTML/JS/CSS
```

## Frontend Deployment (GitHub Pages)

### Automatic Deployment
The frontend automatically deploys when you push to the `main` branch.
The GitHub Actions workflow `.github/workflows/deploy-frontend.yml` handles this.

### Manual Setup
1. Go to your GitHub repository
2. Click Settings > Pages
3. Source: Deploy from a branch
4. Branch: main / frontend folder
5. Save

Your frontend will be available at:
`https://YOUR_USERNAME.github.io/sinhala-agentic-fake-news/`

## Backend Deployment (Render)

### Option 1: One-Click Deploy
1. Go to https://render.com
2. Click New > Blueprint
3. Connect your GitHub repository
4. Select `render.yaml` as the blueprint
5. Click Deploy

### Option 2: Manual Setup
1. Go to https://render.com
2. Click New > Web Service
3. Connect your GitHub repository
4. Configure:
   - Name: sinhala-fake-news-api
   - Runtime: Python
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Environment Variables (Required)
Set these in Render dashboard:
```
OPENROUTER_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=news-store
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIMENSION=1024
```

### Optional Services
- Redis (for caching): Add Redis service in Render
- PostgreSQL (for memory): Add PostgreSQL database in Render

## Update Frontend API URL

After deploying backend to Render, update the API URL in `frontend/script.js`:

```javascript
const API_BASE = 'https://YOUR-APP-NAME.onrender.com';
```

## Verify Deployment

1. Check backend health:
   ```
   curl https://YOUR-APP-NAME.onrender.com/v1/health
   ```

2. Open frontend in browser:
   ```
   https://YOUR_USERNAME.github.io/sinhala-agentic-fake-news/
   ```

3. Enter a claim and verify it works

## Free Tier Limits

### Render Free Tier
- 750 hours/month
- Sleeps after 15 min inactivity
- First request may take 30+ seconds to wake

### GitHub Pages
- Free for public repositories
- 100GB bandwidth/month

## Troubleshooting

### Backend not responding
- Check Render logs for errors
- Verify environment variables are set
- Wait 30 seconds for cold start

### CORS errors
- Backend already allows all origins
- Check browser console for specific error

### API key errors
- Verify OPENROUTER_API_KEY is set
- Verify PINECONE_API_KEY is set
