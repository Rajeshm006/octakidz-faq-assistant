# OctaKidz FAQ Assistant

A production-oriented monorepo for a warm, parent-friendly OctaKidz FAQ chat experience. The Angular app sends questions to FastAPI; it never accesses the FAQ dataset, CrewAI, or an OpenAI API key. The Python backend validates the bundled 103-question JSON dataset at startup and is the only component allowed to run the workflow.

## Safety and grounding

- User-facing OctaKidz facts come only from `backend/app/data/faq_dataset.json`.
- FAQ retrieval is deterministic token overlap. There is no RAG, embeddings, vector database, web search, or external business knowledge source.
- The backend implements four strict Pydantic handoffs: classification, retrieval, drafting, and escalation/refusal.
- A deterministic finalizer enforces refusal, escalation, no-match, and lead-capture policy after the fourth handoff.
- `internal_note` is returned to the API caller for trusted operational use, but the Angular UI never renders it.
- Optional session memory/cache is in process only. It is cleared whenever a Render instance restarts.

## Repository layout

```text
octakidz-faq-assistant/
├── frontend/                 # Angular standalone app for Vercel
├── backend/                  # FastAPI app for Render
│   └── app/data/faq_dataset.json
├── .env.example
├── .gitignore
└── README.md
```

## Local development

### Backend

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
cp .env.example backend/.env
```

Edit `backend/.env` and set only your own secret locally:

```text
OPENAI_API_KEY=your_real_key
MODEL_NAME=gpt-5.6
ALLOWED_ORIGINS=http://localhost:4200
ENABLE_LIVE_CREWAI=false
```

Do not print, share, or commit this file. The default `ENABLE_LIVE_CREWAI=false` keeps API calls disabled while preserving the deterministic, JSON-grounded workflow. If you opt in to the CrewAI pass, it runs server-side only; the final Python policy remains authoritative.

Start the API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm start
```

Check the API without exposing a key:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is OctaKidz?","session_id":"local-demo"}'
```

Run tests:

```bash
pytest backend/tests
cd frontend && npm test
```

# Deployment guide: GitHub → Render + Vercel

## A. Run locally

1. Create and activate the Python virtual environment, install `backend/requirements.txt`, and create `backend/.env` using the commands above.
2. Confirm `backend/.env` contains `OPENAI_API_KEY=your_real_key`, `MODEL_NAME=gpt-5.6`, and `ALLOWED_ORIGINS=http://localhost:4200`.
3. Start FastAPI:

   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Install and start Angular:

   ```bash
   cd frontend
   npm install
   npm start
   ```

5. Open `http://localhost:4200`. Use the curl commands above to check `/health` and `POST /api/chat`.

## B. Upload the project to GitHub

1. Create an empty GitHub repository named `octakidz-faq-assistant`. Before pushing, run `git status` and confirm that `backend/.env` is absent.
2. From the project root, run:

   ```bash
   git init
   git add .gitignore README.md frontend backend
   git status
   git commit -m "Initial FAQ assistant application"
   git branch -M main
   git remote add origin https://github.com/your-github-username/octakidz-faq-assistant.git
   git push -u origin main
   ```

3. Never run `git add .env` and never commit an OpenAI API key. In GitHub, open the repository file list and use search to confirm no `.env` or secret value appears. If one is committed, revoke the key immediately and remove it from Git history before continuing.

## C. Deploy the FastAPI backend to Render

1. Sign in to Render, select **New +** → **Web Service**, and connect the `octakidz-faq-assistant` repository.
2. Select branch `main`, set **Root Directory** to `backend`, and select the Python runtime.
3. Set **Build Command** to:

   ```bash
   pip install -r requirements.txt
   ```

4. Set **Start Command** to:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

5. Add these Render environment variables in the dashboard, never in source code:

   ```text
   OPENAI_API_KEY=your_real_key
   MODEL_NAME=gpt-5.6
   ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
   ENABLE_LIVE_CREWAI=false
   ```

6. Deploy and copy the public service URL. Verify `https://your-render-service.onrender.com/health` in a browser.
7. On plans that sleep when idle, the first request after inactivity can take time to wake up.

## D. Connect Angular to Render

1. Update `frontend/src/environments/environment.production.ts` with the copied Render URL. Keep it as the origin only—do not append `/api`, because the service appends `/api/chat`.
2. Commit and push:

   ```bash
   git add frontend/src/environments/environment.production.ts
   git commit -m "Configure production API URL"
   git push
   ```

3. After Vercel gives you its URL, update Render's `ALLOWED_ORIGINS` value and manually redeploy Render if needed.

## E. Deploy the Angular frontend to Vercel

1. In Vercel, select **Add New…** → **Project** and import `octakidz-faq-assistant`.
2. Set **Root Directory** to `frontend`. Choose the Angular preset, or **Other** if it is not detected.
3. Use build command `npm run build`. Set output directory to `dist/octakidz-faq-assistant/browser` (the Angular application builder output).
4. Do **not** add `OPENAI_API_KEY`, any secret, or an API key to Vercel.
5. Deploy and copy the Vercel production URL. The `frontend/vercel.json` rewrite serves `index.html` on SPA refreshes instead of returning a 404.
