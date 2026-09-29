# Deploying DataSleuth

Backend (FastAPI) on **Render**, frontend (React) on **Vercel**. Deploy the backend first because the frontend needs its URL.

## 1. Backend on Render

1. Render dashboard > **New > Blueprint** > select this GitHub repo. Render reads `render.yaml`.
2. When asked, set the two secrets:
   - `GEMINI_API_KEY`: your Gemini key.
   - `DATASLEUTH_CORS_ORIGINS`: leave a placeholder for now, e.g. `["https://placeholder.vercel.app"]`.
3. Wait for the build. Open `https://<your-service>.onrender.com/healthz`. It should return `{"status":"ok"}`.

## 2. Frontend on Vercel

1. Vercel dashboard > **Add New > Project** > import the same repo.
2. Set **Root Directory** to `frontend`. Vercel detects Vite from `frontend/vercel.json`.
3. Add an environment variable: `VITE_API_URL` = `https://<your-service>.onrender.com` (no trailing slash).
4. Deploy. Note the URL Vercel gives you, e.g. `https://datasleuth.vercel.app`.

## 3. Connect them

Back on Render, edit `DATASLEUTH_CORS_ORIGINS` to your real Vercel URL, for example
`["https://datasleuth.vercel.app"]`, and save (the service redeploys). Without this step the browser blocks the requests.

## Notes for the free tiers

- Render's free web service sleeps after inactivity; the first request after a pause takes about a minute.
- Uploads and sessions live in `/tmp`, which is erased on restart. That is fine for a demo.
- The Docker sandbox is not available on Render; the API uses the subprocess sandbox there.
- Every visitor shares your Gemini quota. `DATASLEUTH_LLM_DAILY_LIMIT` caps how many calls per day can be spent (lower it if you share the URL widely).
- Keep `.env` out of git. Secrets belong only in the Render dashboard.
