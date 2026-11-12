# Seafloor Debris and Anomaly Discovery in Side-Scan Sonar

An unsupervised anomaly detector for side-scan sonar. A convolutional autoencoder trained only on normal seabed ranks every tile from most to least anomalous, so an analyst can review the likely debris first instead of the whole survey. A brightness-threshold baseline, an evaluation suite, and a web tool for reviewing and flagging tiles come with it.

**Live demo:** https://seafloor-anomaly.vercel.app
**API:** https://seafloor-anomaly.onrender.com (free tier: the first request after a quiet period can take up to a minute)

## What it does

1. **Tiles** each sonar image into 64x64 grayscale tiles.
2. **Scores** every tile three ways: an autoencoder, a brightness baseline, and an ensemble of the two.
3. **Ranks** tiles by score. The ranked list is what the analyst sees.
4. **Lets an analyst review** tiles in a web tool: step through them, see a reconstruction-error heatmap, and flag each tile as confirmed debris, false positive or unsure.
5. **Evaluates** the ranking against the baseline with precision, recall and "review effort saved".

## How the scoring works

- **Autoencoder:** a small convolutional autoencoder (4 conv layers down, 4 up, 16x compression at the bottleneck) trained on normal tiles only. Debris is unusual, so it reconstructs badly. The score is the mean of the top 10% per-pixel squared errors, which keeps a small object from being averaged away by clean background.
- **Brightness baseline (required):** the tile's maximum pixel intensity, a plain intensity-threshold approach.
- **Ensemble:** both scores min-max normalized and averaged.

## Results

Test split: 700 tiles, 456 containing debris. The autoencoder trains on 1,500 normal tiles.

**Precision when reviewing the top K% of tiles**

| Review depth | Ensemble | Autoencoder | Brightness baseline |
|---|---|---|---|
| Top 5% (35 tiles) | 100.0% | 100.0% | 100.0% |
| Top 10% (70 tiles) | 100.0% | 100.0% | 100.0% |
| Top 20% (140 tiles) | 99.3% | 99.3% | 93.6% |
| Top 30% (210 tiles) | 95.7% | 95.7% | 90.0% |

**Recall (share of all debris found) when reviewing the top K%**

| Review depth | Ensemble | Autoencoder | Brightness baseline |
|---|---|---|---|
| Top 20% | 30.5% | 30.5% | 28.7% |
| Top 30% | 44.1% | 44.1% | 41.4% |

**Review effort saved** (share of tiles an analyst can skip while still finding a target share of the debris)

| Target | Ensemble | Autoencoder | Brightness baseline |
|---|---|---|---|
| Find 50% of debris | 66.1% | 66.1% | 64.4% |
| Find 80% of debris | 43.6% | 42.7% | 43.7% |
| Find 90% of debris | 34.6% | 29.6% | 35.6% |

**What the results say:** at the top of the ranking, where analysts start, the autoencoder is clearly more precise than the baseline (99.3% vs 93.6% at 20%, 95.7% vs 90.0% at 30%). To find nearly all of the debris, the plain brightness threshold performs as well as the ensemble on this benchmark, and better than the autoencoder alone. The ensemble is a safe middle choice, but it does not clearly beat both methods everywhere.

## Limitations (please read)

- **The test set is about 65% debris.** A real survey has far less, so the raw percentages are benchmark numbers, not real-world performance. Reviewing tiles at random would already be 65% precise here. Compare the methods to each other.
- **Small dataset:** 1,500 normal training tiles, 700 test tiles. Differences of a few tiles between methods are within noise.
- **Dataset:** [rehan9599/drishti-sss](https://huggingface.co/datasets/rehan9599/drishti-sss). Crab pots, shipwrecks, ghost nets and mine-like cylinders count as anomalies. Pipelines count as normal.
- **Edge sensitivity:** the autoencoder reacts to sharp bright-to-dark transitions (for example the edges of the dark nadir stripe), so the heatmap can highlight an edge rather than an object.
- **Heatmaps** are scaled to each tile's own maximum, so read them for where the problem is and the score for how severe it is.
- **Demo hosting:** the free Render server sleeps when idle, and review flags are stored on its temporary disk, so they reset when it restarts. The deployed demo serves the 700 test tiles with pre-rendered heatmaps.

## Run it locally

You need Python 3.12+, Node 20+, and (optionally) Docker. The repository includes `deploy_data/` (the 700 test tiles, ranked index and heatmaps), so you can run the review tool without downloading the dataset.

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app
```
Runs at http://localhost:8000.

**Frontend**
```bash
cd frontend
npm install
npm run dev
```
Runs at http://localhost:3000.

**Backend with Docker**
```bash
docker build -t seafloor-backend .
docker run --rm -p 8000:8000 seafloor-backend
```

## Reproduce the ML pipeline

Needs PyTorch and Pillow (`pip install torch torchvision pillow`) and the drishti-sss dataset downloaded to `data/raw_drishti/`.

```bash
cd ml/data && python prepare_drishti.py          # 64x64 tiles, normal/anomalous split
cd .. && python train.py                          # train the autoencoder on normal tiles
python score_tiles.py --split test                # autoencoder scores
python score_tiles_baseline.py --split test       # brightness baseline scores
python evaluate.py                                # precision/recall at 5/10/20/30%
python build_ranked_index.py --split test         # ranked index used by the API
python build_deploy_data.py                       # package tiles, index and heatmaps for deployment
```

## Project structure

```
ml/         data preparation, model, training, scoring, evaluation
backend/    FastAPI API: ranked tiles, images, heatmaps, flags, evaluation
frontend/   Next.js review tool and evaluation page
deploy_data/  packaged test tiles, ranked index and pre-rendered heatmaps
Dockerfile  backend image
```

## Tech

PyTorch, FastAPI, SQLAlchemy and SQLite, Next.js, Tailwind CSS, Docker, Render (backend) and Vercel (frontend).


