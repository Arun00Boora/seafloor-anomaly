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




## Upload your own images

The **Analyze** page accepts your own side-scan sonar images (up to 10 per run, JPG or PNG, 8 MB each). Each image is cut into square tiles, every tile is scored, and the page shows an error heatmap, boxes around the most suspicious tiles, and a ranked list. For each of the top tiles you can mark **Debris**, **False positive** or **Unsure**. Flags are kept in your browser and can be downloaded as a CSV.

Uploads run on ONNX models through `onnxruntime`, so the server needs no PyTorch.

## Three detectors, compared

| Detector | Idea | Needs sonar training? |
|---|---|---|
| Brightness baseline | Score = brightest pixel in the tile | No |
| Convolutional autoencoder | Trained on normal tiles only; score = how badly it reconstructs a tile | Yes |
| PatchCore-style (pretrained ResNet18) | Compare each patch of a tile with a memory bank of 20,000 normal patches; score = distance of the most unusual patch | No (only a memory bank of normal tiles) |

An ensemble of the autoencoder and the baseline is also shown. **Uploads are ranked by the PatchCore score**, because it performed best in the evaluation below.

### Results on the 700 test tiles (65% debris)

| Tiles an analyst can skip while still finding... | Ensemble | Autoencoder | Baseline | PatchCore |
|---|---|---|---|---|
| 50% of debris | 66.1% | 66.1% | 64.4% | 67.1% |
| 80% of debris | 43.6% | 42.7% | 43.7% | **46.9%** |
| 90% of debris | 34.6% | 29.6% | 35.6% | **39.3%** |

At the top 30% of the ranking, precision is 95.7% for the autoencoder, 90.0% for the baseline and 99.0% for PatchCore.

Ties in a score are broken randomly and averaged over 200 orderings. The 71 tiles with the maximum brightness are all debris, so the baseline's strong result at the very top is real: very bright objects are found by a simple threshold.

### Results when debris is rare

A real survey is mostly empty seabed, so the test set was rebuilt with all 244 normal tiles and a random subset of debris tiles, repeated 300 times. Precision in the top 5% of the ranking (random review would score the debris rate):

| Debris rate | Ensemble | Autoencoder | Baseline | PatchCore |
|---|---|---|---|---|
| 10% | 64.4% | 67.5% | 47.4% | **88.0%** |
| 5% | 38.2% | 39.6% | 28.2% | **71.5%** |

Top 10% of the ranking: at 10% debris, PatchCore 76.9% against 62.1% for the autoencoder; at 5% debris, 45.9% against 34.2%. Deeper rows are limited by the small number of debris tiles available, so the top of the ranking is where the methods differ.

### Score cutoffs on the Analyze page

PatchCore scores are scaled 0 to 1 using the test set. A tile above **0.38** scores higher than 95% of normal test tiles ("worth a quick check"); above **0.50** it scores higher than 99% of normal tiles and matches the median debris tile ("worth a close look").

## Limitations of the upload feature and PatchCore results

- The test split is 65% debris, far more than a real survey, so compare methods to each other rather than reading raw percentages as real-world performance. The rare-debris section corrects for this.
- With 2% or 5% debris only 5 to 13 debris tiles remain, so results swing a lot (see the ranges on the Evaluation page).
- One dataset, with a small number of normal tiles. Other sonar types, resolutions or seabeds will behave differently, and the 64 x 64 resize loses detail.
- Normal and debris scores overlap (roughly 0.38 to 0.45 on the PatchCore scale), so some tiles cannot be decided from the score alone. That is why the tool ranks for a human reviewer instead of deciding automatically.
- Strong edges, such as the water-column stripe at the edge of some images, can raise the scores of normal tiles. One such background image scored 0.62 with PatchCore (autoencoder 0.41, brightness 0.99).
- The heatmap shows autoencoder reconstruction error, scaled per image.
- The free hosting sleeps when idle, so the first request can take up to a minute.

## Reproducing the PatchCore-style results