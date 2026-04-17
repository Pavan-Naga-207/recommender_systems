# Variant C — ItemKNN (BM25)

**Team 11, CMPE 256 — Member: Sarat**

## Section A

**ItemKNN** is a collaborative filtering method that recommends items by looking at what similar items were liked by the same users. For each item, the model finds a neighborhood of “neighbor” items (often using cosine similarity on user interaction vectors, sometimes after weighting like BM25 or TF–IDF), then scores a candidate item for a user by aggregating those neighbors weighted by the user’s past interactions. It is simple, interpretable, and often strong on implicit-feedback data because it directly exploits item–item co-occurrence patterns without learning dense user embeddings.

## Approach

This variant uses **item-based nearest neighbors**:

1. Build a user-item interaction matrix from train split only.
2. Convert it to item-user and apply **BM25 weighting**.
3. Compute cosine neighbors for each item (best `K=320`).
4. Score candidate items by summing neighbor similarities over a user's seen
   items.

The model is: **ItemKNN (BM25-weighted cosine)**.

## How this fits the shared framework

- Consumes the pipeline's `train.csv`, `val.csv`, `test.csv` unchanged.
- Uses the same evaluation: **Precision@10, Recall@10, NDCG@10**.
- Uses the same masking as `phase1/run_baselines.py` for test: items seen in
  **train** are removed from candidates.

## Files

- `train_itemknn.py` — ItemKNN training + evaluation script
- `results_itemknn_selected.json` — selected ItemKNN run
- `requirements.txt` — dependencies

## Running

```bash
pip install -r requirements.txt

python train_itemknn.py \
  --data-dir path/to/shared_pipeline_outputs/ \
  --output results_itemknn.json
```

## Selected run

- Model: **ItemKNN (BM25)**
- Neighbors: **320**
- Test metrics:
  - Precision@10: **0.042994**
  - Recall@10: **0.063505**
  - NDCG@10: **0.065180**
