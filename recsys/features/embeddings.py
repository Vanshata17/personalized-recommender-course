import pandas as pd
import tensorflow as tf


def preprocess(train_df: pd.DataFrame, candidate_features: list) -> pd.DataFrame:
    # Select the candidate features from the training DataFrame
    item_df = train_df[candidate_features]

    # Drop duplicate rows based on the 'article_id' column to get unique candidate items
    item_df.drop_duplicates(subset="article_id", inplace=True)

    return item_df


def embed(df: pd.DataFrame, candidate_model) -> pd.DataFrame:
    ds = tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})
#     Result: Dataset that yields dictionaries
#   {
#     "article_id": "A001",
#     "garment_group_name": "Tops",
#     "index_group_name": "Menswear"
#   },
#   {
#     "article_id": "A002",
#     ...
#   },
#   ...
# ds.batch(2048)

# Groups 2048 items together
# 100K items / 2048 = ~49 batches

# Batch 1: 2048 items
# Batch 2: 2048 items
# ...
# Batch 49: 1,536 items (remaining)
# For each batch:
#   Input: 2048 items with features
#   ↓
#   ItemTower processes all 2048 items
#   ↓
#   Output: 
#     - 2048 article IDs
#     - 2048 embeddings (32-dim each)
# Input batch (2048 items):
#   ["A001", "A002", "A003", ...]
  
# ItemTower encodes each:
#   A001 → [0.13, 0.46, -0.32, ..., 0.22]
#   A002 → [0.12, 0.47, -0.31, ..., 0.21]
#   A003 → [0.50, -0.20, 0.10, ..., 0.05]
#   ...
  
# Output batch (2048 embeddings):
#   [[0.13, 0.46, ...],
#    [0.12, 0.47, ...],
#    [0.50, -0.20, ...],
#    ...]
    candidate_embeddings = ds.batch(2048).map(
        lambda x: (x["article_id"], candidate_model(x))
    )

    all_article_ids = tf.concat([batch[0] for batch in candidate_embeddings], axis=0)
    # Combines all 49 batches
    # Result: [A001, A002, ..., A100000] (100K items)
    all_embeddings = tf.concat([batch[1] for batch in candidate_embeddings], axis=0)

    all_article_ids = all_article_ids.numpy().astype(int).tolist()
    all_embeddings = all_embeddings.numpy().tolist()

    embeddings_df = pd.DataFrame(
        {
            "article_id": all_article_ids,
            "embeddings": all_embeddings,
        }
    )

    return embeddings_df
