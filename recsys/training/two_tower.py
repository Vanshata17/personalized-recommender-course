import tensorflow as tf
import tensorflow_recommenders as tfrs
from loguru import logger
from tensorflow.keras.layers import Normalization, StringLookup

from recsys.config import settings


class QueryTowerFactory:
    def __init__(self, dataset: "TwoTowerDataset") -> None:
        self._dataset = dataset

    def build(
        self, embed_dim: int = settings.TWO_TOWER_MODEL_EMBEDDING_SIZE
    ) -> "QueryTower":
        return QueryTower(
            user_ids=self._dataset.properties["user_ids"],
            emb_dim=embed_dim,
        )


class QueryTower(tf.keras.Model):
    def __init__(self, user_ids: list, emb_dim: int) -> None:
        super().__init__()

        self.user_embedding = tf.keras.Sequential(
            [
                StringLookup(vocabulary=user_ids, mask_token=None),
                tf.keras.layers.Embedding(
                    # Add an additional embedding to account for unknown tokens.
                    len(user_ids) + 1,
                    emb_dim,
                ),
            ]
        )

        self.normalized_age = Normalization(axis=None)

        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        concatenated_inputs = tf.concat(
            [
                self.user_embedding(inputs["customer_id"]),
                tf.reshape(self.normalized_age(inputs["age"]), (-1, 1)),
                tf.reshape(inputs["month_sin"], (-1, 1)),
                tf.reshape(inputs["month_cos"], (-1, 1)),
            ],
            axis=1,
        )

        outputs = self.fnn(concatenated_inputs)

        return outputs


class ItemTowerFactory:
    def __init__(self, dataset: "TwoTowerDataset") -> None:
        self._dataset = dataset

    def build(
        self, embed_dim: int = settings.TWO_TOWER_MODEL_EMBEDDING_SIZE
    ) -> "ItemTower":
        return ItemTower(
            item_ids=self._dataset.properties["item_ids"],
            garment_groups=self._dataset.properties["garment_groups"],
            index_groups=self._dataset.properties["index_groups"],
            emb_dim=embed_dim,
        )


class ItemTower(tf.keras.Model):
    def __init__(
        self,
        item_ids: list,
        garment_groups: list,
        index_groups: list,
        emb_dim: int,
    ):
        super().__init__()

        self.garment_groups = garment_groups
        self.index_groups = index_groups

        self.item_embedding = tf.keras.Sequential(
            [
                StringLookup(vocabulary=item_ids, mask_token=None),
                tf.keras.layers.Embedding(
                    # Add an additional embedding to account for unknown tokens.
                    len(item_ids) + 1,
                    emb_dim,
                ),
            ]
        )
        # Converts strings into integer indices (scikit-learn LabelEncoder analog)
        self.garment_group_tokenizer = StringLookup(
            vocabulary=garment_groups,
            mask_token=None,
        )
        self.index_group_tokenizer = StringLookup(
            vocabulary=index_groups,
            mask_token=None,
        )

        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        garment_group_embedding = tf.one_hot(
            self.garment_group_tokenizer(inputs["garment_group_name"]),
            len(self.garment_groups),
        )

        index_group_embedding = tf.one_hot(
            self.index_group_tokenizer(inputs["index_group_name"]),
            len(self.index_groups),
        )

        concatenated_inputs = tf.concat(
            [
                self.item_embedding(inputs["article_id"]),
                garment_group_embedding,
                index_group_embedding,
            ],
            axis=1,
        )

        outputs = self.fnn(concatenated_inputs)

        return outputs


class TwoTowerFactory:
    def __init__(self, dataset: "TwoTowerDataset") -> None:
        self._dataset = dataset

    def build(
        self,
        query_model: QueryTower,
        item_model: ItemTower,
        batch_size: int = settings.TWO_TOWER_MODEL_BATCH_SIZE,
    ) -> "TwoTowerModel":
        item_ds = self._dataset.get_items_subset()

        return TwoTowerModel(
            query_model,
            item_model,
            item_ds=item_ds,
            batch_size=batch_size,
        )


class TwoTowerModel(tf.keras.Model):
    def __init__(
        self,
        query_model: QueryTower,
        item_model: ItemTower,
        item_ds: tf.data.Dataset,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.query_model = query_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_ds.batch(batch_size).map(self.item_model)
            )
        )

    def train_step(self, batch) -> tf.Tensor:
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Loss computation.
            # The metrics returned by the training step are the 3 types of losses calculated earlier:
            # Retrieval loss: This measures how well the model matches user embeddings to the correct item embeddings.
            # Regularization loss: This prevents overfitting by adding penalties for model complexity, such as large weights. It encourages the model to generalize unseen data better.
            # Total loss: The sum of the retrieval loss and regularization loss. It represents the overall objective the model is optimizing during training.
            user_embeddings = self.query_model(batch)
            item_embeddings = self.item_model(batch)
            loss = self.task(
                user_embeddings,
                item_embeddings,
                compute_metrics=False,
            )

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {
            "loss": loss,
            "regularization_loss": regularization_loss,
            "total_loss": total_loss,
        }

        return metrics

    def test_step(self, batch) -> tf.Tensor:
        # Loss computation.
        user_embeddings = self.query_model(batch)
        item_embeddings = self.item_model(batch)

        loss = self.task(
            user_embeddings,
            item_embeddings,
            compute_metrics=False,
        )

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

# The TwoTowerDataset class is responsible for preparing 
# the data for training the two-tower model.
class TwoTowerDataset:
    def __init__(self, feature_view, batch_size: int) -> None:
        self._feature_view = feature_view
        self._batch_size = batch_size
        self._properties: dict | None
    # Defines properties for query and candidate features, 
    # ensuring that they are only accessed after the dataset 
    # has been initialized with train/validation splits.
    @property
    def query_features(self) -> list[str]:
        return ["customer_id", "age", "month_sin", "month_cos"]

    @property
    def candidate_features(self) -> list[str]:
        return [
            "article_id",
            "garment_group_name",
            "index_group_name",
        ]
    #What it does:
    #Returns stored properties
    #Asserts that get_train_val_split() was called first
    #Prevents errors if accessed before initialization
    @property
    def properties(self) -> dict:
        assert self._properties is not None, "Call get_train_val_split() first."

        return self._properties

    def get_items_subset(self):
        # Select columns: ["article_id", "garment_group_name", "index_group_name"]
        # Shape: (rows, 3)
        item_df = self.properties["train_df"][self.candidate_features]
        # Remove duplicate articles
        # Keep only first occurrence of each article_id
        # Shape reduces from 200K to 100K rows
        item_df.drop_duplicates(subset="article_id", inplace=True)
        # convert to tf dataset with dict of column names to values
        #Gets unique items for pre-computing embeddings
        # Used in TwoTowerModel for retrieval task
        item_ds = self.df_to_ds(item_df)

        return item_ds
    # Retrieves train and validation splits from feature view and converts them to tf datasets
    # Caches datasets for performance and shuffles training data for better generalization
    # Stores properties like user/item ids and group names for model building and training
    # Ensures that properties are only accessed after initialization to prevent errors
    # Used in TwoTowerTrainer to get data for training and validation
    def get_train_val_split(self):
        logger.info("Retrieving and creating train, val test split...")

        train_df, val_df, test_df, _, _, _ = (
            self._feature_view.train_validation_test_split(
                validation_size=settings.TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE,
                test_size=settings.TWO_TOWER_DATASET_TEST_SPLIT_SIZE,
                description="Retrieval dataset splits",
            )
        )

        train_ds = (
            self.df_to_ds(train_df)
            .batch(self._batch_size)
            .cache()
            .shuffle(self._batch_size * 10)
        )
        val_ds = self.df_to_ds(val_df).batch(self._batch_size).cache()

        self._properties = {
            "train_df": train_df,
            "val_df": val_df,
            "query_df": train_df[self.query_features],
            "item_df": train_df[self.candidate_features],
            "user_ids": train_df["customer_id"].unique().tolist(),
            "item_ids": train_df["article_id"].unique().tolist(),
            "garment_groups": train_df["garment_group_name"].unique().tolist(),
            "index_groups": train_df["index_group_name"].unique().tolist(),
        }

        return train_ds, val_ds
    
    def df_to_ds(self, df):
        return tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})

# The TwoTowerTrainer class is responsible for training the two-tower model. 
# It initializes the query tower with the training data, 
# compiles the model with an AdamW optimizer, and runs the 
# training loop for a specified number of epochs. 
# The _initialize_query_model method ensures that the age 
# normalization layer in the query tower is properly adapted 
# to the training data before training begins.
class TwoTowerTrainer:
    def __init__(self, dataset: TwoTowerDataset, model: TwoTowerModel) -> None:
        self._dataset = dataset
        self._model = model

    def train(self, train_ds, val_ds):
        self._initialize_query_model(train_ds)

        # Define an optimizer using AdamW with a learning rate of 0.01
        optimizer = tf.keras.optimizers.AdamW(
            weight_decay=settings.TWO_TOWER_WEIGHT_DECAY,
            learning_rate=settings.TWO_TOWER_LEARNING_RATE,
        )

        # Compile the model using the specified optimizer
        self._model.compile(optimizer=optimizer)

        # Start training
        history = self._model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=settings.TWO_TOWER_NUM_EPOCHS,
        )

        return history

    def _initialize_query_model(self, train_ds):
        # Initialize age normalization layer.
        self._model.query_model.normalized_age.adapt(train_ds.map(lambda x: x["age"]))

        # Initialize model with inputs.
        # Get user features: customer_id, age, month_sin, month_cos
        query_df = self._dataset.properties["query_df"]
        # Convert to Dataset and batch with size 1
        query_ds = self._dataset.df_to_ds(query_df).batch(1)
        # Pass one batch through model
        # Initializes all weights and layers in the query 
        # tower, including the age normalization layer 
        # which needs to see data to compute mean and variance 
        # for normalization
        self._model.query_model(next(iter(query_ds)))
