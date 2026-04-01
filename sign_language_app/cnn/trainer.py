"""CNN-based training for ASL recognition using 1D convolutional architecture."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from sign_language_app.trainer import (
    TrainingConfig,
    _build_train_test_split,
    _infer_landmark_channels,
    _load_csv_dataset,
    _normalize_landmark_tensor,
)


@dataclass
class CNNConfig(TrainingConfig):
    """Configuration for CNN training."""
    pass


def train_cnn_model(
    config: TrainingConfig,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    split_strategy: str = "random",
) -> None:
    """
    Train a 1D CNN model on hand landmark data.
    
    Args:
        config: TrainingConfig with dataset_csv and model_output paths
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Adam optimizer learning rate
        split_strategy: "random" for stratified split or "group-similarity" for stricter leakage prevention
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for CNN training. Install tensorflow (or tensorflow-macos on Apple Silicon)."
        ) from exc

    print("[CNN Trainer] Loading dataset...")
    X, y_labels = _load_csv_dataset(config.dataset_csv)
    channels = _infer_landmark_channels(X.shape[1])
    X = X.reshape((-1, 21, channels)).astype(np.float32)
    X = np.array([_normalize_landmark_tensor(sample) for sample in X], dtype=np.float32)

    print(f"[CNN Trainer] Dataset shape: {X.shape}, Classes: {len(np.unique(y_labels))}")

    classes = np.array(sorted(np.unique(y_labels)))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in y_labels], dtype=np.int32)

    print(f"[CNN Trainer] Building train/test split with strategy: {split_strategy}")
    X_train, X_test, y_train, y_test = _build_train_test_split(
        X,
        y,
        split_strategy=split_strategy,
        test_size=0.2,
        random_state=42,
    )

    print(f"[CNN Trainer] Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Build the model
    print("[CNN Trainer] Building CNN architecture...")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(21, channels)),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(len(classes), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[CNN Trainer] Model summary:")
    model.summary()

    # Train with early stopping
    print(f"[CNN Trainer] Training for up to {epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("[CNN Trainer] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[CNN Trainer] CNN test accuracy: {test_acc:.4f}")

    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save model and wrapper
    output_dir = os.path.dirname(config.model_output) or "."
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(config.model_output))[0]
    keras_path = os.path.join(output_dir, f"{stem}.keras")
    model.save(keras_path)

    wrapper = {
        "model_type": "cnn1d",
        "model_path": keras_path,
        "classes": classes.tolist(),
        "input_shape": [21, channels],
        "dataset_csv": config.dataset_csv,
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "split_strategy": split_strategy,
        "training": {
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "validation_split": 0.1,
            "early_stopping_patience": 8,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        },
    }

    with open(config.model_output, "wb") as handle:
        pickle.dump(wrapper, handle)

    print(f"[CNN Trainer] Saved CNN model to {keras_path}")
    print(f"[CNN Trainer] Saved CNN wrapper to {config.model_output}")
