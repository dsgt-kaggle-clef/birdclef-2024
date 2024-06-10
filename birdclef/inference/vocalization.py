from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH
from birdclef.inference.base import Inference


def compile_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    return converter.convert()


class GoogleVocalizationInference(Inference):
    """Class to perform inference on audio files using a Google Vocalization model."""

    def __init__(
        self,
        metadata_path: str,
        model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
        use_compiled: bool = False,
    ):
        self.model = hub.load(model_path)
        self.metadata = pd.read_csv(metadata_path)
        self.model_labels_df = pd.read_csv(
            hub.resolve(model_path) + "/assets/label.csv"
        )
        self.model_indices = self._get_index_subset(
            self.metadata,
            self.model_labels_df,
        )
        self.use_compiled = use_compiled
        if use_compiled:
            self.compiled_model = tf.lite.Interpreter(
                model_content=compile_tflite_model(self.model)
            )
            self.compiled_model.allocate_tensors()

    def _get_index_subset(
        self, metadata: pd.DataFrame, model_labels: pd.DataFrame
    ) -> list:
        """Get the subset of labels that are in the model.

        We use the species names from the metadata to get the subset of labels that are in the model.
        """
        index_to_label = sorted(metadata.primary_label.unique())
        model_labels = {v: k for k, v in enumerate(model_labels.ebird2021)}
        return sorted(
            [
                model_labels[label] if label in model_labels else -1
                for label in index_to_label
            ]
        )

    def load(self, path: str, window: int = 5 * 32_000) -> np.ndarray:
        """Load an audio file.

        :param path: The absolute path to the audio file.
        """
        audio, _ = torchaudio.load(path)
        audio = audio.squeeze()
        # right pad the audio so we can reshape into a rectangle
        n = audio.shape[0]
        if n % window != 0:
            audio = audio[: n - (n % window)]
        # reshape the audio into windowsize chunks
        audio = audio.reshape(-1, window)
        return audio

    def _infer_tflite(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform inference using a TFLite model.

        :param audio: The audio data to perform inference on.
        """
        input_details = self.compiled_model.get_input_details()
        output_details = self.compiled_model.get_output_details()

        embeddings = []
        logits = []
        for x in audio:
            self.compiled_model.set_tensor(input_details[0]["index"], x.reshape(1, -1))
            self.compiled_model.invoke()
            embeddings.append(
                self.compiled_model.get_tensor(output_details[0]["index"])
            )
            logits.append(self.compiled_model.get_tensor(output_details[1]["index"]))
        return np.stack(logits).squeeze(), np.stack(embeddings).squeeze()

    def predict(
        self, path: str, window: int = 5 * 32_000, **kwargs
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        audio = self.load(path, window).numpy()
        if self.use_compiled:
            logits, embeddings = self._infer_tflite(audio)
        else:
            logits, embeddings = self.model.infer_tf(audio)
            logits = logits.numpy()
            embeddings = embeddings.numpy()

        # extract relevant logits, using -inf as the base condition
        neg_inf = np.ones((audio.shape[0], 1)) * -np.inf
        logits = np.concatenate([logits, neg_inf], axis=1)
        logits = logits[:, self.model_indices]

        return embeddings, logits
