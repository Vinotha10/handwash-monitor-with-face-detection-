import cv2
import os
import numpy as np
from logging import getLogger
from onnxruntime import InferenceSession

from face_recognition.utils.helpers import face_alignment

__all__ = ["ArcFace"]

logger = getLogger(__name__)


class ArcFace:
    """
    ArcFace Model for Face Recognition
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ArcFace face encoder model.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(
            os.path.join(script_dir, "../weights", os.path.basename(model_path))
        )

        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5

        logger.info(f"Initializing ArcFace model from {self.model_path}")

        try:
            self.session = InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Correctly get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        except Exception as e:
            logger.error(f"Failed to load face encoder model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image: resize, normalize, and convert to the required format.
        """
        resized_face = cv2.resize(face_image, self.input_size)

        # Use blobFromImage to normalize and convert to NCHW format
        face_blob = cv2.dnn.blobFromImage(
            resized_face,
            scalefactor=1.0 / self.normalization_scale,
            size=self.input_size,
            mean=(self.normalization_mean,) * 3,
            swapRB=True
        )
        return face_blob

    def get_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        normalized: bool = False
    ) -> np.ndarray:
        """
        Extract face embedding from an image using facial landmarks for alignment.
        """
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        try:
            aligned_face, _ = face_alignment(image, landmarks)
            face_blob = self.preprocess(aligned_face)

            # Run inference
            embedding = self.session.run([self.output_name], {self.input_name: face_blob})[0]

            if normalized:
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                embedding = embedding / norm

            return embedding.flatten()

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            raise
