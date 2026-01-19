import logging
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    """
    Local embedding model using sentence-transformers.
    Recommended model: 'sentence-transformers/all-mpnet-base-v2' (768 dim, high quality)
    Alternative: 'sentence-transformers/all-MiniLM-L6-v2' (384 dim, faster, smaller)
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        logger: logging.Logger = None
    ):
        """
        Initialize local embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model from HuggingFace
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            logger: Logger instance for logging
        """
        self.model_name = model_name
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Loading local embedding model: {model_name}")
        else:
            print(f"Loading local embedding model: {model_name}")
        
        # Auto-detect device if not specified
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
        if self.logger:
            self.logger.info(f"Local embedding model loaded on device: {device}")
        else:
            print(f"Local embedding model loaded on device: {device}")
    
    def encode(self, text):
        """
        Encode text into embeddings.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Single numpy array (if single input) or list of numpy arrays
        """
        try:
            single_input = False
            if isinstance(text, str):
                text = [text]
                single_input = True
            
            # Encode using sentence-transformers
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False  # Keep original scale for compatibility
            )
            
            # Convert to list of numpy arrays
            if isinstance(embeddings, np.ndarray):
                if len(embeddings.shape) == 1:
                    embeddings = [embeddings]
                else:
                    embeddings = [emb for emb in embeddings]
            
            # Ensure float32 for compatibility with FAISS
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings]
            
            if single_input and len(embeddings) == 1:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Local embedding error: {e}", exc_info=True)
            else:
                print(f"Local embedding error: {e}")
            return None
