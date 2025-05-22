from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class EmbeddingMerger:
    def __init__(self, model_name='all-mpnet-base-v2', similarity_threshold=0.9):
        """
        Initialize the embedding model and clustering threshold.
        :param model_name: Pretrained sentence transformer model name.
        :param similarity_threshold: Cosine similarity threshold for merging.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def merge_similar_texts(self, texts):
        """
        Merge similar texts based on embedding similarity.
        :param texts: List of strings to merge.
        :return: List of merged representative texts.
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_tensor=False)
        embeddings = np.array(embeddings)

        # Compute cosine similarity matrix
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix

        # Agglomerative clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=1 - self.similarity_threshold
        )
        clustering.fit(distance_matrix)

        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(texts[idx])

        # For each cluster, pick the most common text as representative
        merged_texts = []
        for cluster_texts in clusters.values():
            # Simple heuristic: pick the longest text as representative
            representative = max(cluster_texts, key=len)
            merged_texts.append(representative)

        return merged_texts

    def merge_similar_texts_with_indices(self, texts):
        """
        Merge similar texts and return cluster indices for bounding box association.
        :param texts: List of strings to merge.
        :return: Tuple (merged_texts, cluster_indices)
            merged_texts: List of merged representative texts.
            cluster_indices: List of lists of original indices in each cluster.
        """
        if not texts:
            return [], []

        # Preprocess texts to separate numeric suffixes for special handling
        base_texts = []
        suffixes = []
        for text in texts:
            parts = text.rsplit(' ', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_texts.append(parts[0])
                suffixes.append(parts[1])
            else:
                base_texts.append(text)
                suffixes.append(None)

        embeddings = self.model.encode(base_texts, convert_to_tensor=False)
        embeddings = np.array(embeddings)

        # Compute cosine similarity matrix
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix

        # Agglomerative clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=1 - self.similarity_threshold
        )
        clustering.fit(distance_matrix)

        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(idx)

        merged_texts = []
        cluster_indices = []
        for label, indices in clusters.items():
            cluster_texts = []
            for i in indices:
                if suffixes[i]:
                    cluster_texts.append(f"{base_texts[i]} {suffixes[i]}")
                else:
                    cluster_texts.append(base_texts[i])
            # Pick the longest text as representative
            representative = max(cluster_texts, key=len)
            merged_texts.append(representative)
            cluster_indices.append(indices)

        return merged_texts, cluster_indices
