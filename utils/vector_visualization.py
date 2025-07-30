"""
Vector Database Visualization - Dimensionality Reduction & Clustering
====================================================================

Visualize high-dimensional embeddings in 2D/3D using:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

With clustering algorithms:
- K-Means
- DBSCAN
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class VectorVisualizer:
    """Visualize vector embeddings with dimensionality reduction"""

    def __init__(self):
        """Initialize visualizer"""
        self.available_methods = ['pca', 'tsne', 'umap']
        self.available_dimensions = [2, 3]
        logger.info("VectorVisualizer initialized")

    def visualize(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        method: str = 'pca',
        n_dimensions: int = 2,
        cluster_method: Optional[str] = None,
        n_clusters: int = 5,
        perplexity: int = 30,
        n_neighbors: int = 15
    ) -> Dict:
        """
        Visualize embeddings with dimensionality reduction

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            metadata: List of metadata dicts for each embedding
            method: 'pca', 'tsne', or 'umap'
            n_dimensions: 2 or 3
            cluster_method: 'kmeans' or 'dbscan' (optional)
            n_clusters: Number of clusters for k-means
            perplexity: Perplexity for t-SNE
            n_neighbors: Number of neighbors for UMAP

        Returns:
            Dict with reduced coordinates, metadata, and cluster labels
        """
        logger.info(f"Visualizing {len(embeddings)} embeddings using {method} ({n_dimensions}D)")

        # Validate inputs
        if method not in self.available_methods:
            raise ValueError(f"Method must be one of {self.available_methods}")
        if n_dimensions not in self.available_dimensions:
            raise ValueError(f"Dimensions must be one of {self.available_dimensions}")

        # Apply dimensionality reduction
        if method == 'pca':
            reduced = self._apply_pca(embeddings, n_dimensions)
        elif method == 'tsne':
            reduced = self._apply_tsne(embeddings, n_dimensions, perplexity)
        elif method == 'umap':
            reduced = self._apply_umap(embeddings, n_dimensions, n_neighbors)

        # Apply clustering if requested
        cluster_labels = None
        if cluster_method:
            if cluster_method == 'kmeans':
                cluster_labels = self._apply_kmeans(embeddings, n_clusters)
            elif cluster_method == 'dbscan':
                cluster_labels = self._apply_dbscan(embeddings)

        # Format results
        result = self._format_results(
            reduced, metadata, cluster_labels, method, n_dimensions
        )

        logger.info(f"✅ Visualization complete: {len(result['points'])} points")
        return result

    def _apply_pca(self, embeddings: np.ndarray, n_components: int) -> np.ndarray:
        """Apply PCA dimensionality reduction"""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)

        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        total_variance = sum(explained_variance)
        logger.info(f"PCA explained variance: {total_variance:.2%}")

        return reduced

    def _apply_tsne(
        self, embeddings: np.ndarray, n_components: int, perplexity: int
    ) -> np.ndarray:
        """Apply t-SNE dimensionality reduction"""
        from sklearn.manifold import TSNE

        # Adjust perplexity if needed
        n_samples = embeddings.shape[0]
        perplexity = min(perplexity, n_samples - 1, 30)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            verbose=0
        )
        reduced = tsne.fit_transform(embeddings)

        logger.info(f"t-SNE complete (perplexity={perplexity})")
        return reduced

    def _apply_umap(
        self, embeddings: np.ndarray, n_components: int, n_neighbors: int
    ) -> np.ndarray:
        """Apply UMAP dimensionality reduction"""
        try:
            import umap

            # Adjust n_neighbors if needed
            n_samples = embeddings.shape[0]
            n_neighbors = min(n_neighbors, n_samples - 1, 15)

            # For small datasets (<50 points), use faster parameters
            if n_samples < 50:
                n_neighbors = min(n_neighbors, max(2, n_samples // 3))
                n_epochs = 200  # Reduced from default 500 for speed
                min_dist = 0.1
            else:
                n_epochs = 500
                min_dist = 0.1

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_epochs=n_epochs,
                metric='cosine',  # Better for embeddings
                random_state=42,
                low_memory=True,  # Better for small datasets
                verbose=False
            )
            reduced = reducer.fit_transform(embeddings)

            logger.info(f"UMAP complete (n_neighbors={n_neighbors}, n_epochs={n_epochs})")
            return reduced

        except ImportError:
            logger.warning("UMAP not installed, falling back to PCA")
            return self._apply_pca(embeddings, n_components)

    def _apply_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Apply K-means clustering"""
        from sklearn.cluster import KMeans

        # Adjust n_clusters if needed
        n_samples = embeddings.shape[0]
        n_clusters = min(n_clusters, n_samples)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        logger.info(f"K-means clustering: {n_clusters} clusters")
        return labels

    def _apply_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply DBSCAN clustering"""
        from sklearn.cluster import DBSCAN

        # Use euclidean distance
        dbscan = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
        labels = dbscan.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
        return labels

    def _format_results(
        self,
        reduced: np.ndarray,
        metadata: List[Dict],
        cluster_labels: Optional[np.ndarray],
        method: str,
        n_dimensions: int
    ) -> Dict:
        """Format visualization results for API response"""

        points = []
        for i, coords in enumerate(reduced):
            meta = metadata[i] if i < len(metadata) else {}

            point = {
                'id': i,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'title': meta.get('title', 'Unknown')[:100],
                'source': meta.get('source', 'Unknown'),
                'paper_id': meta.get('paper_id', ''),
                'text': meta.get('text', '')[:200]
            }

            # Add z coordinate for 3D
            if n_dimensions == 3:
                point['z'] = float(coords[2])

            # Add cluster label
            if cluster_labels is not None:
                point['cluster'] = int(cluster_labels[i])

            points.append(point)

        # Calculate statistics
        stats = {
            'total_points': len(points),
            'dimensions': n_dimensions,
            'method': method,
            'sources': self._count_sources(metadata),
        }

        if cluster_labels is not None:
            stats['n_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            stats['cluster_sizes'] = self._count_clusters(cluster_labels)

        return {
            'points': points,
            'stats': stats,
            'config': {
                'method': method,
                'dimensions': n_dimensions
            }
        }

    def _count_sources(self, metadata: List[Dict]) -> Dict[str, int]:
        """Count papers by source"""
        sources = {}
        for meta in metadata:
            source = meta.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources

    def _count_clusters(self, labels: np.ndarray) -> Dict[int, int]:
        """Count points per cluster"""
        clusters = {}
        for label in labels:
            label = int(label)
            clusters[label] = clusters.get(label, 0) + 1
        return clusters

    def create_plotly_figure(
        self,
        visualization_data: Dict,
        title: str = "Vector Space Visualization"
    ) -> str:
        """
        Create interactive Plotly figure (JSON format)

        Returns JSON string that can be used with Plotly.js
        """
        try:
            import plotly.graph_objects as go

            points = visualization_data['points']
            config = visualization_data['config']

            # Prepare data
            x = [p['x'] for p in points]
            y = [p['y'] for p in points]
            titles = [p['title'] for p in points]
            sources = [p['source'] for p in points]

            # Color by cluster if available, otherwise by source
            if 'cluster' in points[0]:
                colors = [p['cluster'] for p in points]
                color_label = 'Cluster'
            else:
                # Map sources to numbers for coloring
                unique_sources = list(set(sources))
                source_to_num = {s: i for i, s in enumerate(unique_sources)}
                colors = [source_to_num[s] for s in sources]
                color_label = 'Source'

            # Create figure based on dimensions
            if config['dimensions'] == 3:
                z = [p['z'] for p in points]

                fig = go.Figure(data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_label),
                        line=dict(width=0.5, color='white')
                    ),
                    text=titles,
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Source: %{customdata[0]}<br>' +
                                  'x: %{x:.2f}<br>' +
                                  'y: %{y:.2f}<br>' +
                                  'z: %{z:.2f}<br>' +
                                  '<extra></extra>',
                    customdata=[[s] for s in sources]
                )])

                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=f'{config["method"].upper()} 1',
                        yaxis_title=f'{config["method"].upper()} 2',
                        zaxis_title=f'{config["method"].upper()} 3',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    autosize=True,
                    width=None,
                    height=None,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

            else:  # 2D
                fig = go.Figure(data=[go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_label),
                        line=dict(width=0.5, color='white')
                    ),
                    text=titles,
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Source: %{customdata[0]}<br>' +
                                  'x: %{x:.2f}<br>' +
                                  'y: %{y:.2f}<br>' +
                                  '<extra></extra>',
                    customdata=[[s] for s in sources]
                )])

                fig.update_layout(
                    title=title,
                    xaxis_title=f'{config["method"].upper()} 1',
                    yaxis_title=f'{config["method"].upper()} 2',
                    autosize=True,
                    width=None,
                    height=None,
                    margin=dict(l=50, r=50, t=60, b=50),
                    hovermode='closest',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15,15,25,0.5)'
                )

            # Convert to JSON
            return fig.to_json()

        except ImportError as e:
            logger.warning(f"Plotly not installed: {e}, returning raw data")
            return json.dumps(visualization_data)
        except Exception as e:
            logger.error(f"Error creating Plotly figure: {e}", exc_info=True)
            return json.dumps(visualization_data)

    def get_embedding_statistics(self, embeddings: np.ndarray) -> Dict:
        """Calculate statistics about embeddings"""
        return {
            'n_embeddings': int(embeddings.shape[0]),
            'embedding_dim': int(embeddings.shape[1]),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            'min_value': float(np.min(embeddings)),
            'max_value': float(np.max(embeddings)),
            'mean_value': float(np.mean(embeddings)),
            'std_value': float(np.std(embeddings))
        }
