import os
import trimesh
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from config.settings import Settings
from utils.math_utils import normalize_mesh_vertices

logger = logging.getLogger(__name__)

@dataclass
class MeshData:
    vertices: np.ndarray    # (N, 3) float32 - unit normalized
    normals: np.ndarray     # (N, 3) float32 - normal vectors
    indices: np.ndarray     # (M,) uint32 - flat index buffer
    name: str
    vertex_count: int
    index_count: int

class ModelLoader:
    """Recursively scans the configured model directory for supported 3D models.
    Provides functionality to load, normalize, and cache meshes. Includes a fallback
    procedural shape generator if no assets are found.
    """
    
    def __init__(self, settings: Settings):
        self._model_dir = settings.model_directory
        self._supported_extensions = {".obj", ".glb", ".gltf", ".ply", ".stl", ".dae", ".fbx"}
        self._model_paths: List[str] = []
        self._cache = {}

    def scan_directory(self) -> List[str]:
        """Scans assets directory for supported model files. Returns list of absolute paths."""
        logger.info(f"Scanning directory: {self._model_dir} for 3D models...")
        self._model_paths = []
        
        if not os.path.exists(self._model_dir):
            try:
                os.makedirs(self._model_dir, exist_ok=True)
                logger.info(f"Created assets directory: {self._model_dir}")
            except Exception as e:
                logger.error(f"Failed to create assets directory: {e}")
                return []

        for root, _, files in os.walk(self._model_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self._supported_extensions:
                    full_path = os.path.join(root, file)
                    self._model_paths.append(os.path.abspath(full_path))
                    
        self._model_paths.sort()
        logger.info(f"Found {len(self._model_paths)} supported models.")
        return self._model_paths

    def load_model(self, index_or_path) -> Optional[MeshData]:
        """Loads a model by index or by absolute file path.
        Automatically normalizes vertices to unit scale and centers them.
        Caches results to avoid reload lag.
        """
        path = ""
        if isinstance(index_or_path, int):
            if not self._model_paths:
                self.scan_directory()
            if not self._model_paths:
                return None
            path = self._model_paths[index_or_path % len(self._model_paths)]
        else:
            path = os.path.abspath(index_or_path)

        if path in self._cache:
            return self._cache[path]

        if not os.path.exists(path):
            logger.error(f"Model path does not exist: {path}")
            return None

        logger.info(f"Loading 3D model: {path}...")
        try:
            # Load mesh with trimesh. Force single mesh representation.
            scene_or_mesh = trimesh.load(path)
            
            # If loaded object is a Scene, concatenate all sub-meshes
            if isinstance(scene_or_mesh, trimesh.Scene):
                if len(scene_or_mesh.geometry) == 0:
                    logger.warning(f"Scene contains no geometry: {path}")
                    return None
                mesh = trimesh.util.concatenate(
                    [geom for geom in scene_or_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                )
            else:
                mesh = scene_or_mesh

            if not isinstance(mesh, trimesh.Trimesh):
                logger.error(f"Asset is not a valid mesh format: {path}")
                return None

            # Get raw geometries
            vertices = mesh.vertices
            indices = mesh.faces.flatten()
            
            # Verify normals exist or compute them
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
                normals = mesh.vertex_normals
            else:
                mesh.create_vertex_normals()
                normals = mesh.vertex_normals

            # Normalize to unit cube centered at origin
            norm_vertices, scale, offset = normalize_mesh_vertices(vertices)
            
            mesh_data = MeshData(
                vertices=norm_vertices.astype(np.float32),
                normals=normals.astype(np.float32),
                indices=indices.astype(np.uint32),
                name=os.path.basename(path),
                vertex_count=len(norm_vertices),
                index_count=len(indices)
            )
            
            self._cache[path] = mesh_data
            logger.info(f"Successfully loaded and normalized {mesh_data.name} (Vertices: {mesh_data.vertex_count})")
            return mesh_data

        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}", exc_info=True)
            return None

    def load_default_model(self) -> MeshData:
        """Generates a procedural glowing icosphere to use as a fallback
        when no files are found in the assets folder.
        """
        logger.warning("No custom assets found. Generating default icosphere.")
        # Create subdivision level 3 icosphere
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        
        vertices = mesh.vertices
        indices = mesh.faces.flatten()
        normals = mesh.vertex_normals
        
        # Already normalized center-wise, but run the utility to be safe
        norm_vertices, _, _ = normalize_mesh_vertices(vertices)
        
        return MeshData(
            vertices=norm_vertices.astype(np.float32),
            normals=normals.astype(np.float32),
            indices=indices.astype(np.uint32),
            name="Default Holographic Sphere",
            vertex_count=len(norm_vertices),
            index_count=len(indices)
        )

    @property
    def model_paths(self) -> List[str]:
        return self._model_paths

    @property
    def model_count(self) -> int:
        return len(self._model_paths)
