"""
Mutable mesh data structure to create and modify room geometries.
These are then converted to PyTorch tensor for optimization.
"""

from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np


class Mesh:
    def __init__(self):
        self.vertex_coords = []
        self.vertex_coord_to_id = {}
        self.patch_vertex_ids = {}
        self.object_ids = {}
        self.counter = 0

    def __str__(self):
        strings = []
        main = f"Mesh with {len(self.vertex_coords)} vertices and {len(self.patch_vertex_ids)} patches"
        strings.append(main)
        for i, v in enumerate(self.vertex_coords):
            strings.append(f"v{i}: {v}")
        for i, (p, v) in enumerate(self.patch_vertex_ids.items()):
            strings.append(f"f{i}: {p} ({v})")
        return "\n".join(strings)

    def update_vertex_coords_and_get_ids(self, v_list):
        ids = []
        for v in v_list:
            if not isinstance(v, tuple):
                v = tuple(v)
            v = tuple([round(u, 4) for u in v])
            if v not in self.vertex_coord_to_id:
                id = len(self.vertex_coords)
                self.vertex_coord_to_id[v] = id
                self.vertex_coords.append(v)
            else:
                id = self.vertex_coord_to_id[v]
            ids.append(id)
        return ids

    def add_patch_with_vertex_ids(self, *v, object_id=None):
        keys = [self.sort_vertex_ids(p) for p in v]

        for k, p in zip(keys, v):
            if k not in self.patch_vertex_ids:
                self.patch_vertex_ids[k] = p
                if object_id is not None:
                    self.object_ids[k] = object_id
                else:
                    self.object_ids[k] = self.counter
                    self.counter = max(self.object_ids.values()) + 1
            else:
                self.patch_vertex_ids.pop(k)
                self.object_ids.pop(k)

    def add_triangle_with_coords(self, v1, v2, v3, **kwargs):
        v_list = [v1, v2, v3]
        i_list = self.update_vertex_coords_and_get_ids(v_list)
        i1, i2, i3 = i_list
        self.add_patch_with_vertex_ids((i1, i2, i3), **kwargs)

    def add_parallelogram_with_coords(self, v1, v2, v3, **kwargs):
        v1, v2, v3 = map(np.array, [v1, v2, v3])
        v4 = v2 + v3 - v1
        v_list = [v1, v2, v3, v4]
        v_list = [tuple(u) for u in v_list]
        i_list = self.update_vertex_coords_and_get_ids(v_list)
        i1, i2, i3, i4 = i_list
        self.add_patch_with_vertex_ids((i1, i2, i4, i3), **kwargs)

    def add_parallelogram(self, v1, u2, u3, **kwargs):
        v1, u2, u3 = map(lambda x: np.array(x, dtype=float), [v1, u2, u3])
        v2 = v1 + u2
        v3 = v1 + u3
        self.add_parallelogram_with_coords(v1, v2, v3, **kwargs)

    def sort_vertex_ids(self, patch_vertex_ids):
        patch_vertex_ids = list(patch_vertex_ids)
        n = len(patch_vertex_ids)
        min_id = min(patch_vertex_ids)
        i = patch_vertex_ids.index(min_id)
        patch_vertex_ids = patch_vertex_ids[i:] + patch_vertex_ids[:i]
        if patch_vertex_ids[1] < patch_vertex_ids[-1]:
            patch_vertex_ids = [patch_vertex_ids[0]] + patch_vertex_ids[1:][::-1]
        return tuple(patch_vertex_ids)

    def add_parallelotope_with_coords(self, v, u1, u2, u3, **kwargs):
        v, u1, u2, u3 = map(lambda x: np.array(x, dtype=float), [v, u1, u2, u3])

        v1 = v + u1
        v2 = v + u2
        v3 = v + u3

        v12 = v1 + u2
        v13 = v1 + u3
        v23 = v2 + u3
        v123 = v1 + u2 + u3

        v_list = [v, v1, v2, v3, v12, v13, v23, v123]
        v_list = [tuple(u) for u in v_list]
        i_list = self.update_vertex_coords_and_get_ids(v_list)
        i, i1, i2, i3, i12, i13, i23, i123 = i_list

        parallelograms = [
            (i, i1, i12, i2),
            (i, i3, i13, i1),
            (i, i2, i23, i3),
            (i123, i12, i1, i13),
            (i123, i23, i2, i12),
            (i123, i13, i3, i23),
        ]

        self.add_patch_with_vertex_ids(*parallelograms, **kwargs)

    def add_shoebox_with_coords(self, v, x, y, z):
        self.add_parallelotope_with_coords(v, [x, 0, 0], [0, y, 0], [0, 0, z])

    def remove_patch_with_vertex_coords(self, patch_vertex_coords):
        patch_vertex_ids = set()
        for patch in self.patch_vertex_ids:
            if self.compare_patch(patch, patch_vertex_coords):
                patch_vertex_ids.add(patch)
        self.remove_patch_with_vertex_ids(patch_vertex_ids)

    def remove_patch_with_vertex_ids(self, patch_vertex_ids, ignore_orientation=True):
        for patch in self.patch_vertex_ids:
            if self.compare_patch(patch, patch_vertex_ids, ignore_orientation):
                self.patch_vertex_ids.pop(patch)
                self.object_ids.pop(patch)
                break

    def compare_patch(self, a, b, ignore_orientation=True):
        if ignore_orientation:
            return set(a) == set(b)
        else:
            if set(a) == set(b):
                anchor = a[0]
                if a[1] == b[1 + b.index(anchor)]:
                    return True
                else:
                    return False
            else:
                return False

    def remove_unused_vertices(self):
        raise NotImplementedError

    def consistent_winding(self):
        def check_winding(patch, connected, shared_edge):
            patch_ids = [patch.index(i) for i in shared_edge]
            connected_ids = [connected.index(i) for i in shared_edge]
            patch_direction = (patch_ids[1] - patch_ids[0]) % len(patch)
            connected_direction = (connected_ids[1] - connected_ids[0]) % len(connected)
            flip = patch_direction == connected_direction
            return flip

        G = construct_patch_connectivity_graph(self.patch_vertex_ids)
        anchor = next(iter(G.nodes))
        for patch_id, connected_ids in nx.bfs_successors(G, anchor):
            patch = self.patch_vertex_ids[patch_id]
            for connected_id in connected_ids:
                connected = self.patch_vertex_ids[connected_id]
                shared_edge = set(patch) & set(connected)
                if check_winding(patch, connected, shared_edge):
                    self.patch_vertex_ids[connected_id] = connected[::-1]

    @property
    def num_vertices(self):
        return len(self.vertex_coords)

    @property
    def num_patches(self):
        return len(self.patch_vertex_ids)

    @property
    def next_id(self):
        return 1 + len(self.object_ids.values())


def construct_patch_connectivity_graph(patch_vertex_ids):
    edge_to_face = defaultdict(list)
    for id, v in patch_vertex_ids.items():
        num_edges = len(v)
        edges = [(v[i], v[(i + 1) % num_edges]) for i in range(num_edges)]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))
            edge_to_face[sorted_edge].append(id)
    G = nx.Graph()
    for edge, faces in edge_to_face.items():
        assert len(faces) < 3
        if len(faces) == 2:
            u, v = faces
            G.add_edge(u, v)
    return G


def split_parallelogram_patches(mesh, max_length=2):
    split_mesh = Mesh()

    for k, patch in mesh.patch_vertex_ids.items():
        object_id = mesh.object_ids[k]
        patch_coords = [mesh.vertex_coords[i] for i in patch]
        if len(patch) == 3:
            ids = split_mesh.update_vertex_coords_and_get_ids(patch_coords)
            split_mesh.add_patch_with_vertex_ids(ids, object_id=object_id)
        elif len(patch) == 4:
            v1, v2, _, v3 = np.array(patch_coords)
            u1 = v2 - v1
            u2 = v3 - v1
            u1_len = np.linalg.norm(u1)
            u2_len = np.linalg.norm(u2)
            n = int(np.ceil(u1_len / max_length))
            m = int(np.ceil(u2_len / max_length))
            for i in range(n):
                for j in range(m):
                    vij1 = v1 + i * u1 / n + j * u2 / m
                    vij2 = vij1 + u1 / n
                    vij3 = vij1 + u2 / m
                    split_mesh.add_parallelogram_with_coords(
                        vij1, vij2, vij3, object_id=object_id
                    )

    return split_mesh


def triangulate_mesh(mesh, split_method="center"):
    r"""
    Returns a triangular mesh from a mesh with triangles and parallelograms.

    Args:
        mesh (:python:`Mesh`):
            Mesh with triangles and parallelograms
        split_method (:python:`str`, *optional*):
            Method to split parallelograms. Options are :python:`"center"` and :python:`"diagonal"`,
            where the former splits the parallelogram into four triangles by ading a vertex at the center
            and the latter splits the parallelogram into two triangles with a diagonal
            (default: :python:`"center"`).
    """
    tri_mesh = Mesh()

    for k, patch in mesh.patch_vertex_ids.items():
        object_id = mesh.object_ids[k]
        patch_coords = [mesh.vertex_coords[i] for i in patch]
        if len(patch) == 3:
            ids = tri_mesh.update_vertex_coords_and_get_ids(patch_coords)
            tri_mesh.add_patch_with_vertex_ids(ids, object_id=object_id)
        elif len(patch) == 4:
            midpoint = list(np.mean(np.array(patch_coords), axis=0))
            patch_coords.append(midpoint)
            ids = tri_mesh.update_vertex_coords_and_get_ids(patch_coords)
            tri_mesh.add_patch_with_vertex_ids(
                [ids[0], ids[1], ids[4]],
                [ids[1], ids[2], ids[4]],
                [ids[2], ids[3], ids[4]],
                [ids[3], ids[0], ids[4]],
                object_id=object_id,
            )
    return tri_mesh


def perturb_mesh(mesh, scale=0.01, seed=0):
    print(f"Geometry distortion of scale {scale}")
    mesh = deepcopy(mesh)
    rng = np.random.default_rng(seed)
    for i, v in enumerate(mesh.vertex_coords):
        mesh.vertex_coords[i] = v + rng.uniform(-scale, scale, size=3)
    return mesh