from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union


DEFAULT_SETTINGS = {
    "paths": {
        "chm_raster": "CHM_Asa.tif",
        "dtw_raster": "DTW_Asa.tif",
        "species_raster": "Asa_tree_species_classification_MajFilter3m.tif",
        "watercourse_shapefile": "Modellerade_Vattendrag_30ha_Bestand_over25ar.shp",
        "output_dir": "outputs",
    },
    "model": {
        "resolution_m": 1.0,
        "max_buffer_width_m": 30.0,
        "lambda_ecology_weight": 1.0,
        "mu_boundary_weight": 20.0,
        "sun_altitude_deg": 30.0,
        "sun_azimuth_deg": 180.0,
        "zoom_regions": 4,
        "scenario_lambdas": {
            "max_ecology": 0.0,
            "balanced_low": 1.0,
            "balanced": 5.0,
            "balanced_high": 15.0,
            "max_economy": 50.0,
        },
    },
    "species": {
        "codes": {"1": "broadleaf", "2": "pine", "3": "spruce"},
        "volume_coefficients": {"broadleaf": 0.9e-3, "pine": 1.1e-3, "spruce": 1.3e-3},
        "prices_sek_per_m3": {"broadleaf": 380.0, "pine": 520.0, "spruce": 550.0},
        "ecological_scores": {"broadleaf": 1.0, "pine": 0.55, "spruce": 0.30},
    },
    "ecology": {
        "component_weights": {"dtw": 1.0, "shade": 1.4, "species": 1.0, "proximity": 0.8},
        "eco_unit_sek": 3.0,
    },
}

INF_CAPACITY = 1.0e12


@dataclass
class Grid:
    transform: rasterio.Affine
    width: int
    height: int
    crs: object
    pixel_size: float


@dataclass
class ProjectPaths:
    chm_raster: Path
    dtw_raster: Path
    species_raster: Path
    watercourse_shapefile: Path
    output_dir: Path


@dataclass
class ModelSettings:
    resolution_m: float
    max_buffer_width_m: float
    lambda_ecology_weight: float
    mu_boundary_weight: float
    sun_altitude_deg: float
    sun_azimuth_deg: float
    zoom_regions: int
    scenario_lambdas: dict[str, float]


@dataclass
class SpeciesSettings:
    codes: dict[int, str]
    volume_coefficients: dict[str, float]
    prices_sek_per_m3: dict[str, float]
    ecological_scores: dict[str, float]


@dataclass
class EcologySettings:
    component_weights: dict[str, float]
    eco_unit_sek: float


@dataclass
class Settings:
    paths: ProjectPaths
    model: ModelSettings
    species: SpeciesSettings
    ecology: EcologySettings


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: Path | None) -> Settings:
    raw = DEFAULT_SETTINGS
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = deep_update(DEFAULT_SETTINGS, json.load(handle))

    paths = ProjectPaths(
        chm_raster=Path(raw["paths"]["chm_raster"]),
        dtw_raster=Path(raw["paths"]["dtw_raster"]),
        species_raster=Path(raw["paths"]["species_raster"]),
        watercourse_shapefile=Path(raw["paths"]["watercourse_shapefile"]),
        output_dir=Path(raw["paths"]["output_dir"]),
    )
    model = ModelSettings(**raw["model"])
    species = SpeciesSettings(
        codes={int(key): value for key, value in raw["species"]["codes"].items()},
        volume_coefficients=raw["species"]["volume_coefficients"],
        prices_sek_per_m3=raw["species"]["prices_sek_per_m3"],
        ecological_scores=raw["species"]["ecological_scores"],
    )
    ecology = EcologySettings(**raw["ecology"])
    return Settings(paths=paths, model=model, species=species, ecology=ecology)


def build_target_grid(reference_raster: Path, resolution_m: float) -> Grid:
    with rasterio.open(reference_raster) as dataset:
        left, bottom, right, top = dataset.bounds
        crs = dataset.crs
    width = int(math.floor((right - left) / resolution_m))
    height = int(math.floor((top - bottom) / resolution_m))
    transform = from_origin(left, top, resolution_m, resolution_m)
    return Grid(transform=transform, width=width, height=height, crs=crs, pixel_size=resolution_m)


def reproject_to_grid(
    raster_path: Path,
    grid: Grid,
    resampling: Resampling,
    nodata_out=None,
    dtype=np.float32,
) -> np.ndarray:
    if nodata_out is None:
        nodata_out = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    output = np.full((grid.height, grid.width), nodata_out, dtype=dtype)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=output,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=nodata_out,
        )
    return output


def load_watercourse(shapefile_path: Path, target_crs) -> MultiLineString:
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)
    return unary_union(list(gdf.geometry))


def rasterize_lines(geometry, grid: Grid) -> np.ndarray:
    shapes = [(geometry, 1)]
    water_mask = rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )
    return water_mask.astype(bool)


def shade_first_hit(water_mask: np.ndarray, sun_azimuth_deg: float, max_steps: int) -> np.ndarray:
    azimuth_rad = math.radians(sun_azimuth_deg)
    shadow_dx = -math.sin(azimuth_rad)
    shadow_dy = -math.cos(azimuth_rad)
    d_col = shadow_dx
    d_row = -shadow_dy

    first_hit = np.zeros(water_mask.shape, dtype=np.int32)
    for step in range(1, max_steps + 1):
        dr = int(round(step * d_row))
        dc = int(round(step * d_col))
        shifted = np.zeros_like(water_mask)
        r0_src = max(dr, 0)
        r1_src = water_mask.shape[0] + min(dr, 0)
        c0_src = max(dc, 0)
        c1_src = water_mask.shape[1] + min(dc, 0)
        r0_dst = r0_src - dr
        r1_dst = r1_src - dr
        c0_dst = c0_src - dc
        c1_dst = c1_src - dc
        shifted[r0_dst:r1_dst, c0_dst:c1_dst] = water_mask[r0_src:r1_src, c0_src:c1_src]
        first_hit[(shifted) & (first_hit == 0)] = step
    return first_hit


def compute_pixel_values(
    chm: np.ndarray,
    species_raster: np.ndarray,
    dtw: np.ndarray,
    water_mask: np.ndarray,
    distance_to_water: np.ndarray,
    valid_mask: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_area = settings.model.resolution_m ** 2
    max_shadow_steps = int(
        math.ceil(40.0 / math.tan(math.radians(settings.model.sun_altitude_deg)) / settings.model.resolution_m)
    )
    first_water_hit = shade_first_hit(water_mask, settings.model.sun_azimuth_deg, max_shadow_steps)

    chm_values = chm[valid_mask].astype(np.float32, copy=True)
    np.clip(chm_values, 0.0, 40.0, out=chm_values)
    chm_values[chm_values < 3.0] = 0.0

    species_values = species_raster[valid_mask].astype(np.int16, copy=True)
    known_codes = set(settings.species.codes)
    species_values[~np.isin(species_values, list(known_codes))] = 0

    dtw_values = dtw[valid_mask].astype(np.float32, copy=True)
    dtw_values[~np.isfinite(dtw_values) | (dtw_values > 1e6)] = 20.0
    np.clip(dtw_values, 0.0, 20.0, out=dtw_values)

    dist_values = distance_to_water[valid_mask].astype(np.float32, copy=False)
    shade_hit_values = first_water_hit[valid_mask]

    revenue = np.zeros_like(chm_values, dtype=np.float32)
    for code, species_name in settings.species.codes.items():
        mask = species_values == code
        if not mask.any():
            continue
        revenue[mask] = (
            settings.species.volume_coefficients[species_name]
            * chm_values[mask]
            * pixel_area
            * settings.species.prices_sek_per_m3[species_name]
        ).astype(np.float32)

    dtw_score = np.exp(-dtw_values / 2.0)
    max_shadow_reach = chm_values / math.tan(math.radians(settings.model.sun_altitude_deg)) / settings.model.resolution_m
    shade_score = ((shade_hit_values > 0) & (shade_hit_values <= max_shadow_reach)).astype(np.float32)

    species_score = np.zeros_like(chm_values)
    for code, species_name in settings.species.codes.items():
        species_score[species_values == code] = settings.species.ecological_scores[species_name]
    species_score *= np.clip(chm_values / 20.0, 0.0, 1.0)

    proximity_score = np.exp(-np.clip(dist_values, 0, settings.model.max_buffer_width_m) / 10.0)
    weights = settings.ecology.component_weights
    ecological_loss = (
        weights["dtw"] * dtw_score
        + weights["shade"] * shade_score
        + weights["species"] * species_score
        + weights["proximity"] * proximity_score
    ) * settings.ecology.eco_unit_sek * pixel_area

    return revenue.astype(np.float32), ecological_loss.astype(np.float32)


def solve_graph_cut(
    revenue: np.ndarray,
    ecological_loss: np.ndarray,
    distance_to_water: np.ndarray,
    valid_mask: np.ndarray,
    lam: float,
    mu: float,
) -> np.ndarray:
    rows, cols = valid_mask.shape
    index_grid = -np.ones((rows, cols), dtype=np.int32)
    n_pixels = int(valid_mask.sum())
    index_grid[valid_mask] = np.arange(n_pixels, dtype=np.int32)

    graph = maxflow.Graph[float](n_pixels, 6 * n_pixels)
    graph.add_nodes(n_pixels)

    gain = revenue - lam * ecological_loss
    source_caps = np.maximum(gain, 0.0).astype(np.float64)
    sink_caps = np.maximum(-gain, 0.0).astype(np.float64)
    for idx in range(n_pixels):
        graph.add_tedge(idx, float(source_caps[idx]), float(sink_caps[idx]))

    def add_edges(a_idx, b_idx, cap_forward, cap_reverse) -> None:
        if a_idx.size == 0:
            return
        graph.add_edges(
            a_idx.astype(np.int64),
            b_idx.astype(np.int64),
            np.asarray(cap_forward, dtype=np.float64),
            np.asarray(cap_reverse, dtype=np.float64),
        )

    horizontal = valid_mask[:, :-1] & valid_mask[:, 1:]
    rr, cc = np.where(horizontal)
    a_h = index_grid[rr, cc]
    b_h = index_grid[rr, cc + 1]
    add_edges(a_h, b_h, np.full(a_h.size, mu), np.full(a_h.size, mu))

    vertical = valid_mask[:-1, :] & valid_mask[1:, :]
    rr2, cc2 = np.where(vertical)
    a_v = index_grid[rr2, cc2]
    b_v = index_grid[rr2 + 1, cc2]
    add_edges(a_v, b_v, np.full(a_v.size, mu), np.full(a_v.size, mu))

    dist_a_h = distance_to_water[rr, cc]
    dist_b_h = distance_to_water[rr, cc + 1]
    add_edges(a_h[dist_b_h > dist_a_h], b_h[dist_b_h > dist_a_h], np.full((dist_b_h > dist_a_h).sum(), INF_CAPACITY), np.zeros((dist_b_h > dist_a_h).sum()))
    add_edges(b_h[dist_a_h > dist_b_h], a_h[dist_a_h > dist_b_h], np.full((dist_a_h > dist_b_h).sum(), INF_CAPACITY), np.zeros((dist_a_h > dist_b_h).sum()))

    dist_a_v = distance_to_water[rr2, cc2]
    dist_b_v = distance_to_water[rr2 + 1, cc2]
    add_edges(a_v[dist_b_v > dist_a_v], b_v[dist_b_v > dist_a_v], np.full((dist_b_v > dist_a_v).sum(), INF_CAPACITY), np.zeros((dist_b_v > dist_a_v).sum()))
    add_edges(b_v[dist_a_v > dist_b_v], a_v[dist_a_v > dist_b_v], np.full((dist_a_v > dist_b_v).sum(), INF_CAPACITY), np.zeros((dist_a_v > dist_b_v).sum()))

    log("running max-flow...")
    graph.maxflow()
    segments = np.fromiter((graph.get_segment(i) for i in range(n_pixels)), dtype=np.int8, count=n_pixels)
    harvest_mask = np.zeros((rows, cols), dtype=bool)
    harvest_mask[valid_mask] = segments == 0
    return harvest_mask


def protected_width_raster(harvest_mask: np.ndarray, distance_to_water: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(distance_to_water, dtype=np.float32)
    protected = valid_mask & (~harvest_mask)
    result[protected] = distance_to_water[protected]
    return result


def write_geotiff(path: Path, raster: np.ndarray, grid: Grid, dtype=rasterio.float32, nodata=None) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass

    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "lzw",
        "tiled": True,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(raster.astype(dtype), 1)


def plot_region(ax, pixel_extent, grid: Grid, chm, water_mask, harvest_mask, protected_mask, distance_to_water, d_max, title, water_dilate_px=1):
    r0, r1, c0, c1 = pixel_extent
    chm_sub = chm[r0:r1, c0:c1]
    water_sub = water_mask[r0:r1, c0:c1]
    harvest_sub = harvest_mask[r0:r1, c0:c1]
    protected_sub = protected_mask[r0:r1, c0:c1]
    dist_sub = distance_to_water[r0:r1, c0:c1]

    if water_dilate_px > 0:
        water_sub = binary_dilation(water_sub, iterations=water_dilate_px)

    x0, y0 = grid.transform * (c0, r1)
    x1, y1 = grid.transform * (c1, r0)
    geo_extent = (x0, x1, y0, y1)

    ax.imshow(np.clip(chm_sub, 0, 25), cmap="Greys", vmin=0, vmax=25, extent=geo_extent, origin="upper", alpha=0.55, interpolation="nearest")
    ax.imshow(np.where(protected_sub, dist_sub, np.nan), cmap="YlGn", vmin=0, vmax=d_max, extent=geo_extent, origin="upper", interpolation="nearest")

    harvest_rgba = np.zeros((*harvest_sub.shape, 4), dtype=np.float32)
    harvest_rgba[..., 0] = 0.85
    harvest_rgba[..., 1] = 0.20
    harvest_rgba[..., 2] = 0.15
    harvest_rgba[..., 3] = np.where(harvest_sub, 0.45, 0.0)
    ax.imshow(harvest_rgba, extent=geo_extent, origin="upper", interpolation="nearest")

    water_rgba = np.zeros((*water_sub.shape, 4), dtype=np.float32)
    water_rgba[..., 0] = 0.05
    water_rgba[..., 1] = 0.35
    water_rgba[..., 2] = 0.85
    water_rgba[..., 3] = np.where(water_sub, 1.0, 0.0)
    ax.imshow(water_rgba, extent=geo_extent, origin="upper", interpolation="nearest")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")


def build_geo_extent(pixel_extent, grid: Grid):
    r0, r1, c0, c1 = pixel_extent
    x0, y0 = grid.transform * (c0, r1)
    x1, y1 = grid.transform * (c1, r0)
    return (x0, x1, y0, y1)


def save_zoom_layers(
    zoom_dir: Path,
    pixel_extent,
    grid: Grid,
    chm: np.ndarray,
    water_mask: np.ndarray,
    harvest_mask: np.ndarray,
    protected_mask: np.ndarray,
    distance_to_water: np.ndarray,
    d_max: float,
    title: str,
) -> None:
    zoom_dir.mkdir(parents=True, exist_ok=True)
    r0, r1, c0, c1 = pixel_extent
    chm_sub = chm[r0:r1, c0:c1]
    water_sub = water_mask[r0:r1, c0:c1]
    harvest_sub = harvest_mask[r0:r1, c0:c1]
    protected_sub = protected_mask[r0:r1, c0:c1]
    dist_sub = distance_to_water[r0:r1, c0:c1]
    geo_extent = build_geo_extent(pixel_extent, grid)

    def save_layer(name: str, draw_fn) -> None:
        fig, ax = plt.subplots(figsize=(9, 8))
        draw_fn(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(zoom_dir / name, dpi=160)
        plt.close(fig)

    save_layer(
        "combined.png",
        lambda ax: plot_region(ax, pixel_extent, grid, chm, water_mask, harvest_mask, protected_mask, distance_to_water, d_max, title),
    )

    save_layer(
        "chm_background.png",
        lambda ax: ax.imshow(
            np.clip(chm_sub, 0, 25),
            cmap="Greys",
            vmin=0,
            vmax=25,
            extent=geo_extent,
            origin="upper",
            interpolation="nearest",
        ),
    )

    save_layer(
        "protected_buffer.png",
        lambda ax: ax.imshow(
            np.where(protected_sub, dist_sub, np.nan),
            cmap="YlGn",
            vmin=0,
            vmax=d_max,
            extent=geo_extent,
            origin="upper",
            interpolation="nearest",
        ),
    )

    def draw_harvest(ax):
        harvest_rgba = np.zeros((*harvest_sub.shape, 4), dtype=np.float32)
        harvest_rgba[..., 0] = 0.85
        harvest_rgba[..., 1] = 0.20
        harvest_rgba[..., 2] = 0.15
        harvest_rgba[..., 3] = np.where(harvest_sub, 0.65, 0.0)
        ax.imshow(harvest_rgba, extent=geo_extent, origin="upper", interpolation="nearest")

    save_layer("harvested_area.png", draw_harvest)

    def draw_water(ax):
        water_rgba = np.zeros((*water_sub.shape, 4), dtype=np.float32)
        water_rgba[..., 0] = 0.05
        water_rgba[..., 1] = 0.35
        water_rgba[..., 2] = 0.85
        water_rgba[..., 3] = np.where(binary_dilation(water_sub, iterations=1), 1.0, 0.0)
        ax.imshow(water_rgba, extent=geo_extent, origin="upper", interpolation="nearest")

    save_layer("watercourse.png", draw_water)


def make_region_tiles(water_geometry, n_regions: int, half_size_m: float = 400.0):
    if isinstance(water_geometry, MultiLineString):
        parts = list(water_geometry.geoms)
    elif isinstance(water_geometry, LineString):
        parts = [water_geometry]
    else:
        parts = [LineString(list(water_geometry.coords))]

    lengths = [part.length for part in parts]
    total_length = sum(lengths)

    def point_at_length(target_length: float):
        cumulative = 0.0
        for part, part_length in zip(parts, lengths):
            if cumulative + part_length >= target_length:
                return part.interpolate(target_length - cumulative)
            cumulative += part_length
        return parts[-1].interpolate(lengths[-1])

    tiles = []
    for index in range(n_regions):
        center_length = (index + 0.5) * total_length / n_regions
        center = point_at_length(center_length)
        tiles.append((f"zoom_{index + 1}", (center.x - half_size_m, center.y - half_size_m, center.x + half_size_m, center.y + half_size_m)))
    return tiles


def save_plots(out_dir: Path, grid: Grid, chm, water_mask, harvest_mask, distance_to_water, d_max, water_geometry, valid_mask, n_regions: int) -> None:
    protected_mask = valid_mask & (~harvest_mask)

    stride = max(1, min(grid.width, grid.height) // 1500)
    overview_dilate = max(1, min(grid.width, grid.height) // 700)
    harvest_overview = binary_dilation(harvest_mask, iterations=overview_dilate)
    protected_overview = binary_dilation(protected_mask, iterations=overview_dilate)

    overview_grid = Grid(
        transform=rasterio.Affine(grid.transform.a * stride, 0, grid.transform.c, 0, grid.transform.e * stride, grid.transform.f),
        width=grid.width // stride,
        height=grid.height // stride,
        crs=grid.crs,
        pixel_size=grid.pixel_size * stride,
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    plot_region(
        ax,
        (0, overview_grid.height, 0, overview_grid.width),
        overview_grid,
        chm[::stride, ::stride],
        water_mask[::stride, ::stride],
        harvest_overview[::stride, ::stride],
        protected_overview[::stride, ::stride],
        distance_to_water[::stride, ::stride],
        d_max,
        "Buffer-zone optimization overview",
        water_dilate_px=2,
    )
    legend = [
        Patch(facecolor="#2e7d32", label="Protected buffer (green = close to water)"),
        Patch(facecolor="#d95a30", label="Harvested"),
        Patch(facecolor="#0f60d5", label="Watercourse"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=160)
    plt.close(fig)

    for name, (xmin, ymin, xmax, ymax) in make_region_tiles(water_geometry, n_regions):
        c0, r0 = ~grid.transform * (xmin, ymax)
        c1, r1 = ~grid.transform * (xmax, ymin)
        c0, c1 = sorted([int(c0), int(c1)])
        r0, r1 = sorted([int(r0), int(r1)])
        c0 = max(0, c0)
        r0 = max(0, r0)
        c1 = min(grid.width, c1)
        r1 = min(grid.height, r1)
        if c1 - c0 < 5 or r1 - r0 < 5:
            continue

        fig, ax = plt.subplots(figsize=(9, 8))
        plot_region(ax, (r0, r1, c0, c1), grid, chm, water_mask, harvest_mask, protected_mask, distance_to_water, d_max, f"{name} (D_max={d_max} m)")
        ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=160)
        plt.close(fig)

        save_zoom_layers(
            out_dir / name,
            (r0, r1, c0, c1),
            grid,
            chm,
            water_mask,
            harvest_mask,
            protected_mask,
            distance_to_water,
            d_max,
            f"{name} (D_max={d_max} m)",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Riparian buffer optimization with graph cut.")
    parser.add_argument("--config", type=Path, help="Path to a JSON settings file.")
    parser.add_argument("--lam", type=float, help="Override lambda ecology weight from config.")
    parser.add_argument("--mu", type=float, help="Override mu boundary weight from config.")
    parser.add_argument("--resolution", type=float, help="Override resolution from config.")
    parser.add_argument("--d-max", type=float, help="Override maximum buffer width from config.")
    return parser.parse_args()


def write_summary(
    summary_path: Path,
    settings: Settings,
    scenario_name: str,
    scenario_lambda: float,
    harvested_area_ha: float,
    protected_area_ha: float,
    harvested_revenue: float,
    harvested_eco_loss: float,
    protected_eco_value: float,
) -> None:
    summary_path.write_text(
        "\n".join(
            [
                "Riparian buffer optimization",
                f"scenario_name          : {scenario_name}",
                f"resolution_m           : {settings.model.resolution_m}",
                f"max_buffer_width_m     : {settings.model.max_buffer_width_m}",
                f"lambda_ecology_weight  : {scenario_lambda}",
                f"mu_boundary_weight     : {settings.model.mu_boundary_weight}",
                "",
                f"harvested_area_ha      : {harvested_area_ha:.2f}",
                f"protected_area_ha      : {protected_area_ha:.2f}",
                f"harvested_revenue_msek : {harvested_revenue / 1e6:.3f}",
                f"harvested_eco_msek_eq  : {harvested_eco_loss / 1e6:.3f}",
                f"protected_eco_msek_eq  : {protected_eco_value / 1e6:.3f}",
            ]
        ),
        encoding="utf-8",
    )


def remove_legacy_root_outputs(output_dir: Path) -> None:
    legacy_names = [
        "buffer_width.tif",
        "harvest_mask.tif",
        "overview.png",
        "summary.txt",
        "zoom_1.png",
        "zoom_2.png",
        "zoom_3.png",
        "zoom_4.png",
    ]
    for name in legacy_names:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def run_scenario(
    settings: Settings,
    scenario_name: str,
    scenario_lambda: float,
    grid: Grid,
    chm: np.ndarray,
    dtw: np.ndarray,
    species: np.ndarray,
    water_geometry,
    water_mask: np.ndarray,
    distance_to_water: np.ndarray,
    corridor_mask: np.ndarray,
) -> None:
    scenario_dir = settings.paths.output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    log(f"computing ecological and economic values for scenario '{scenario_name}'...")
    revenue, ecological_loss = compute_pixel_values(chm, species, dtw, water_mask, distance_to_water, corridor_mask, settings)

    log(f"solving graph cut for scenario '{scenario_name}' (lambda={scenario_lambda}, mu={settings.model.mu_boundary_weight})...")
    harvest_mask = solve_graph_cut(
        revenue,
        ecological_loss,
        distance_to_water,
        corridor_mask,
        scenario_lambda,
        settings.model.mu_boundary_weight,
    )
    protected_mask = corridor_mask & (~harvest_mask)

    harvested_revenue = float(revenue[harvest_mask[corridor_mask]].sum())
    harvested_eco_loss = float(ecological_loss[harvest_mask[corridor_mask]].sum())
    protected_eco_value = float(ecological_loss[protected_mask[corridor_mask]].sum())
    harvested_area_ha = float(harvest_mask.sum()) * settings.model.resolution_m ** 2 / 1e4
    protected_area_ha = float(protected_mask.sum()) * settings.model.resolution_m ** 2 / 1e4

    log(f"[{scenario_name}] harvested area: {harvested_area_ha:.2f} ha")
    log(f"[{scenario_name}] protected area: {protected_area_ha:.2f} ha")
    log(f"[{scenario_name}] harvested revenue: {harvested_revenue / 1e6:.3f} MSEK")
    log(f"[{scenario_name}] harvested ecological loss: {harvested_eco_loss / 1e6:.3f} MSEK-eq")

    write_geotiff(scenario_dir / "buffer_width.tif", protected_width_raster(harvest_mask, distance_to_water, corridor_mask), grid)
    write_geotiff(
        scenario_dir / "harvest_mask.tif",
        np.where(corridor_mask, harvest_mask.astype(np.float32), np.nan).astype(np.float32),
        grid,
        nodata=np.nan,
    )
    save_plots(
        scenario_dir,
        grid,
        chm,
        water_mask,
        harvest_mask,
        distance_to_water,
        settings.model.max_buffer_width_m,
        water_geometry,
        corridor_mask,
        settings.model.zoom_regions,
    )
    write_summary(
        scenario_dir / "summary.txt",
        settings,
        scenario_name,
        scenario_lambda,
        harvested_area_ha,
        protected_area_ha,
        harvested_revenue,
        harvested_eco_loss,
        protected_eco_value,
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    if args.lam is not None:
        settings.model.lambda_ecology_weight = args.lam
    if args.mu is not None:
        settings.model.mu_boundary_weight = args.mu
    if args.resolution is not None:
        settings.model.resolution_m = args.resolution
    if args.d_max is not None:
        settings.model.max_buffer_width_m = args.d_max

    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_root_outputs(settings.paths.output_dir)

    log("building target grid...")
    grid = build_target_grid(settings.paths.chm_raster, settings.model.resolution_m)
    log(f"grid size: {grid.width} x {grid.height} pixels")

    log("reading and resampling rasters...")
    chm = reproject_to_grid(settings.paths.chm_raster, grid, Resampling.average)
    dtw = reproject_to_grid(settings.paths.dtw_raster, grid, Resampling.average)
    species = reproject_to_grid(settings.paths.species_raster, grid, Resampling.mode, dtype=np.int16)

    log("loading watercourse...")
    water_geometry = load_watercourse(settings.paths.watercourse_shapefile, grid.crs)
    water_mask = rasterize_lines(water_geometry, grid)

    log("computing distance to water...")
    band_px = int(math.ceil(settings.model.max_buffer_width_m / settings.model.resolution_m)) + 2
    rr, cc = np.where(water_mask)
    if rr.size == 0:
        raise RuntimeError("No water pixels were found. Check the shapefile and CRS alignment.")

    r_lo = max(0, int(rr.min()) - band_px)
    r_hi = min(grid.height, int(rr.max()) + band_px + 1)
    c_lo = max(0, int(cc.min()) - band_px)
    c_hi = min(grid.width, int(cc.max()) + band_px + 1)

    water_crop = water_mask[r_lo:r_hi, c_lo:c_hi].copy()
    dist_crop = np.full(water_crop.shape, np.float32(settings.model.max_buffer_width_m + 1.0), dtype=np.float32)
    dist_crop[water_crop] = 0.0
    struct8 = np.ones((3, 3), dtype=bool)
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    current = water_crop.copy()
    for step in range(1, band_px + 1):
        if step % 2 == 0:
            next_mask = binary_dilation(current, structure=struct4, iterations=1)
        else:
            next_mask = binary_dilation(current, structure=struct8, iterations=1)
        new_pixels = next_mask & (~current)
        if not new_pixels.any():
            break
        dist_crop[new_pixels] = np.minimum(dist_crop[new_pixels], np.float32(step * settings.model.resolution_m))
        current = next_mask

    distance_to_water = np.full(water_mask.shape, np.float32(settings.model.max_buffer_width_m + 1.0), dtype=np.float32)
    distance_to_water[r_lo:r_hi, c_lo:c_hi] = dist_crop

    valid_data = np.isfinite(chm) & np.isfinite(dtw) & (chm > -50) & (dtw >= 0)
    corridor_mask = (distance_to_water > 0) & (distance_to_water <= settings.model.max_buffer_width_m) & valid_data
    log(f"corridor pixels: {int(corridor_mask.sum()):,}")

    scenario_lambdas = settings.model.scenario_lambdas or {
        "default": settings.model.lambda_ecology_weight
    }
    if args.lam is not None:
        scenario_lambdas = {"custom_lambda": settings.model.lambda_ecology_weight}

    for scenario_name, scenario_lambda in scenario_lambdas.items():
        run_scenario(
            settings,
            scenario_name,
            float(scenario_lambda),
            grid,
            chm,
            dtw,
            species,
            water_geometry,
            water_mask,
            distance_to_water,
            corridor_mask,
        )

    log(f"finished. outputs written to: {settings.paths.output_dir}")


if __name__ == "__main__":
    main()
