#!/usr/bin/env python3
"""
MCP Tool: Map Composition Processor for NEPAssist Tanks.

This standalone CLI tool generates PDF maps from KMZ or GeoJSON inputs, pairing
tank points with images, using the ArcGIS Online printing service. It replicates
the functionality of the NEPAssist Tanks app with identical inputs and parameters.
Requires only the 'requests' library.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


PRINTING_SERVICE_URL = "https://utility.arcgisonline.com/arcgis/rest/services/Utilities/PrintingTools/GPServer/Export%20Web%20Map%20Task/execute"


@dataclass
class Style:
    """Default styling for map elements."""
    site_outline_px: float = 1.0
    buffer_outline_px: float = 0.25
    buffer_fill_rgba: Tuple[int, int, int, int] = (255, 215, 0, 48)  # Gold, very transparent
    site_outline_rgba: Tuple[int, int, int, int] = (255, 0, 0, 255)   # Red
    site_fill_rgba: Tuple[int, int, int, int] = (255, 0, 0, 0)        # Transparent
    padding: float = 0.07  # 7% padding for framing
    base_map_url: str = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"


def _read_kmz(kmz_path: str) -> Tuple[List[List[List[Tuple[float, float]]]], List[List[List[Tuple[float, float]]]], List[Tuple[float, float, str]]]:
    """Extract site polygons, buffer polygons, and points from KMZ file."""
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kml_name = next((n for n in z.namelist() if n.endswith('.kml')), 'doc.kml')
        with z.open(kml_name) as f:
            raw = f.read().decode('utf-8', 'ignore')
    cleaned = re.sub(r"</?ns\d+:[^>]*>", "", raw)
    root = ET.fromstring(cleaned)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    polys = []  # List of (name, rings)
    points: List[Tuple[float, float, str]] = []

    for pm in root.findall('.//kml:Placemark', ns):
        name_el = pm.find('kml:name', ns)
        name = (name_el.text or '').strip() if name_el is not None else ''
        poly_el = pm.find('.//kml:Polygon', ns)
        pt_el = pm.find('.//kml:Point', ns)
        if poly_el is not None:
            rings: List[List[Tuple[float, float]]] = []
            for path, _ in [('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', True),
                            ('.//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates', False)]:
                for coords_el in poly_el.findall(path, ns):
                    coords = []
                    for tok in (coords_el.text or '').strip().split():
                        parts = tok.split(',')
                        try:
                            coords.append((float(parts[0]), float(parts[1])))
                        except (ValueError, IndexError):
                            continue
                    if coords:
                        rings.append(coords)
            if rings:
                polys.append((name, rings))
        elif pt_el is not None:
            coords_el = pt_el.find('kml:coordinates', ns)
            if coords_el is not None and coords_el.text:
                try:
                    lon_str, lat_str, *_ = coords_el.text.strip().split(',')
                    points.append((float(lon_str), float(lat_str), name or 'Point'))
                except (ValueError, IndexError):
                    continue

    # Separate site vs. buffer polygons
    buffer_polys = [rings for n, rings in polys if 'buffer' in n.lower()]
    non_buffer = [rings for n, rings in polys if 'buffer' not in n.lower()]
    # Select the smallest non-buffer polygon as site
    def area_est(rings: List[List[Tuple[float, float]]]) -> float:
        if not rings or not rings[0]:
            return 0.0
        ex = rings[0]
        return abs(sum(ex[i][0] * ex[(i + 1) % len(ex)][1] - ex[(i + 1) % len(ex)][0] * ex[i][1]
                       for i in range(len(ex)))) / 2.0

    site_polys = []
    if non_buffer:
        site_polys = [min(non_buffer, key=area_est)]
    return site_polys, buffer_polys, points


def _read_geojson(geojson_path: str) -> Tuple[List[List[List[Tuple[float, float]]]], List[List[List[Tuple[float, float]]]], List[Tuple[float, float, str]]]:
    """Extract site polygons, buffer polygons, and points from GeoJSON file."""
    data = json.loads(Path(geojson_path).read_text('utf-8'))
    site_polys: List[List[List[Tuple[float, float]]]] = []
    buffer_polys: List[List[List[Tuple[float, float]]]] = []
    points: List[Tuple[float, float, str]] = []
    for f in data.get('features', []):
        geom = f.get('geometry', {})
        geom_type = (geom.get('type') or '').lower()
        props = f.get('properties', {})
        name = props.get('name') or props.get('label') or ''
        if geom_type == 'polygon':
            rings = [[(float(x), float(y)) for x, y in ring] for ring in (geom.get('coordinates') or [])]
            label = name.lower()
            cat = (props.get('category') or '').lower()
            if 'buffer' in label or cat == 'offsite':
                buffer_polys.append(rings)
            else:
                site_polys.append(rings)
        elif geom_type == 'multipolygon':
            for poly_coords in geom.get('coordinates') or []:
                rings = [[(float(x), float(y)) for x, y in ring] for ring in poly_coords]
                label = name.lower()
                cat = (props.get('category') or '').lower()
                if 'buffer' in label or cat == 'offsite':
                    buffer_polys.append(rings)
                else:
                    site_polys.append(rings)
        elif geom_type == 'point':
            coords = geom.get('coordinates', [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                points.append((float(coords[0]), float(coords[1]), name or 'Point'))

    # Select smallest site if multiple
    if len(site_polys) > 1:
        def area_est(rings):
            if not rings or not rings[0]:
                return 0.0
            ex = rings[0]
            return abs(sum(ex[i][0] * ex[(i + 1) % len(ex)][1] - ex[(i + 1) % len(ex)][0] * ex[i][1]
                           for i in range(len(ex)))) / 2.0
        site_polys = [min(site_polys, key=area_est)]

    return site_polys, buffer_polys, points


def calculate_optimal_marker_size(marker_count: int) -> int:
    """Determine optimal image marker size based on count for visibility and legend fit."""
    if marker_count <= 5:
        return 35
    elif marker_count <= 8:
        return 28
    elif marker_count <= 10:
        return 24
    elif marker_count <= 15:
        return 21
    elif marker_count <= 20:
        return 20
    elif marker_count <= 30:
        return 17
    else:
        return 14


def _sorted_images(images_dir: str) -> List[Path]:
    """Sort image files by numeric prefix in stem, then by name."""
    directory = Path(images_dir)
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.gif'}]

    def sort_key(p: Path) -> Tuple[int, str]:
        match = re.search(r'(\d+)', p.stem)
        num = int(match.group(1)) if match else 10**9
        return (num, p.name.lower())

    return sorted(files, key=sort_key)


def _encode_image(p: Path) -> str:
    """Encode image file to base64."""
    return base64.b64encode(p.read_bytes()).decode('ascii')


def _build_image_markers(points: List[Tuple[float, float, str]], images_dir: str, size: int) -> List[Dict[str, Any]]:
    """Pair points with sorted images and encode as base64 markers."""
    images = _sorted_images(images_dir)
    n = min(len(points), len(images))
    markers: List[Dict[str, Any]] = []
    for (lon, lat, name), img_path in zip(points[:n], images[:n]):
        markers.append({
            'lon': lon,
            'lat': lat,
            'png_b64': _encode_image(img_path),
            'size': size,
            'label': name
        })
    return markers


def _polygon_layer(layer_name: str, polys: List[List[List[Tuple[float, float]]]], fill_rgba: Tuple[int, int, int, int],
                   outline_rgba: Tuple[int, int, int, int], outline_width: float) -> Optional[Dict[str, Any]]:
    """Create polygon layer matching portable implementation for proper legend.

    - Uses layerDefinition.drawingInfo (simple renderer) to drive legend swatches
    - Provides a minimal fields schema with a name attribute
    - Does NOT set per-feature symbols
    """
    if not polys:
        return None

    features = []
    for rings in polys:
        features.append({
            "geometry": {
                "rings": [[[lon, lat] for lon, lat in ring] for ring in rings],
                "spatialReference": {"wkid": 4326}
            },
            "attributes": {"name": layer_name}
        })

    layer_id = f"layer_{layer_name.lower().replace(' ', '_')}"
    return {
        "id": layer_id,
        "title": layer_name,
        "featureCollection": {
            "layers": [{
                "layerDefinition": {
                    "name": layer_name,
                    "geometryType": "esriGeometryPolygon",
                    "fields": [
                        {"name": "name", "type": "esriFieldTypeString", "alias": "Name"}
                    ],
                    "drawingInfo": {
                        "renderer": {
                            "type": "simple",
                            "symbol": {
                                "type": "esriSFS",
                                "style": "esriSFSSolid",
                                "color": list(fill_rgba),
                                "outline": {
                                    "type": "esriSLS",
                                    "style": "esriSLSSolid",
                                    "color": list(outline_rgba),
                                    "width": outline_width
                                }
                            }
                        }
                    }
                },
                "featureSet": {"geometryType": "esriGeometryPolygon", "features": features}
            }]
        }
    }


def _points_layer(markers: List[Dict[str, Any]], legend_label_prefix: str = "Tank", label_max_len: int = 40) -> Optional[Dict[str, Any]]:
    """Create points layer for image markers.

    Use a unique-value renderer with embedded imageData for each marker, mirroring
    the portable implementation for reliable rendering by ArcGIS PrintingTools.
    """
    if not markers:
        return None

    unique_infos: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []
    for idx, m in enumerate(markers):
        marker_id = f"mk_{idx}"
        b64 = (m.get('png_b64') or '').split(',', 1)[-1]
        size = int(m.get('size') or 28)
        base_label = (m.get('label') or '').strip() or f"{legend_label_prefix} {idx+1}"
        # Compose legend label like "Tank 1 — Name" and trim overly long text
        composed_label = f"{legend_label_prefix} {idx+1} — {base_label}"
        if len(composed_label) > label_max_len:
            composed_label = composed_label[: label_max_len - 1] + "…"
        unique_infos.append({
            "value": marker_id,
            "label": composed_label,
            "symbol": {
                "type": "esriPMS",
                "imageData": b64,
                "contentType": "image/png",
                "width": size,
                "height": size
            }
        })
        features.append({
            "geometry": {"x": float(m['lon']), "y": float(m['lat']), "spatialReference": {"wkid": 4326}},
            "attributes": {"marker_id": marker_id, "name": base_label}
        })

    return {
        "id": "image_markers",
        "title": "Tanks",
        "layerType": "FeatureLayer",
        "featureCollection": {
            "layers": [{
                "layerDefinition": {
                    "name": "Tanks",
                    "geometryType": "esriGeometryPoint",
                    "fields": [
                        {"name": "marker_id", "type": "esriFieldTypeString", "alias": "Id"},
                        {"name": "name", "type": "esriFieldTypeString", "alias": "Name"}
                    ],
                    "drawingInfo": {
                        "renderer": {
                            "type": "uniqueValue",
                            "field1": "marker_id",
                            "uniqueValueInfos": unique_infos
                        }
                    }
                },
                "featureSet": {"geometryType": "esriGeometryPoint", "features": features}
            }]
        }
    }


def _extent_from_polys(polys: List[List[List[Tuple[float, float]]]], padding: float) -> Dict[str, Any]:
    """Compute map extent from polygons with padding."""
    all_points = []
    for rings in polys:
        for ring in rings:
            all_points.extend(ring)
    if not all_points:
        return {"xmin": -180, "ymin": -90, "xmax": 180, "ymax": 90, "spatialReference": {"wkid": 4326}}

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    pad_x = dx * padding
    pad_y = dy * padding
    return {
        "xmin": min(xs) - pad_x,
        "ymin": min(ys) - pad_y,
        "xmax": max(xs) + pad_x,
        "ymax": max(ys) + pad_y,
        "spatialReference": {"wkid": 4326}
    }


def export_pdf(web_map: Dict[str, Any], pdf_path: Path, layout_template: str = "Letter ANSI A Landscape") -> bool:
    """Export web map to PDF via ArcGIS PrintingTools (form-encoded params).

    Matches the portable implementation signature: sends `Web_Map_as_JSON`, `Format`,
    and `Layout_Template` as form data, not JSON body.
    """
    session = requests.Session()
    for attempt in range(3):
        try:
            print(f"📤 Submitting map export request (Attempt {attempt + 1}/3)...")
            export_params = {
                "f": "json",
                "Web_Map_as_JSON": json.dumps(web_map),
                "Format": "PDF",
                "Layout_Template": layout_template,
            }
            resp = session.post(PRINTING_SERVICE_URL, data=export_params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if 'results' in data and data['results']:
                url = data['results'][0]['value']['url']
                pdf_resp = session.get(url, timeout=60)
                pdf_resp.raise_for_status()
                pdf_path.write_bytes(pdf_resp.content)
                print(f"   📄 PDF saved: {pdf_path}")
                return True
        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    print("   ❌ PDF export failed after 3 attempts. Check connectivity to ArcGIS service.")
    return False


def run(cfg: Dict[str, Any], out_dir: Path) -> None:
    """Core function to generate map from config."""
    print("🚀 Starting MCP map generation...")

    # Determine source
    source_type = cfg.get('source_type')
    if not source_type:
        source_type = 'kmz' if cfg.get('input_kmz') else 'geojson' if cfg.get('input_geojson') else None
    if not source_type:
        raise ValueError("No valid input source specified (input_kmz or input_geojson required).")
    input_path = cfg.get('input_kmz') if source_type == 'kmz' else cfg.get('input_geojson')

    # Read geometries
    if source_type == 'kmz':
        site_polys, buffer_polys, points = _read_kmz(input_path)
    elif source_type == 'geojson':
        site_polys, buffer_polys, points = _read_geojson(input_path)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    print(f"   📍 Read: {len(site_polys)} site poly(s), {len(buffer_polys)} buffer(s), {len(points)} point(s)")

    # Map options (used in several sections below)
    map_opts = cfg.get('map_options', {}) if isinstance(cfg.get('map_options', {}), dict) else {}

    # Prepare image markers
    images_dir = cfg.get('images_dir', '')
    use_images = cfg.get('use_images', True)
    markers = []
    image_size = int(cfg.get('image_size', 0))
    if use_images and images_dir and Path(images_dir).exists():
        if image_size <= 0:  # Auto-size
            num_images = len(_sorted_images(images_dir))
            marker_count = min(num_images, len(points))
            image_size = calculate_optimal_marker_size(marker_count)
            print(f"   🎯 Auto-sizing: {marker_count} markers → {image_size}px")
        markers = _build_image_markers(points, images_dir, image_size)
        print(f"   🖼️ Added {len(markers)} image markers from {images_dir}")
    else:
        print(f"   ⚠️ Skipping images (dir: {images_dir}, use: {use_images})")

    # Build layers
    style = Style()
    layers: List[Dict[str, Any]] = []
    # Match portable layer titles/IDs for legend
    site_layer = _polygon_layer('Site', site_polys, style.site_fill_rgba, style.site_outline_rgba, style.site_outline_px)
    if site_layer:
        layers.append(site_layer)
    buffer_layer = _polygon_layer('5280ft Buffer', buffer_polys, style.buffer_fill_rgba, style.site_outline_rgba, style.buffer_outline_px)
    if buffer_layer:
        layers.append(buffer_layer)
    legend_label_prefix = (map_opts.get('legend_label_prefix') or 'Tank') if isinstance(map_opts, dict) else 'Tank'
    label_max_len = int(map_opts.get('legend_label_max_len', 40)) if isinstance(map_opts, dict) else 40
    points_layer = _points_layer(markers, legend_label_prefix=legend_label_prefix, label_max_len=label_max_len)
    if points_layer:
        layers.append(points_layer)

    # Compute extent
    extent_polys = buffer_polys if map_opts.get('zoom_to_features', True) else site_polys or buffer_polys
    # Support multiple padding locations: top-level 'zoom_padding', map_options['zoom_padding'], or legacy 'padding_percent'
    padding = cfg.get('zoom_padding')
    if padding is None:
        padding = map_opts.get('zoom_padding')
    if padding is None:
        padding = map_opts.get('padding_percent')
        try:
            if padding is not None and float(padding) > 1:
                padding = float(padding) / 100.0
        except (TypeError, ValueError):
            padding = None
    try:
        padding = float(padding) if padding is not None else style.padding
    except (TypeError, ValueError):
        padding = style.padding
    extent = _extent_from_polys(extent_polys or [[]], padding)

    # Base map
    map_style = cfg.get('map_style', 'satellite')
    base_map_url = style.base_map_url
    if isinstance(map_style, dict):
        base_map_url = map_style.get('base_map_url', base_map_url)
    elif map_style == 'satellite':
        base_map_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
    elif map_style == 'professional':
        base_map_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer"

    # Assemble web map JSON
    # Build legend inclusion list with optional hide flags
    legend_layers: List[Dict[str, Any]] = []
    hide_site = bool(map_opts.get('legend_hide_site', False))
    hide_buffer = bool(map_opts.get('legend_hide_buffer', False))
    hide_points = bool(map_opts.get('legend_hide_points', False))
    for lyr in layers:
        lyr_id = lyr.get('id')
        lyr_title = lyr.get('title', '')
        if lyr_title == 'Site' and hide_site:
            continue
        if lyr_title == '5280ft Buffer' and hide_buffer:
            continue
        if lyr_id == 'image_markers' and hide_points:
            continue
        if lyr_id:
            legend_layers.append({"id": lyr_id})

    # Determine map title
    title_text_val = (
        map_opts.get('title_text')
        or cfg.get('map_title')
        or cfg.get('title')
        or " "
    )

    web_map = {
        "mapOptions": {"extent": extent, "spatialReference": {"wkid": 4326}, "showAttribution": True},
        "operationalLayers": layers,
        "baseMap": {"baseMapLayers": [{"url": base_map_url, "opacity": 1, "visibility": True, "layerType": "ArcGISMapServiceLayer"}]},
        "exportOptions": {
            "outputSize": map_opts.get('output_size', [792, 612]),  # Default landscape letter
            "dpi": cfg.get('dpi', 300)
        },
        "layoutOptions": {
            "titleText": title_text_val,
            "authorText": "Data Source: https://nepassisttool.epa.gov/nepassist/nepamap.aspx",
            "legendOptions": {"operationalLayers": legend_layers},
            "scaleBarOptions": {
                "metricUnit": "esriFeet",
                "metricLabel": "ft",
                "nonMetricUnit": "esriMiles",
                "nonMetricLabel": "mi",
                "show": map_opts.get('show_scale', True)
            },
            "copyrightText": "Spatial Reference: NAD83 StatePlane Puerto Rico Virgin Islands FIPS 5200 Feet"
        }
    }

    # Legend position (if supported by template)
    legend_pos = map_opts.get('legend_position')
    if legend_pos and isinstance(web_map.get("layoutOptions", {}).get("legendOptions"), dict):
        web_map["layoutOptions"]["legendOptions"]["position"] = legend_pos

    # North arrow
    if map_opts.get('show_north_arrow', True):
        web_map["layoutOptions"]["elementOverrides"] = {
            "North Arrow": {"name": "North Arrow", "type": "CIMMarkerNorthArrow", "visible": True}
        }

    # Hide date
    custom_text = []
    if map_opts.get('hide_date', True):
        custom_text.append({"Date": " "})
    if custom_text:
        web_map["layoutOptions"]["customTextElements"] = custom_text

    # Layout template
    layout_template = map_opts.get('layout_template', 'Letter ANSI A Landscape')

    # Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.get('output_filename_stem', 'NEPAssist_Tanks')
    pdf_path = out_dir / f"{stem}.pdf"
    geojson_path = out_dir / f"{stem}.geojson"

    # Generate PDF
    pdf_success = False
    if cfg.get('export_pdf', True):
        pdf_success = export_pdf(web_map, pdf_path, layout_template)
        if pdf_success:
            print(f"   📄 PDF saved: {pdf_path}")

    # Generate GeoJSON (independent of PDF success)
    # Previously this only wrote when PDF succeeded (or PDF was skipped),
    # which hid useful outputs during network outages. Always write when requested.
    if cfg.get('export_geojson', True):
        features = []
        for rings in site_polys:
            features.append({
                "type": "Feature",
                "properties": {"name": "Site", "category": "site"},
                "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring] for ring in rings]}
            })
        for rings in buffer_polys:
            features.append({
                "type": "Feature",
                "properties": {"name": "Buffer", "category": "offsite"},
                "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring] for ring in rings]}
            })
        for lon, lat, name in points:
            features.append({
                "type": "Feature",
                "properties": {"name": name, "feature_type": "point"},
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })
        geojson_data = {"type": "FeatureCollection", "features": features}
        geojson_path.write_text(json.dumps(geojson_data, indent=2))
        print(f"   🔄 GeoJSON saved: {geojson_path}")

    print(f"✅ Generation complete. Outputs in: {out_dir}")


@dataclass
class RuntimeOptions:
    """Runtime configuration from CLI args."""
    kmz: Optional[Path] = None
    geojson: Optional[Path] = None
    images_dir: Path = Path("images")
    output_dir: Path = Path("output")
    out_stem: str = "NEPAssist_Tanks"
    image_size: int = 0  # 0 = auto
    map_style: str = "satellite"
    layout_template: Optional[str] = None
    legend_position: Optional[str] = None
    show_scale: bool = True
    show_north_arrow: bool = True
    hide_date: bool = True
    zoom_padding: float = 0.07
    dpi: int = 300
    export_pdf: bool = True
    export_geojson: bool = True

    def to_config(self) -> Dict[str, Any]:
        """Convert options to run() config dict."""
        source_type = "kmz" if self.kmz else "geojson"
        cfg: Dict[str, Any] = {
            "job_name": "NEPAssist Tanks MCP",
            "source_type": source_type,
            "map_style": self.map_style,
            "export_pdf": self.export_pdf,
            "export_geojson": self.export_geojson,
            "output_filename_stem": self.out_stem,
            "images_dir": str(self.images_dir),
            "image_size": self.image_size,
            "use_images": True,
            "zoom_padding": self.zoom_padding,
            "dpi": self.dpi if self.dpi != 300 else None,  # Omit default
            "map_options": {
                "hide_date": self.hide_date,
                "show_north_arrow": self.show_north_arrow,
                "show_scale": self.show_scale,
                "layout_template": self.layout_template,
                "legend_position": self.legend_position,
                "zoom_to_features": True,
            }
        }
        if self.kmz:
            cfg["input_kmz"] = str(self.kmz)
        if self.geojson:
            cfg["input_geojson"] = str(self.geojson)
        return cfg


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with defaults matching the original app."""
    parser = argparse.ArgumentParser(
        description="MCP Tool: Generate NEPAssist Tanks PDF maps from KMZ/GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kmz", type=Path, help="Path to KMZ file (polygons and points)")
    source.add_argument("--geojson", type=Path, help="Path to GeoJSON file (polygons and points)")

    default_images = Path(__file__).parent / "images"
    default_output = Path(__file__).parent / "output"

    parser.add_argument("--images-dir", type=Path, default=default_images, help="Directory of marker images")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output directory")
    parser.add_argument("--out-stem", default="NEPAssist_Tanks", help="Output filename stem")
    parser.add_argument("--image-size", type=int, default=0, help="Marker size px (0=auto)")
    parser.add_argument("--map-style", default="satellite", choices=["satellite", "professional"], help="Base map style")
    parser.add_argument("--layout-template", default="Letter ANSI A Landscape", help="ArcGIS layout template")
    parser.add_argument("--legend-position", choices=["left", "right", "top", "bottom"], help="Legend position")
    parser.add_argument("--zoom-padding", type=float, default=0.07, help="Extent padding fraction (0-0.5)")
    parser.add_argument("--dpi", type=int, default=300, help="Export DPI")

    parser.add_argument("--no-scale", dest="show_scale", action="store_false", help="Hide scale bar")
    parser.add_argument("--no-north-arrow", dest="show_north_arrow", action="store_false", help="Hide north arrow")
    parser.add_argument("--no-hide-date", dest="hide_date", action="store_false", help="Show print date")
    parser.add_argument("--no-pdf", dest="export_pdf", action="store_false", help="Skip PDF")
    parser.add_argument("--no-geojson", dest="export_geojson", action="store_false", help="Skip GeoJSON")

    parser.set_defaults(
        show_scale=True, show_north_arrow=True, hide_date=True,
        export_pdf=True, export_geojson=True
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> RuntimeOptions:
    """Parse CLI args and validate paths."""
    parser = build_parser()
    args = parser.parse_args(argv)

    kmz = args.kmz.resolve() if args.kmz else None
    geojson = args.geojson.resolve() if args.geojson else None
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    if kmz and not kmz.exists():
        parser.error(f"KMZ not found: {kmz}")
    if geojson and not geojson.exists():
        parser.error(f"GeoJSON not found: {geojson}")
    if images_dir and not images_dir.exists():
        parser.error(f"Images dir not found: {images_dir}")

    return RuntimeOptions(
        kmz=kmz, geojson=geojson, images_dir=images_dir, output_dir=output_dir,
        out_stem=args.out_stem, image_size=args.image_size, map_style=args.map_style,
        layout_template=args.layout_template, legend_position=args.legend_position,
        show_scale=args.show_scale, show_north_arrow=args.show_north_arrow,
        hide_date=args.hide_date, zoom_padding=args.zoom_padding, dpi=args.dpi,
        export_pdf=args.export_pdf, export_geojson=args.export_geojson
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    options = parse_args(argv)
    cfg = options.to_config()
    try:
        run(cfg, options.output_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
