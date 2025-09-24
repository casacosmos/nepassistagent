#!/usr/bin/env python3
"""
LangGraph Agent for NEPAssist Tanks Map Generation Chatbot.

This script sets up a LangGraph-based chatbot agent that:
- Engages in conversation with the user.
- Accepts a KMZ or GeoJSON file path via chat input.
- Uses the MCP Tool (from mcp_tool.py) to generate PDF maps when requested.
- Responds with generation status and output paths.

Requirements:
- langchain, langgraph, langchain_openai (or your preferred LLM provider).
- Set OPENAI_API_KEY environment variable for OpenAI LLM.

Run with: python this_script.py
Then interact via the console (input messages, receive responses).
"""

import os
import asyncio
import subprocess
import atexit
import shlex
import json
import re
import zipfile
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
from typing import Annotated, Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.config import get_stream_writer

# Optional MCP adapters (loaded at runtime if installed)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
except Exception:
    MultiServerMCPClient = None  # type: ignore

# --- Excel MCP hardcoded defaults ---
EXCEL_MCP_ENABLED: bool = True  # hard-enable unless you flip to False
EXCEL_MCP_PORT_DEFAULT: int = 8123
EXCEL_MCP_URL_DEFAULT: str = f"http://127.0.0.1:{EXCEL_MCP_PORT_DEFAULT}/mcp"
EXCEL_MCP_FILES_DIR_DEFAULT: Path = Path("output/excel_files")

# Import the MCP tool's run function (assume mcp_tool.py is in the same directory)
from mcp_tool import run as mcp_run  # Replace with actual import if needed

# Excel template headers (in required order) - moved here to be available for tools
EXCEL_TEMPLATE_HEADERS: List[str] = [
    "Site Name or Business Name",
    "Person Contacted",
    "Tank Capacity",
    "Tank Measurements",
    "Has Dike",
    "Dike Measurements",
    "Diesel",
    "Pressurized Gas",
    "Acceptable Separation Distance Calculated",
    "Approximate Distance to Site (approximately)",
    "Compliance",
    "Additional information",
    "Latitude (NAD83)",
    "Longitude (NAD83)",
    "Calculated Distance to Polygon (ft)",
    "Tank Type",
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Use your preferred model

# System prompt to guide tool use and CRS detection
SYSTEM_PROMPT = (
    "You are the NEPAssist Tanks assistant. Detect coordinate formats and datums, "
    "then route through the right tools.\n"
    "- Recognize WGS84 decimal degrees (lat ∈ [-90,90], lon ∈ [-180,180]), DMS (e.g., 18°12'54\" N), "
    "UTM (e.g., Zone 19N), and NAD83 / Puerto Rico State Plane feet.\n"
    "- Before computing distances or writing spreadsheets, state the assumed CRS and units.\n"
    "- For map generation, use WGS84 lon/lat (the map engine will handle basemaps).\n"
    "- For Excel columns 'Latitude (NAD83)' and 'Longitude (NAD83)', convert from WGS84 to NAD83 (EPSG:4269) "
    "using the convert_coordinates tool when needed.\n"
    "- For distance calculations, ensure inputs are WGS84 lon/lat; if user provides State Plane, UTM, or DMS, "
    "first call convert_coordinates to WGS84, then call calculate_distance_to_polygon (it projects internally to UTM 19N).\n"
    "- If a requested output path does not exist, prefer the managed workspace 'output/excel_files' and inform the user.\n"
    "- Prefer concise answers; show exact output paths and summarize key results.\n\n"
    "IMPORTANT - EXCEL/CSV DATA STANDARDIZATION:\n"
    "- The EXCEL_TEMPLATE_HEADERS format is the OPTIMAL standard for all tank data storage and tool input.\n"
    "- When users upload CSV or Excel files, ALWAYS use the standardize_excel_data tool to map and reorganize columns.\n"
    "- The standard template ensures proper column order: Site Info → Tank Details → Has Dike → Dike Measurements → "
    "Diesel → Pressurized Gas → Distances → Compliance → Location → Type.\n"
    "- This standardization is REQUIRED before using data with other tools (map generation, calculations, etc.).\n"
    "- Inform users that their data will be reorganized to match the standard template for optimal processing.\n\n"
    "FUEL TYPE CLASSIFICATION:\n"
    "- Only two tank types exist: 'diesel' and 'pressurized_gas'\n"
    "- Pressurized gas includes: propane, LPG, compressed natural gas, and similar pressurized gases\n"
    "- Diesel category includes: diesel fuel, gasoline, kerosene, heating oil, jet fuel, and most liquid fuels\n"
    "- When ambiguous, interpret context: 'gas station' likely means gasoline (→diesel), 'gas tank' might mean pressurized gas\n"
    "- Set the Diesel and Pressurized Gas boolean columns appropriately during standardization"
)
# Define the MCP Tool
@tool
def generate_map(
    input_file: str,
    images_dir: str = "images",
    output_dir: str = "output",
    out_stem: str = "NEPAssist_Tanks",
    source_type: str = "kmz",  # or "geojson"
    image_size: int = 0,  # 0 for auto
    map_style: str = "satellite",
    export_pdf: bool = True,
    export_geojson: bool = True,
    # Layout and rendering options (defaults apply unless overridden by chat)
    dpi: int = 300,
    layout_template: str = "Letter ANSI A Landscape",
    map_title: str = "",
    zoom_padding: float = 0.07,
    zoom_to_features: bool = True,
    hide_date: bool = True,
    show_north_arrow: bool = True,
    show_scale: bool = True,
    legend_position: str = "right",
    legend_hide_site: bool = False,
    legend_hide_buffer: bool = False,
    legend_hide_points: bool = False,
    legend_label_prefix: str = "Tank",
    legend_label_max_len: int = 40,
) -> str:
    """
    Generate a PDF map from a KMZ or GeoJSON file using the MCP Tool.

    Args:
        input_file: Path to the KMZ or GeoJSON file.
        images_dir: Directory containing marker images (default: "images").
        output_dir: Output directory for PDF and GeoJSON (default: "output").
        out_stem: Filename stem for outputs (default: "NEPAssist_Tanks").
        source_type: "kmz" or "geojson" (auto-detected if possible).
        image_size: Marker size in pixels (0 for auto-sizing).
        map_style: Base map style ("satellite" or "professional").
        export_pdf: Whether to export PDF (default: True).
        export_geojson: Whether to export GeoJSON (default: True).

    Returns:
        Status message with output paths.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        return f"Error: Input file '{input_file}' not found."

    # Prepare config dict matching MCP Tool's expectations
    cfg = {
        "source_type": source_type,
        "input_kmz": str(input_path) if source_type == "kmz" else None,
        "input_geojson": str(input_path) if source_type == "geojson" else None,
        "images_dir": images_dir,
        "output_filename_stem": out_stem,
        "image_size": image_size,
        "use_images": True,
        "map_style": map_style,
        "export_pdf": export_pdf,
        "export_geojson": export_geojson,
        # Top-level options recognized by mcp_tool
        "dpi": dpi,
        "zoom_padding": zoom_padding,
        # Nested map_options also recognized by mcp_tool
        "map_options": {
            "hide_date": hide_date,
            "show_north_arrow": show_north_arrow,
            "show_scale": show_scale,
            "legend_position": legend_position,
            "legend_hide_site": legend_hide_site,
            "legend_hide_buffer": legend_hide_buffer,
            "legend_hide_points": legend_hide_points,
            "legend_label_prefix": legend_label_prefix,
            "legend_label_max_len": legend_label_max_len,
            "zoom_to_features": zoom_to_features,
            "zoom_padding": zoom_padding,
            "layout_template": layout_template,
            "title_text": (map_title if str(map_title).strip() else " "),
        },
    }

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir_path / f"{out_stem}.pdf"
    geojson_path = out_dir_path / f"{out_stem}.geojson"

    writer = None
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    try:
        if writer:
            writer({"type": "progress", "stage": "start", "message": f"Generating map for {input_path.name}"})
        mcp_run(cfg, out_dir_path)
        if writer:
            writer({"type": "progress", "stage": "done_run", "message": "Map engine finished"})
    except Exception as e:
        if writer:
            writer({"type": "error", "message": str(e)})
        return f"Error during map generation: {str(e)}"

    # Check if requested files were actually created
    status = "Map generation completed with the following results:\n"
    pdf_generated = export_pdf and pdf_path.exists()
    geojson_generated = export_geojson and geojson_path.exists()

    if pdf_generated:
        status += f"- PDF: {pdf_path.absolute()}\n"
    elif export_pdf:
        status += "- PDF: Failed to generate (check connectivity to ArcGIS service or logs for details)\n"

    if geojson_generated:
        status += f"- GeoJSON: {geojson_path.absolute()}\n"
    elif export_geojson:
        status += "- GeoJSON: Failed to generate\n"

    if (export_pdf and not pdf_generated) or (export_geojson and not geojson_generated):
        status = "Partial failure in generation.\n" + status
    else:
        status = "Map generation successful!\n" + status

    return status

# Bind the tools to the LLM (add KMZ tool by default)
# (Defined after kmz_tool to avoid NameError on first import)

# Define the state (using MessagesState for chat history)
class AgentState(MessagesState):
    pass  # Extends MessagesState for simplicity

# Agent node: Decide action based on LLM output
async def agent(state: AgentState) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    # Prepend system guidance for CRS detection and tool routing
    guided_messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = await llm_with_tools.ainvoke(guided_messages)
    return {"messages": [response]}


# -------------------------------- KMZ Parser Tool -------------------------------- #
@tool
def kmz_tool(
    operation: str,  # 'inspect', 'create', 'modify'
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    data: Optional[str] = None,  # JSON string for 'create'
    modifications: Optional[str] = None,  # JSON string for 'modify'
) -> str:
    """
    KMZ Parser Tool: Inspect, modify, or create KMZ files.

    - inspect: returns features as JSON
    - create: builds a KMZ from JSON features -> output_file
    - modify: edits existing KMZ with JSON modifications -> output_file
    """
    try:
        if operation not in {"inspect", "create", "modify"}:
            return json.dumps({"error": "Invalid operation. Use 'inspect', 'create', or 'modify'."})

        in_path = Path(input_file) if input_file else None
        out_path = Path(output_file) if output_file else None

        if operation in {"inspect", "modify"} and not in_path:
            return json.dumps({"error": "input_file required for inspect/modify."})
        if operation in {"create", "modify"} and not out_path:
            return json.dumps({"error": "output_file required for create/modify."})

        def _parse_coords(coord_str: str):
            coords = []
            for tok in coord_str.split():
                parts = tok.split(',')
                if len(parts) >= 2:
                    try:
                        coords.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            return coords

        def _extract(kmz_path: Path):
            with zipfile.ZipFile(kmz_path, 'r') as z:
                kml_name = next((n for n in z.namelist() if n.endswith('.kml')), 'doc.kml')
                with z.open(kml_name) as f:
                    raw = f.read().decode('utf-8', 'ignore')
            cleaned = re.sub(r"</?ns\d+:[^>]*>", "", raw)
            root = ET.fromstring(cleaned)
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            feats: List[Dict[str, Any]] = []
            for pm in root.findall('.//kml:Placemark', ns):
                name_el = pm.find('kml:name', ns)
                name = name_el.text.strip() if name_el is not None and name_el.text else 'Unnamed'
                pt_el = pm.find('kml:Point', ns)
                if pt_el is not None:
                    coords_el = pt_el.find('kml:coordinates', ns)
                    if coords_el is not None and coords_el.text:
                        try:
                            lon_str, lat_str, *_ = coords_el.text.strip().split(',')
                            feats.append({"type": "Point", "name": name, "coordinates": (float(lon_str), float(lat_str))})
                        except ValueError:
                            pass
                poly_el = pm.find('kml:Polygon', ns)
                if poly_el is not None:
                    rings = []
                    outer = poly_el.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
                    if outer is not None and outer.text:
                        ring = _parse_coords(outer.text.strip())
                        if ring:
                            rings.append(ring)
                    for inner in poly_el.findall('.//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates', ns):
                        if inner.text:
                            ring = _parse_coords(inner.text.strip())
                            if ring:
                                rings.append(ring)
                    if rings:
                        feats.append({"type": "Polygon", "name": name, "coordinates": rings})
            return feats

        def _apply_mods(feats: List[Dict[str, Any]], mods: Dict[str, Dict[str, Any]]):
            for feat in feats:
                if feat.get('name') in mods:
                    mod = mods[feat['name']]
                    if 'new_name' in mod:
                        feat['name'] = mod['new_name']
                    if 'new_coordinates' in mod:
                        feat['coordinates'] = mod['new_coordinates']

        def _build(feats: List[Dict[str, Any]], kmz_path: Path):
            kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
            doc = ET.SubElement(kml, 'Document')
            ET.SubElement(doc, 'name').text = 'Generated Document'
            ET.SubElement(doc, 'open').text = '1'
            for feat in feats:
                pm = ET.SubElement(doc, 'Placemark')
                ET.SubElement(pm, 'name').text = feat['name']
                if feat['type'] == 'Point':
                    point = ET.SubElement(pm, 'Point')
                    lon, lat = feat['coordinates']
                    ET.SubElement(point, 'coordinates').text = f"{lon},{lat},0"
                elif feat['type'] == 'Polygon':
                    polygon = ET.SubElement(pm, 'Polygon')
                    outer = ET.SubElement(polygon, 'outerBoundaryIs')
                    lr_outer = ET.SubElement(outer, 'LinearRing')
                    coords_str = ' '.join(f"{lon},{lat},0" for lon, lat in feat['coordinates'][0])
                    ET.SubElement(lr_outer, 'coordinates').text = coords_str
                    for inner_ring in feat['coordinates'][1:]:
                        inner = ET.SubElement(polygon, 'innerBoundaryIs')
                        lr_inner = ET.SubElement(inner, 'LinearRing')
                        coords_str = ' '.join(f"{lon},{lat},0" for lon, lat in inner_ring)
                        ET.SubElement(lr_inner, 'coordinates').text = coords_str
            kmz_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('doc.kml', ET.tostring(kml, encoding='utf-8', method='xml'))

        # Execute operation
        if operation == 'inspect':
            features = _extract(in_path)
            return json.dumps({"status": "success", "features": features})
        if operation == 'create':
            if not data:
                return json.dumps({"error": "data required for create."})
            feats = json.loads(data)
            _build(feats, out_path)
            return json.dumps({"status": "success", "output": str(out_path)})
        if operation == 'modify':
            if not modifications:
                return json.dumps({"error": "modifications required for modify."})
            mods = json.loads(modifications)
            feats = _extract(in_path)
            _apply_mods(feats, mods)
            _build(feats, out_path)
            return json.dumps({"status": "success", "output": str(out_path), "modified_features": feats})
        return json.dumps({"error": "Unhandled operation"})
    except Exception as e:
        return json.dumps({"error": f"Operation failed: {str(e)}"})

# Bind the tools to the LLM (add KMZ tool by default)
@tool
def create_excel_template(
    output_file: Optional[str] = None,
    sheet_name: str = "Data",
    kmz_file: Optional[str] = None,
) -> str:
    """
    Create an Excel workbook with the required header columns and optional rows from a KMZ.

    Includes columns for tank data, coordinates, compliance info, and fuel type indicators
    (Diesel and Pressurized Gas as boolean columns).

    Args:
        output_file: Where to save (defaults to output/excel_files/Tanks_Template.xlsx).
        sheet_name: Worksheet name (default 'Data').
        kmz_file: Optional KMZ path to prefill rows from Point/Polygon placemarks.

    Returns:
        Status JSON: {"status":"success","output":"/abs/path.xlsx","rows":N}
    """
    try:
        # Resolve output path (default to managed Excel workspace)
        if not output_file:
            output_path = EXCEL_MCP_FILES_DIR_DEFAULT / "Tanks_Template.xlsx"
        else:
            output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract simple rows from KMZ if provided
        rows: List[List[Any]] = []
        def _extract_feats(kmz_path: Path) -> List[Dict[str, Any]]:
            with zipfile.ZipFile(kmz_path, 'r') as z:
                kml_name = next((n for n in z.namelist() if n.endswith('.kml')), 'doc.kml')
                with z.open(kml_name) as f:
                    raw = f.read().decode('utf-8', 'ignore')
            cleaned = re.sub(r"</?ns\d+:[^>]*>", "", raw)
            root = ET.fromstring(cleaned)
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            feats: List[Dict[str, Any]] = []
            for pm in root.findall('.//kml:Placemark', ns):
                name_el = pm.find('kml:name', ns)
                name = name_el.text.strip() if name_el is not None and name_el.text else 'Unnamed'
                pt_el = pm.find('kml:Point', ns)
                if pt_el is not None:
                    coords_el = pt_el.find('kml:coordinates', ns)
                    if coords_el is not None and coords_el.text:
                        try:
                            lon_str, lat_str, *_ = coords_el.text.strip().split(',')
                            feats.append({"type": "Point", "name": name, "lon": float(lon_str), "lat": float(lat_str)})
                        except ValueError:
                            pass
                poly_el = pm.find('kml:Polygon', ns)
                if poly_el is not None:
                    outer = poly_el.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
                    lat = lon = None
                    if outer is not None and outer.text:
                        xs: List[float] = []
                        ys: List[float] = []
                        for tok in outer.text.strip().split():
                            parts = tok.split(',')
                            if len(parts) >= 2:
                                try:
                                    xs.append(float(parts[0]))
                                    ys.append(float(parts[1]))
                                except ValueError:
                                    continue
                        if xs and ys:
                            lon = sum(xs) / len(xs)
                            lat = sum(ys) / len(ys)
                    feats.append({"type": "Polygon", "name": name, "lon": lon, "lat": lat})
            return feats

        if kmz_file:
            kpath = Path(kmz_file)
            if not kpath.exists():
                return json.dumps({"error": f"KMZ not found: {kpath}"})
            for f in _extract_feats(kpath):
                # Build a row matching headers
                row = [None] * len(EXCEL_TEMPLATE_HEADERS)
                # Map fields
                def setcol(col_name: str, value: Any):
                    try:
                        idx = EXCEL_TEMPLATE_HEADERS.index(col_name)
                        row[idx] = value
                    except ValueError:
                        pass
                setcol("Site Name or Business Name", f.get("name"))
                if f.get("lat") is not None:
                    setcol("Latitude (NAD83)", f.get("lat"))
                if f.get("lon") is not None:
                    setcol("Longitude (NAD83)", f.get("lon"))
                rows.append(row)

        # Try to write XLSX via openpyxl; fallback to CSV
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment
            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name[:31] or "Data"
            ws.append(EXCEL_TEMPLATE_HEADERS)
            # style header
            bold = Font(bold=True)
            for cell in ws[1]:
                cell.font = bold
                cell.alignment = Alignment(vertical="center")
            # data rows
            for r in rows:
                ws.append(r)
            # basic column width
            for i, header in enumerate(EXCEL_TEMPLATE_HEADERS, start=1):
                ws.column_dimensions[chr(64+i)].width = min(max(len(header)+2, 14), 40)
            ws.freeze_panes = "A2"
            wb.save(output_path)
            return json.dumps({"status": "success", "output": str(output_path.resolve()), "rows": len(rows)})
        except ImportError:
            # CSV fallback
            csv_path = output_path.with_suffix('.csv')
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(EXCEL_TEMPLATE_HEADERS)
                writer.writerows(rows)
            return json.dumps({"status": "success", "output": str(csv_path.resolve()), "rows": len(rows), "note": "openpyxl not installed; wrote CSV instead"})
    except Exception as e:
        return json.dumps({"error": f"Failed to create Excel template: {str(e)}"})

@tool
async def convert_coordinates(
    input_lat: float,
    input_lon: float,
    input_datum: str = "WGS84",
    output_datum: str = "NAD83_PR_STATE_PLANE",
    dms_degrees: Optional[int] = None,
    dms_minutes: Optional[int] = None,
    dms_seconds: Optional[float] = None,
    dms_direction: Optional[str] = None,
) -> dict:
    """
    Convert coordinates between datums/CRS, with optional DMS override.

    - If DMS fields are provided, they override either latitude (N/S) or
      longitude (E/W) based on `dms_direction`.
    - input_datum: 'WGS84' (EPSG:4326) or 'NAD83' (EPSG:4269)
    - output_datum: 'NAD83_PR_STATE_PLANE' (EPSG:32161) or 'WGS84'
    """
    try:
        # Optional DMS override
        if (
            dms_degrees is not None
            and dms_minutes is not None
            and dms_seconds is not None
            and dms_direction
        ):
            dec = float(dms_degrees) + float(dms_minutes) / 60.0 + float(dms_seconds) / 3600.0
            dir_up = dms_direction.upper()
            if dir_up in {"S", "W"}:
                dec = -dec
            if dir_up in {"N", "S"}:
                input_lat = dec
            elif dir_up in {"E", "W"}:
                input_lon = dec

        # Map CRS codes
        inp = input_datum.upper()
        outp = output_datum.upper()
        if inp == "WGS84":
            input_crs = "EPSG:4326"
        elif inp == "NAD83":
            input_crs = "EPSG:4269"
        else:
            return {"error": f"Unsupported input datum: {input_datum}"}

        if outp == "NAD83_PR_STATE_PLANE":
            output_crs = "EPSG:32161"
        elif outp == "WGS84":
            output_crs = "EPSG:4326"
        else:
            return {"error": f"Unsupported output datum: {output_datum}"}

        try:
            import pyproj  # type: ignore
        except Exception:
            return {"error": "pyproj not installed. Install with: pip install pyproj"}

        transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
        output_lon, output_lat = transformer.transform(float(input_lon), float(input_lat))

        return {
            "input_datum": input_datum,
            "output_datum": output_datum,
            "input_lat": float(input_lat),
            "input_lon": float(input_lon),
            "output_lat": round(float(output_lat), 6),
            "output_lon": round(float(output_lon), 6),
            "output_crs": output_crs,
        }
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}

# -------------------------------- Distance to Polygon Tool -------------------------------- #
@tool
async def calculate_distance_to_polygon(
    point_lat: float,
    point_lon: float,
    polygon_coords: List[List[float]],
) -> dict:
    """
    Calculate minimum distance from a point to a polygon using a projected CRS.

    - Inputs are assumed WGS84 lon/lat pairs for both the point and polygon.
    - Uses UTM Zone 19N (EPSG:32619) suitable for Puerto Rico for accurate meters/feet.
    """
    try:
        try:
            import pyproj  # type: ignore
            from shapely.geometry import Point, Polygon  # type: ignore
            from shapely.ops import nearest_points  # type: ignore
        except Exception:
            return {"error": "Dependencies missing. Install with: pip install pyproj shapely"}

        # Build WGS84 geometries
        wgs84 = pyproj.CRS("EPSG:4326")
        utm19n = pyproj.CRS("EPSG:32619")
        to_utm = pyproj.Transformer.from_crs(wgs84, utm19n, always_xy=True)
        to_wgs = pyproj.Transformer.from_crs(utm19n, wgs84, always_xy=True)

        # Transform point
        px, py = to_utm.transform(float(point_lon), float(point_lat))
        pt_utm = Point(px, py)

        # Transform polygon
        poly_utm_coords = [tuple(to_utm.transform(float(lon), float(lat))) for lon, lat in polygon_coords]
        poly_utm = Polygon(poly_utm_coords)

        # Distance in meters (0 if inside or on boundary)
        distance_m = pt_utm.distance(poly_utm)
        is_inside = distance_m == 0.0

        # Nearest point on polygon (even if inside, distance 0)
        nearest_on_poly = nearest_points(pt_utm, poly_utm)[1]
        cx, cy = nearest_on_poly.x, nearest_on_poly.y
        clon, clat = to_wgs.transform(cx, cy)

        distance_ft = distance_m * 3.28084
        return {
            "distance_feet": round(distance_ft, 2),
            "distance_meters": round(distance_m, 2),
            "is_inside": bool(is_inside),
            "closest_point_lat": round(float(clat), 6),
            "closest_point_lon": round(float(clon), 6),
        }
    except Exception as e:
        return {"error": f"Distance calculation failed: {str(e)}"}

@tool
def standardize_excel_data(
    input_file: str,
    output_file: Optional[str] = None,
    column_mapping: Optional[str] = None,  # JSON string for custom mapping
) -> str:
    """
    Standardize Excel/CSV data to match the optimal template format.

    This tool MUST be used when users upload data files to ensure compatibility with all other tools.
    Maps columns from various formats to the standard EXCEL_TEMPLATE_HEADERS structure.

    Args:
        input_file: Path to input Excel (.xlsx) or CSV file
        output_file: Output path (defaults to output/excel_files/[name]_standardized.xlsx)
        column_mapping: Optional JSON string with custom column mappings
                       e.g., '{"Business":"Site Name or Business Name","Lat":"Latitude (NAD83)"}'

    Returns:
        Status JSON with standardization results and column mapping details
    """
    try:
        from pathlib import Path
        import pandas as pd
        import json
        import re

        input_path = Path(input_file)
        if not input_path.exists():
            return json.dumps({"error": f"Input file not found: {input_file}"})

        # Read input file
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path)

        original_columns = list(df.columns)

        # Parse custom mapping if provided
        custom_map = {}
        if column_mapping:
            try:
                custom_map = json.loads(column_mapping)
            except:
                pass

        # Auto-detect column mappings using fuzzy matching
        mapping = {}
        used_targets = set()

        # Common variations for each standard column
        variations = {
            "Site Name or Business Name": ["site", "business", "name", "facility", "company", "location name"],
            "Person Contacted": ["contact", "person", "contacted", "representative", "contact person"],
            "Tank Capacity": ["capacity", "volume", "gallons", "size", "tank size"],
            "Tank Measurements": ["measurements", "dimensions", "tank dimensions", "size"],
            "Has Dike": ["has dike", "dike", "secondary containment", "containment", "berm"],
            "Dike Measurements": ["dike measurements", "dike dimensions", "containment size", "berm dimensions", "dike size"],
            "Diesel": ["diesel", "diesel fuel", "is diesel", "fuel type diesel"],
            "Pressurized Gas": ["gas", "pressurized", "lpg", "propane", "compressed gas"],
            "Acceptable Separation Distance Calculated": ["separation", "acceptable distance", "safe distance"],
            "Approximate Distance to Site (approximately)": ["distance to site", "site distance", "approximate distance"],
            "Compliance": ["compliance", "compliant", "status", "violation"],
            "Additional information": ["notes", "comments", "additional", "remarks", "info"],
            "Latitude (NAD83)": ["lat", "latitude", "y", "northing", "lat_nad83"],
            "Longitude (NAD83)": ["lon", "long", "longitude", "x", "easting", "lon_nad83"],
            "Calculated Distance to Polygon (ft)": ["calculated distance", "distance to polygon", "polygon distance"],
            "Tank Type": ["fuel", "fuel type", "type", "tank type", "storage type", "tank category"],
        }

        # Try to map columns
        for orig_col in original_columns:
            orig_lower = orig_col.lower()

            # Check custom mapping first
            if orig_col in custom_map:
                target = custom_map[orig_col]
                if target in EXCEL_TEMPLATE_HEADERS and target not in used_targets:
                    mapping[orig_col] = target
                    used_targets.add(target)
                    continue

            # Auto-match based on variations
            best_match = None
            best_score = 0

            for target_col, keywords in variations.items():
                if target_col in used_targets:
                    continue

                # Check for keyword matches
                for keyword in keywords:
                    if keyword in orig_lower:
                        score = len(keyword) / len(orig_lower)  # Higher score for more complete match
                        if score > best_score:
                            best_score = score
                            best_match = target_col

            if best_match and best_score > 0.3:  # Threshold for accepting match
                mapping[orig_col] = best_match
                used_targets.add(best_match)

        # Create standardized dataframe
        standardized_data = {}
        for header in EXCEL_TEMPLATE_HEADERS:
            standardized_data[header] = []

        # Transfer data with mapping
        for idx, row in df.iterrows():
            for orig_col, target_col in mapping.items():
                if idx >= len(standardized_data[target_col]):
                    # Extend lists to match row count
                    for col in standardized_data:
                        while len(standardized_data[col]) <= idx:
                            standardized_data[col].append(None)

                value = row[orig_col]

                # Special handling for fuel type columns when mapped from a general "Fuel Type" column
                if orig_col in mapping and "fuel" in orig_col.lower() and target_col == "Tank Type":
                    # Also set Diesel and Pressurized Gas flags based on fuel type
                    if pd.notna(value):
                        fuel_lower = str(value).lower()
                        # Check for pressurized gas indicators (avoid matching "gasoline")
                        if any(gas in fuel_lower for gas in ['propane', 'lpg', 'pressurized', 'compressed']):
                            standardized_data["Diesel"][idx] = "No"
                            standardized_data["Pressurized Gas"][idx] = "Yes"
                        elif 'diesel' in fuel_lower:
                            standardized_data["Diesel"][idx] = "Yes"
                            standardized_data["Pressurized Gas"][idx] = "No"
                        else:
                            # Default liquid fuels (gasoline, fuel oil, etc.) to diesel
                            standardized_data["Diesel"][idx] = "Yes"
                            standardized_data["Pressurized Gas"][idx] = "No"

                # Convert boolean columns
                if target_col in ["Has Dike", "Diesel", "Pressurized Gas"]:
                    if pd.notna(value):
                        value_str = str(value).lower()
                        if value_str in ['1', 'true', 'yes', 'y', 't']:
                            value = "Yes"
                        elif value_str in ['0', 'false', 'no', 'n', 'f']:
                            value = "No"
                        else:
                            value = "No"  # Default

                standardized_data[target_col][idx] = value

        # Ensure all columns have same length
        max_len = max(len(v) for v in standardized_data.values()) if standardized_data else 0
        for col in standardized_data:
            while len(standardized_data[col]) < max_len:
                standardized_data[col].append(None)

        # Create output dataframe
        standardized_df = pd.DataFrame(standardized_data)

        # Determine output path
        if not output_file:
            stem = input_path.stem
            output_path = EXCEL_MCP_FILES_DIR_DEFAULT / f"{stem}_standardized.xlsx"
        else:
            output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            standardized_df.to_excel(writer, sheet_name='Tank Data', index=False)

            # Format the Excel file
            worksheet = writer.sheets['Tank Data']
            from openpyxl.styles import Font, Alignment

            # Bold headers
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # Auto-adjust column widths
            for column_cells in worksheet.columns:
                length = max(len(str(cell.value or '')) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 40)

        # Prepare summary
        unmapped = [col for col in original_columns if col not in mapping]

        result = {
            "status": "success",
            "output": str(output_path.absolute()),
            "rows_processed": len(standardized_df),
            "columns_mapped": len(mapping),
            "mapping": mapping,
            "unmapped_columns": unmapped,
            "message": f"Standardized {len(mapping)}/{len(original_columns)} columns to template format"
        }

        if unmapped:
            result["warning"] = f"Could not map columns: {', '.join(unmapped)}"

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Standardization failed: {str(e)}"})

@tool
def generate_verified_tanks_json(
    input_file: str,
    output_file: Optional[str] = None,
    standardize_first: bool = True,
) -> str:
    """
    Parse Excel/CSV data and generate verified JSON with Pydantic validation.

    Enforces units: volumes in U.S. gallons, dike dimensions in ft x ft or ft².
    Automatically handles unit conversions and validates tank data structure.

    FUEL TYPE DETERMINATION:
    - Only 'diesel' or 'pressurized_gas' are valid tank types
    - Pressurized gas: propane, LPG, compressed gases
    - Diesel: diesel fuel, heating oil, gasoline, kerosene, most liquid fuels
    - When processing data, the agent should:
      1. Check if Diesel/Pressurized Gas boolean columns are already set
      2. Use context clues (e.g., "propane tank" → pressurized_gas)
      3. Default ambiguous cases to diesel (most common)
      4. Can pre-process data to set boolean columns for better accuracy

    Args:
        input_file: Path to Excel or CSV file with tank data
        output_file: Output JSON path (defaults to output/verified_tanks.json)
        standardize_first: Whether to standardize the data format first (recommended)

    Returns:
        Status JSON with generation results and validation details
    """
    try:
        from pathlib import Path
        import pandas as pd
        import json
        import re
        from typing import List, Dict, Any, Optional, Union

        # Define tank validation schema
        def validate_tank_data(tank: Dict[str, Any]) -> Dict[str, Any]:
            """Validate and normalize a single tank record."""
            validated = {}

            # Name (required)
            name = tank.get("Site Name or Business Name") or tank.get("name") or tank.get("site")
            if not name:
                raise ValueError("Tank name/site is required")
            validated["name"] = str(name).strip()

            # Volume (required) - parse and convert to gallons
            volume_str = tank.get("Tank Capacity") or tank.get("volume") or tank.get("capacity")
            if not volume_str:
                raise ValueError(f"Tank volume required for {name}")

            volume_val = parse_volume_to_gallons(str(volume_str))
            if volume_val <= 0:
                raise ValueError(f"Volume must be positive for {name}")
            validated["volume"] = volume_val

            # Type - determine from multiple sources
            tank_type = determine_tank_type(tank)
            validated["type"] = tank_type

            # Has Dike (boolean)
            has_dike_val = tank.get("Has Dike") or tank.get("has_dike")
            has_dike = parse_boolean(has_dike_val)
            validated["has_dike"] = has_dike

            # Dike dimensions (if has_dike)
            if has_dike:
                dike_str = tank.get("Dike Measurements") or tank.get("Dike Size") or tank.get("dike_dims")
                # Check if the value is valid (not NaN, not None, not empty)
                if dike_str and pd.notna(dike_str) and str(dike_str).strip() and str(dike_str).upper() not in ["N/A", "NAN", ""]:
                    try:
                        dike_dims = parse_dike_dimensions(str(dike_str))
                        validated["dike_dims"] = dike_dims
                    except ValueError as e:
                        raise ValueError(f"Cannot parse dike dimensions for {name}: {str(dike_str)}")
                else:
                    raise ValueError(f"Dike dimensions required for {name} when has_dike is true")
            else:
                validated["dike_dims"] = None

            return validated

        def parse_volume_to_gallons(volume_str: str) -> float:
            """Parse volume string and convert to gallons."""
            volume_str = volume_str.strip().lower()

            # Extract number and unit - improved regex to handle "bbl" better
            match = re.search(r'([\d,]+(?:\.\d+)?)\s*(\w*)', volume_str)
            if not match:
                raise ValueError(f"Cannot parse volume: {volume_str}")

            num_str = match.group(1).replace(',', '')
            value = float(num_str)
            unit = match.group(2) if match.group(2) else ''

            # Convert to gallons - check for specific units first
            if 'bbl' in unit or 'barrel' in unit:
                return value * 42.0
            elif 'm3' in unit or 'cubic' in unit:
                return value * 264.172
            elif 'liter' in unit or unit == 'l':
                return value * 0.264172
            elif 'gal' in unit:
                return value
            else:
                # Default to gallons if no unit or unrecognized unit
                return value

        def determine_tank_type(tank: Dict[str, Any]) -> str:
            """Determine tank type from various fields. Only diesel or pressurized_gas allowed.

            The agent should help interpret ambiguous cases:
            - Pressurized gas includes: propane, LPG, compressed gases
            - Diesel includes: diesel fuel, heating oil, most liquid fuels
            - When unclear, the agent can use context clues from the data
            """
            # Check explicit boolean flags first
            is_diesel = parse_boolean(tank.get("Diesel"))
            is_gas = parse_boolean(tank.get("Pressurized Gas"))

            if is_diesel:
                return 'diesel'
            if is_gas:
                return 'pressurized_gas'

            # Check type field for clear indicators
            tank_type_str = str(tank.get("Tank Type", "")).lower()

            # Clear pressurized gas indicators (avoid matching "gasoline")
            pressurized_indicators = ['propane', 'lpg', 'compressed', 'pressurized']
            if any(indicator in tank_type_str for indicator in pressurized_indicators):
                return 'pressurized_gas'

            # Clear diesel indicators
            diesel_indicators = ['diesel', 'heating oil', 'fuel oil']
            if any(indicator in tank_type_str for indicator in diesel_indicators):
                return 'diesel'

            # For ambiguous cases (gasoline, kerosene, etc.), default to diesel
            # The agent can override this by setting the boolean columns explicitly
            return 'diesel'

        def parse_boolean(value: Any) -> bool:
            """Parse various boolean representations."""
            if pd.isna(value) or value is None:
                return False
            value_str = str(value).lower().strip()
            return value_str in ['yes', 'y', 'true', '1', 't']

        def parse_dike_dimensions(dike_str: str) -> Union[List[float], float]:
            """Parse dike dimensions to [length, width] in feet or area in ft²."""
            dike_str = dike_str.strip().lower()

            # Pattern 1: LxW format (e.g., "15x12x3" or "15x12")
            match = re.search(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)', dike_str)
            if match:
                length = float(match.group(1))
                width = float(match.group(2))
                return [length, width]

            # Pattern 2: L ft x W ft format
            match = re.search(r'(\d+(?:\.\d+)?)\s*ft?\s*x\s*(\d+(?:\.\d+)?)\s*ft?', dike_str)
            if match:
                length = float(match.group(1))
                width = float(match.group(2))
                return [length, width]

            # Pattern 3: Area in square feet
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq|square)?\s*(?:ft|feet)', dike_str)
            if match:
                area = float(match.group(1))
                return area

            # Pattern 4: Just a number (assume ft²)
            match = re.search(r'(\d+(?:\.\d+)?)', dike_str)
            if match:
                area = float(match.group(1))
                # If it looks like dimensions (>50), assume area; otherwise might be single dimension
                if area > 50:
                    return area
                else:
                    # Assume square dike if single dimension given
                    return [area, area]

            raise ValueError(f"Cannot parse dike dimensions: {dike_str}")

        # Main processing
        input_path = Path(input_file)
        if not input_path.exists():
            return json.dumps({"error": f"Input file not found: {input_file}"})

        # If standardize_first, run standardization
        if standardize_first:
            # Call the function directly, not as a tool
            std_result = standardize_excel_data.func(
                input_file=str(input_path),
                output_file=None  # Use default
            )
            std_data = json.loads(std_result)
            if std_data.get("status") != "success":
                return json.dumps({"error": f"Standardization failed: {std_data.get('error')}"})

            # Use standardized file
            working_file = std_data.get("output")
        else:
            working_file = str(input_path)

        # Read the data
        if working_file.endswith('.csv'):
            df = pd.read_csv(working_file)
        else:
            df = pd.read_excel(working_file)

        # Process each row
        tanks_data = []
        validation_errors = []

        for idx, row in df.iterrows():
            try:
                tank_dict = row.to_dict()
                validated_tank = validate_tank_data(tank_dict)
                tanks_data.append(validated_tank)
            except Exception as e:
                validation_errors.append({
                    "row": idx + 1,
                    "name": row.get("Site Name or Business Name", f"Row {idx+1}"),
                    "error": str(e)
                })

        if not tanks_data and validation_errors:
            return json.dumps({
                "error": "No valid tanks found",
                "validation_errors": validation_errors
            })

        # Create the final structure
        output_data = {"tanks": tanks_data}

        # Validate with Pydantic schema if available
        try:
            # Import schema classes
            from pydantic import BaseModel, Field, validator

            class TankSchema(BaseModel):
                name: str
                volume: float = Field(gt=0)
                type: str
                has_dike: bool
                dike_dims: Optional[Union[List[float], float]] = None

                @validator('type')
                def validate_type(cls, v):
                    valid_types = ['diesel', 'pressurized_gas']
                    if v not in valid_types:
                        # Force to diesel if not recognized
                        return 'diesel'
                    return v

            class TanksList(BaseModel):
                tanks: List[TankSchema]

            # Validate with Pydantic
            validated = TanksList(**output_data)
            final_data = validated.dict()
        except:
            # If Pydantic not available, use raw data
            final_data = output_data

        # Write JSON file
        if not output_file:
            output_path = Path("output") / "verified_tanks.json"
        else:
            output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        # Prepare result
        result = {
            "status": "success",
            "output": str(output_path.absolute()),
            "tanks_processed": len(tanks_data),
            "message": f"Generated verified JSON with {len(tanks_data)} tanks"
        }

        if validation_errors:
            result["warning"] = f"Skipped {len(validation_errors)} invalid rows"
            result["validation_errors"] = validation_errors

        # Add summary statistics
        result["summary"] = {
            "total_volume_gallons": sum(t["volume"] for t in tanks_data),
            "diesel_tanks": sum(1 for t in tanks_data if t["type"] == "diesel"),
            "gas_tanks": sum(1 for t in tanks_data if t["type"] == "pressurized_gas"),
            "tanks_with_dike": sum(1 for t in tanks_data if t["has_dike"]),
            "tanks_without_dike": sum(1 for t in tanks_data if not t["has_dike"])
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"JSON generation failed: {str(e)}"})

@tool
def tank_volume_calculator(
    dimensions_str: str,
    unit: str = "inches",  # "inches" or "feet"
) -> dict:
    """
    Calculate rectangular tank volume in US gallons from dimensions like
    39"x46"x229"; 182"x55"x26"

    - Supports multiple entries separated by semicolons
    - unit = inches (default) or feet
    - Gallons = (L * W * H in cubic inches) / 231
    """
    import re as _re
    if unit not in {"inches", "feet"}:
        return {"error": "Unit must be 'inches' or 'feet'."}

    entries = [e.strip() for e in (dimensions_str or "").split(";")]
    results: List[Dict[str, Any]] = []
    pattern = _re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", _re.IGNORECASE)

    for entry in entries:
        if not entry:
            continue
        cleaned = _re.sub(r"[\"']", "", entry)  # remove inch/quote symbols
        m = pattern.search(cleaned)
        if not m:
            results.append({"dimensions": entry, "error": "Invalid format. Use LxWxH."})
            continue
        try:
            L, W, H = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if unit == "feet":
                L *= 12.0; W *= 12.0; H *= 12.0
            vol_cuin = L * W * H
            vol_gal = vol_cuin / 231.0
            pretty_dims = f"{int(L)}x{int(W)}x{int(H)} in" if unit == "inches" else f"{L/12:.1f}x{W/12:.1f}x{H/12:.1f} ft"
            results.append({
                "dimensions": pretty_dims,
                "volume_gallons": round(vol_gal, 2),
            })
        except Exception:
            results.append({"dimensions": entry, "error": "Non-numeric dimensions."})

    return {"status": "success", "results": results}

tools = [
    generate_map,
    kmz_tool,
    create_excel_template,
    standardize_excel_data,
    generate_verified_tanks_json,
    convert_coordinates,
    calculate_distance_to_polygon,
    tank_volume_calculator,
]
llm_with_tools = llm.bind_tools(tools)

# Build the graph
workflow = StateGraph(state_schema=AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" if tool calls, else END
    {
        "tools": "tools",
        END: END,
    },
)

# Set entry point
workflow.set_entry_point("agent")

# Compile with memory checkpoint
memory = MemorySaver()
graph = None  # Will be compiled after loading optional MCP tools


async def load_excel_mcp_tools_from_env() -> List:
    """Load Excel MCP tools using langchain-mcp-adapters if configured.

    Defaults to HTTP transport to avoid spawning a stdio subprocess.

    Env vars:
    - EXCEL_MCP_ENABLE: set to '1' to enable loading (or provide TRANSPORT/URL)
    - EXCEL_MCP_TRANSPORT: 'http'/'streamable_http' (default), 'sse', or 'stdio'
    - EXCEL_MCP_URL: when using http/sse (default: http://127.0.0.1:8000/mcp)
    - EXCEL_MCP_COMMAND: when using stdio (default: 'uvx')
    - EXCEL_MCP_ARGS: when using stdio (default: 'excel-mcp-server stdio')
    """
    tools: List = []
    if MultiServerMCPClient is None:
        return tools

    # Hardcoded enablement and HTTP transport by default
    if not EXCEL_MCP_ENABLED or MultiServerMCPClient is None:
        return tools
    transport = "http"

    cfg: Dict[str, Dict] = {}
    if transport in ("http", "streamable_http"):
        desired_url = EXCEL_MCP_URL_DEFAULT
        # Autostart a local HTTP server and get the actual URL (handles port-in-use)
        try:
            actual_url = await _ensure_excel_http_server(desired_url)
        except Exception as e:
            print(f"[Excel MCP] Autostart failed ({e}); will try desired URL anyway…")
            actual_url = desired_url
        cfg["excel"] = {"transport": "streamable_http", "url": actual_url}
    elif transport == "sse":
        url = os.getenv("EXCEL_MCP_URL", "http://127.0.0.1:8000/sse")
        cfg["excel"] = {"transport": "sse", "url": url}
    elif transport == "stdio":
        # Not used by default, but left for completeness
        cfg["excel"] = {"transport": "stdio", "command": "uvx", "args": ["excel-mcp-server", "stdio"]}
    else:
        # Fallback: treat unknown value as HTTP
        cfg["excel"] = {"transport": "streamable_http", "url": EXCEL_MCP_URL_DEFAULT}

    try:
        client = MultiServerMCPClient(cfg)  # type: ignore
        # Retry a few times to allow autostarted server to come up
        for attempt in range(10):
            try:
                tools = await client.get_tools()
                break
            except Exception:
                if attempt == 9:
                    raise
                await asyncio.sleep(0.6)
        return tools
    except Exception as e:
        print(f"[Excel MCP] Skipping Excel tools: {e}")
        return []


# --- Managed HTTP server helpers ---
_excel_http_proc: Optional[subprocess.Popen] = None


async def _ensure_excel_http_server(url: str) -> str:
    """Start an excel-mcp-server (HTTP) as a child process if not running.

    Chooses uvx excel-mcp-server streamable-http by default. You can override command
    via EXCEL_MCP_HTTP_COMMAND and EXCEL_MCP_HTTP_ARGS.
    """
    global _excel_http_proc
    if _excel_http_proc and _excel_http_proc.poll() is None:
        return url

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or EXCEL_MCP_PORT_DEFAULT

    # If something already listens on the desired port, assume it's the server and reuse the URL
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            if s.connect_ex((host, port)) == 0:
                return url
        except Exception:
            pass

    # Find a free port near the default
    free_port = None
    for candidate in [port] + list(range(port + 1, port + 21)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            try:
                s2.bind((host, candidate))
                free_port = candidate
                break
            except OSError:
                continue
    if free_port is None:
        free_port = port

    # Build command
    cmd = os.getenv("EXCEL_MCP_HTTP_COMMAND")
    args_env = os.getenv("EXCEL_MCP_HTTP_ARGS")
    if cmd and args_env:
        cmd_list = [cmd] + shlex.split(args_env)
    else:
        # Default to uvx runner with streamable-http; excel-mcp-server uses FASTMCP_PORT env, not --port flags
        cmd_list = [
            os.getenv("EXCEL_MCP_COMMAND", "uvx"),
            "excel-mcp-server",
            "streamable-http",
        ]

    # Launch server with output suppressed
    log_path = Path("output/excel_mcp_server.log")
    log_path.parent.mkdir(exist_ok=True)
    # Prepare environment: set FASTMCP_PORT and EXCEL_FILES_PATH for the server
    env = os.environ.copy()
    env["FASTMCP_PORT"] = str(free_port)
    # Default Excel files dir inside current project unless provided by user
    default_excel_dir = EXCEL_MCP_FILES_DIR_DEFAULT.resolve()
    default_excel_dir.mkdir(parents=True, exist_ok=True)
    env["EXCEL_FILES_PATH"] = str(default_excel_dir)

    with open(log_path, "ab", buffering=0) as logf:
        _excel_http_proc = subprocess.Popen(
            cmd_list,
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            env=env,
        )

    # Ensure cleanup on exit
    def _cleanup():
        global _excel_http_proc
        if _excel_http_proc and _excel_http_proc.poll() is None:
            try:
                _excel_http_proc.terminate()
            except Exception:
                pass
    atexit.register(_cleanup)

    # Give the server a moment to start
    await asyncio.sleep(0.6)

    # Return the actual URL with the chosen port
    return f"http://{host}:{free_port}/mcp"

# Chatbot interaction loop
async def chat_async():
    config = {"configurable": {"thread_id": "map_chat_thread"}}
    print("Chatbot ready! Type 'exit' to quit.")
    print("Example: 'Generate a map from /path/to/my.kmz' (defaults to satellite style).")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Track whether we're currently streaming tokens from the AI
        streaming_ai_response = False

        # Invoke the graph asynchronously with multi-mode streaming
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode=["updates", "messages", "custom"],
        ):
            # 1) LLM token stream - these come as tuples (message_chunk, metadata)
            if mode == "messages":
                try:
                    # Unpack the tuple returned by messages mode
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        message_chunk, metadata = chunk
                        # Extract the content from the message chunk
                        if hasattr(message_chunk, 'content') and message_chunk.content:
                            if not streaming_ai_response:
                                print("\nAssistant: ", end="", flush=True)
                                streaming_ai_response = True
                            print(message_chunk.content, end="", flush=True)
                except Exception as e:
                    # Debug: uncomment if you need to see what's happening
                    # print(f"\n[Debug] Message streaming error: {e}, chunk type: {type(chunk)}")
                    pass
                continue

            # 2) Agent/tool progress updates
            if mode == "updates" and isinstance(chunk, dict):
                # Reset streaming flag when we get an update (means LLM is done)
                if streaming_ai_response:
                    print()  # New line after streamed response
                    streaming_ai_response = False

                for node, update in chunk.items():
                    msgs = update.get("messages") if isinstance(update, dict) else None
                    if not msgs:
                        continue
                    try:
                        last = msgs[-1]
                        # Check if this is a tool message
                        if hasattr(last, 'type') and last.type == "tool":
                            print(f"\nTool Result: {last.content}")
                            # Provide helpful hint for missing paths in Excel tools
                            if last.content and "No such file or directory" in str(last.content):
                                print("Hint: Use a path under 'output/excel_files' or a relative filename. Missing parent folders are not created for absolute paths.")
                        # Note: AI messages are now streamed, so we don't print them here
                    except Exception:
                        continue

            # 3) Custom stream (from tools via get_stream_writer)
            if mode == "custom":
                try:
                    data = chunk
                    if isinstance(data, dict) and data.get("type") in {"progress", "error"}:
                        tag = "Progress" if data.get("type") == "progress" else "Error"
                        msg = data.get("message", "")
                        print(f"\n[{tag}] {msg}")
                    else:
                        print(f"\n[Custom] {data}")
                except Exception:
                    pass

        # Ensure we end with a newline if we were streaming
        if streaming_ai_response:
            print()  # Final newline

if __name__ == "__main__":
    # Ensure required directories exist (optional)
    Path("images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    # Assemble tools: local map tool + optional Excel MCP tools
    async def _bootstrap_and_run():
        global llm_with_tools, graph
        excel_tools: List = []
        if EXCEL_MCP_ENABLED:
            excel_tools = await load_excel_mcp_tools_from_env()
            if excel_tools:
                print(f"[Excel MCP] Loaded {len(excel_tools)} tool(s) from Excel MCP server.")
        all_tools = tools + (excel_tools or [])
        llm_with_tools = llm.bind_tools(all_tools)

        # Build and compile graph with the combined tools
        workflow = StateGraph(state_schema=AgentState)
        workflow.add_node("agent", agent)
        workflow.add_node("tools", ToolNode(all_tools))
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        workflow.set_entry_point("agent")
        graph = workflow.compile(checkpointer=memory)

        await chat_async()

    asyncio.run(_bootstrap_and_run())
