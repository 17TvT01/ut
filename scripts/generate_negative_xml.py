from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - script level feedback
    raise SystemExit("pydicom is required. Install it with `pip install pydicom`.") from exc


NAMESPACE = "http://www.nih.gov/LIDC"
DEFAULT_FILENAME = "auto_negative.xml"


def find_series_directories(root: Path) -> List[Path]:
    """Find all directories under root that contain at least one DICOM file."""
    series_dirs = set()
    for pattern in ("*.dcm", "*.DCM"):
        for dcm_path in root.rglob(pattern):
            series_dirs.add(dcm_path.parent)
    return sorted(series_dirs)


def series_has_xml(series_dir: Path) -> bool:
    return any(series_dir.glob("*.xml"))


def detect_series_uid(series_dir: Path) -> Optional[str]:
    for dcm_path in sorted(series_dir.rglob("*.dcm")) + sorted(series_dir.rglob("*.DCM")):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        series_uid = getattr(ds, "SeriesInstanceUID", None)
        if series_uid:
            return str(series_uid)
    return None


def build_empty_annotation(series_uid: Optional[str]) -> ET.Element:
    root = ET.Element("LidcReadMessage", xmlns=NAMESPACE)
    header = ET.SubElement(root, "ResponseHeader")
    if series_uid:
        ET.SubElement(header, "SeriesInstanceUID").text = series_uid
    ET.SubElement(root, "readingSession")
    return root


def write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def generate_for_root(root: Path, filename: str, overwrite: bool) -> Sequence[Path]:
    created: List[Path] = []
    for series_dir in find_series_directories(root):
        xml_path = series_dir / filename
        if series_has_xml(series_dir) and not overwrite and not xml_path.exists():
            # Directory already has some annotation file, skip to avoid conflicts.
            continue
        if xml_path.exists() and not overwrite:
            continue
        series_uid = detect_series_uid(series_dir)
        xml_root = build_empty_annotation(series_uid)
        write_xml(xml_root, xml_path)
        created.append(xml_path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate empty LIDC-style XML annotations for negative (no nodule) studies."
    )
    parser.add_argument("root", type=Path, help="Root folder that contains DICOM studies.")
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Name of the XML file to create in each study (default: {DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing XML files with the same name.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path '{root}' does not exist.")

    created = generate_for_root(root, args.filename, args.overwrite)
    if created:
        print(f"Created {len(created)} annotation files:")
        for xml_path in created:
            print(f" - {xml_path}")
    else:
        print("No new annotations were created (folders may already contain XML files).")


if __name__ == "__main__":
    main()
