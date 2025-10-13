from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EdgeCoord:
    x: int
    y: int


@dataclass
class NoduleSlice:
    sop_uid: str
    z_position: float
    inclusion: bool
    edges: List[EdgeCoord]


@dataclass
class NoduleAnnotation:
    nodule_id: str
    slices: List[NoduleSlice] = field(default_factory=list)
    characteristics: Dict[str, float] = field(default_factory=dict)

    def centroid(
        self,
        sop_uid_to_index: Dict[str, Dict[str, float]],
    ) -> Optional[Tuple[float, float, float]]:
        z_acc: List[float] = []
        y_acc: List[float] = []
        x_acc: List[float] = []
        for sl in self.slices:
            meta = sop_uid_to_index.get(sl.sop_uid)
            if meta is None or not sl.edges:
                continue
            z_acc.append(float(meta["index"]))
            y_acc.extend(float(edge.y) for edge in sl.edges)
            x_acc.extend(float(edge.x) for edge in sl.edges)
        if not z_acc or not y_acc or not x_acc:
            return None
        return (
            float(np.mean(z_acc)),
            float(np.mean(y_acc)),
            float(np.mean(x_acc)),
        )


def _namespace(root: ET.Element) -> Dict[str, str]:
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        return {"lidc": uri}
    return {"lidc": ""}


def parse_annotation_xml(xml_path: Path) -> List[NoduleAnnotation]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _namespace(root)
    annotations: List[NoduleAnnotation] = []
    for session in root.findall("lidc:readingSession", ns):
        for nodule_elem in session.findall("lidc:unblindedReadNodule", ns):
            nodule_id = nodule_elem.findtext("lidc:noduleID", default="unknown", namespaces=ns)
            characteristics: Dict[str, float] = {}
            char_elem = nodule_elem.find("lidc:characteristics", ns)
            if char_elem is not None:
                for child in char_elem:
                    try:
                        characteristics[child.tag.split("}")[-1]] = float(child.text)
                    except (TypeError, ValueError):
                        continue
            slices: List[NoduleSlice] = []
            for roi_elem in nodule_elem.findall("lidc:roi", ns):
                sop_uid = roi_elem.findtext("lidc:imageSOP_UID", default="", namespaces=ns)
                z_position_text = roi_elem.findtext("lidc:imageZposition", default="0", namespaces=ns)
                inclusion_text = roi_elem.findtext("lidc:inclusion", default="FALSE", namespaces=ns)
                edges: List[EdgeCoord] = []
                for edge_elem in roi_elem.findall("lidc:edgeMap", ns):
                    x_text = edge_elem.findtext("lidc:xCoord", default="0", namespaces=ns)
                    y_text = edge_elem.findtext("lidc:yCoord", default="0", namespaces=ns)
                    try:
                        edges.append(EdgeCoord(x=int(float(x_text)), y=int(float(y_text))))
                    except ValueError:
                        continue
                try:
                    z_position = float(z_position_text)
                except ValueError:
                    z_position = 0.0
                slices.append(
                    NoduleSlice(
                        sop_uid=sop_uid.strip(),
                        z_position=z_position,
                        inclusion=inclusion_text.strip().upper() == "TRUE",
                        edges=edges,
                    )
                )
            annotations.append(NoduleAnnotation(nodule_id=nodule_id, slices=slices, characteristics=characteristics))
    return annotations


__all__ = [
    "EdgeCoord",
    "NoduleSlice",
    "NoduleAnnotation",
    "parse_annotation_xml",
]
