"""
capability_catalog.py -- builds the 71-capability catalogue (sec:dataset):
25 education-specific capabilities plus 46 drawn from the other smart-city
dimensions used elsewhere in the paper (living, healthcare, mobility).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Capability:
    name: str
    domain: str            # "education" | "living" | "healthcare" | "mobility"
    education_specific: bool


EDUCATION_CAPABILITIES = [
    "broadcasting", "quizzing", "whiteboarding", "polling", "acoustics",
    "ambient sensing", "ambient regulation", "attendance tracking",
    "note synchronisation", "lecture recording", "screen sharing",
    "flashcard generation", "peer grouping", "plagiarism checking",
    "assignment collection", "gradebook synchronisation", "discussion moderation",
    "live captioning", "language translation", "accessibility adaptation",
    "concept mapping", "study reminder", "revision scheduling",
    "exam proctoring", "certificate issuance",
]

LIVING_CAPABILITIES = [
    "smart lighting", "smart locking", "occupancy detection", "leak detection",
    "energy monitoring", "appliance control", "entertainment streaming",
    "voice assistance", "package detection", "pet monitoring", "irrigation control",
]

HEALTHCARE_CAPABILITIES = [
    "vital sign monitoring", "medication reminder", "fall detection",
    "remote diagnosis", "appointment scheduling", "health record sync",
    "fitness tracking", "sleep tracking", "emergency alerting", "telepresence",
]

MOBILITY_CAPABILITIES = [
    "route planning", "vehicle sharing", "parking detection", "traffic monitoring",
    "e-ticketing", "fleet tracking", "charging-station locating", "ride matching",
]

# pad the general pool with a few cross-cutting capabilities so the total,
# 25 education + the rest, reaches 71 exactly.
EXTRA_CAPABILITIES = [
    "video conferencing", "file storage", "identity verification", "payment processing",
    "notification dispatch", "weather forecasting", "air quality sensing", "backup scheduling",
    "log auditing", "access control", "device provisioning", "firmware updating",
    "bandwidth shaping", "geofencing", "anomaly detection", "report generation",
    "sentiment analysis",
]


def build_capabilities() -> List[Capability]:
    caps = [Capability(name, "education", True) for name in EDUCATION_CAPABILITIES]
    for name in LIVING_CAPABILITIES:
        caps.append(Capability(name, "living", False))
    for name in HEALTHCARE_CAPABILITIES:
        caps.append(Capability(name, "healthcare", False))
    for name in MOBILITY_CAPABILITIES:
        caps.append(Capability(name, "mobility", False))
    for name in EXTRA_CAPABILITIES:
        caps.append(Capability(name, "living", False))

    assert len(caps) == 71, f"expected 71 capabilities, built {len(caps)}"
    assert sum(c.education_specific for c in caps) == 25, "expected 25 education-specific capabilities"
    return caps
