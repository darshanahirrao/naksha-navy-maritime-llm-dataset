#!/usr/bin/env python3
"""
Naksha Navy: synthetic maritime question-answer dataset generator.

Builds the six question-answer JSON files used in the hands-on fine-tuning
lecture "Fine tuning a LLM for maritime domain awareness and decision
support". One file per data type the officers work with:

    domain_vessel_reports.json      vessel identity and registry dossiers
    contact_classification.json     classification doctrine and contact entries
    navigational_warnings.json      NAVAREA IX, coastal warnings, notices to mariners
    ais_vessel_tracks.json          AIS track, passage and port call records
    shipping_reports.json           monthly North Arabian Sea shipping reports
    synthetic_contact_reports.json  synthetic contact reports from exercise units

Every vessel, IMO and MMSI number, owner, warning, contact, incident and
statistic below is invented. The world is internally consistent on purpose:
the same three ships, ports, dates and warnings recur across all six files so
a lecture can follow one story from a sighting on the horizon to a decision.

Run:  python3 scripts/generate_dataset.py
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260922
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset"
GENERATED = "2026-09-22"
COVERAGE = {"from": "2026-01-01", "to": "2026-08-31"}

DISCLAIMER = (
    "SYNTHETIC TRAINING DATA. Every vessel, IMO and MMSI number, owner, port "
    "call, warning, contact, incident and statistic in this file is fictional "
    "and was generated for a training lecture. Nothing here describes a real "
    "ship, broadcast, incident or port condition, and nothing here may be used "
    "for navigation, intelligence or any operational decision."
)

rng = random.Random(SEED)


# --------------------------------------------------------------------------
# geo and formatting helpers
# --------------------------------------------------------------------------


def haversine_nm(lat1, lon1, lat2, lon2):
    radius = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


_COMPASS = [
    "north", "north-north-east", "north-east", "east-north-east",
    "east", "east-south-east", "south-east", "south-south-east",
    "south", "south-south-west", "south-west", "west-south-west",
    "west", "west-north-west", "north-west", "north-north-west",
]


def compass(deg):
    return _COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def fmt_pos(lat, lon):
    return f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'} {abs(lon):.2f}{'E' if lon >= 0 else 'W'}"


def dm(value, hemi):
    deg = int(abs(value))
    minutes = (abs(value) - deg) * 60
    return f"{deg:02d}-{minutes:04.1f}{hemi}"


def parse_dt(text):
    return datetime.strptime(text, "%Y-%m-%dT%H%MZ")


def fmt_dt(dt):
    return f"{dt.day} {dt.strftime('%B %Y')} at {dt.strftime('%H%M')}Z"


def fmt_date(dt):
    return f"{dt.day} {dt.strftime('%B %Y')}"


def fmt_short(dt):
    return dt.strftime("%d %b %Y").lstrip("0")


# --------------------------------------------------------------------------
# the fictional world: ports, routeing, exercise areas
# --------------------------------------------------------------------------

PORTS = {
    "AEJEA": ("Jebel Ali", 25.00, 55.06, "United Arab Emirates"),
    "AEFJR": ("Fujairah", 25.12, 56.35, "United Arab Emirates"),
    "INMUN": ("Mundra", 22.84, 69.72, "India"),
    "INIXY": ("Deendayal (Kandla)", 22.75, 70.22, "India"),
    "INVAD": ("Vadinar", 22.47, 69.72, "India"),
    "INOKH": ("Okha", 22.47, 69.07, "India"),
    "INPBD": ("Porbandar", 21.64, 69.60, "India"),
    "INBOM": ("Mumbai", 18.94, 72.84, "India"),
    "INCOK": ("Kochi", 9.93, 76.27, "India"),
    "LKCMB": ("Colombo", 6.93, 79.84, "Sri Lanka"),
    "PKKHI": ("Karachi", 24.79, 66.98, "Pakistan"),
    "PKGWD": ("Gwadar", 25.12, 62.32, "Pakistan"),
    "OMSLL": ("Salalah", 16.94, 54.00, "Oman"),
    "OMMCT": ("Muscat", 23.62, 58.55, "Oman"),
}


def port_name(code):
    return PORTS[code][0]


# Routeing waypoints keep the tracks clear of land. Keyed by (from, to).
VIA = {
    ("AEJEA", "INMUN"): [(24.60, 60.00), (23.60, 64.80), (22.75, 68.30)],
    ("INMUN", "AEJEA"): [(22.75, 68.30), (23.60, 64.80), (24.60, 60.00)],
    ("AEFJR", "INVAD"): [(24.30, 60.50), (23.10, 65.20), (22.40, 68.20)],
    ("INVAD", "AEFJR"): [(22.40, 68.20), (23.10, 65.20), (24.30, 60.50)],
    ("AEJEA", "INVAD"): [(24.60, 60.00), (23.30, 65.00), (22.40, 68.20)],
    ("INVAD", "AEJEA"): [(22.40, 68.20), (23.30, 65.00), (24.60, 60.00)],
    ("INBOM", "LKCMB"): [(15.00, 72.50), (9.60, 74.20)],
    ("LKCMB", "INBOM"): [(9.60, 74.20), (15.00, 72.50)],
    ("INCOK", "INMUN"): [(12.50, 73.60), (17.00, 71.80), (21.00, 70.20)],
    ("INMUN", "INCOK"): [(21.00, 70.20), (17.00, 71.80), (12.50, 73.60)],
    ("LKCMB", "INIXY"): [(9.40, 74.60), (15.00, 72.40), (19.80, 70.60)],
    ("INIXY", "LKCMB"): [(19.80, 70.60), (15.00, 72.40), (9.40, 74.60)],
    ("LKCMB", "INMUN"): [(9.40, 74.60), (15.00, 72.40), (20.20, 70.40)],
    ("INMUN", "LKCMB"): [(20.20, 70.40), (15.00, 72.40), (9.40, 74.60)],
    ("LKCMB", "INCOK"): [(7.60, 76.60)],
    ("INCOK", "LKCMB"): [(7.60, 76.60)],
    ("INMUN", "PKKHI"): [(23.30, 68.40)],
    ("PKKHI", "INMUN"): [(23.30, 68.40)],
    ("INMUN", "INBOM"): [(20.60, 71.30)],
    ("INBOM", "INMUN"): [(20.60, 71.30)],
    ("INIXY", "INBOM"): [(20.60, 71.30)],
    ("INBOM", "INIXY"): [(20.60, 71.30)],
    ("INIXY", "INMUN"): [(22.80, 69.90)],
    ("INMUN", "INIXY"): [(22.80, 69.90)],
    ("INIXY", "AEJEA"): [(22.60, 68.30), (23.60, 64.80), (24.60, 60.00)],
    ("AEJEA", "INIXY"): [(24.60, 60.00), (23.60, 64.80), (22.60, 68.30)],
    ("INIXY", "AEFJR"): [(22.60, 68.30), (23.20, 65.00), (24.30, 60.50)],
    ("AEFJR", "INIXY"): [(24.30, 60.50), (23.20, 65.00), (22.60, 68.30)],
}

# Exercise and hazard areas: name, south, north, west, east.
AREAS = {
    "ALPHA": ("Area Alpha, North Arabian Sea", 21.90, 22.80, 68.40, 69.40),
    "BRAVO": ("Area Bravo, approaches to Mundra", 22.20, 22.60, 69.50, 69.90),
    "CHARLIE": ("Area Charlie, south-west of Porbandar", 20.80, 21.20, 69.60, 70.00),
    "DELTA": ("Area Delta, off Mumbai", 19.50, 20.00, 71.50, 72.00),
}


def area_text(key):
    name, south, north, west, east = AREAS[key]
    return (
        f"{name}, bounded by {dm(south, 'N')} {dm(west, 'E')}, "
        f"{dm(north, 'N')} {dm(west, 'E')}, {dm(north, 'N')} {dm(east, 'E')}, "
        f"{dm(south, 'N')} {dm(east, 'E')}"
    )


def in_box(lat, lon, key):
    _, south, north, west, east = AREAS[key]
    return south <= lat <= north and west <= lon <= east


REF_POINTS = [(name, lat, lon) for name, lat, lon, _ in PORTS.values()]
REF_POINTS += [
    ("Okha, India", 22.47, 69.07),
    ("Diu Head, India", 20.70, 70.98),
    ("Minicoy, India", 8.28, 73.05),
    ("Gwadar, Pakistan", 25.12, 62.32),
    ("Ras Al Hadd, Oman", 22.53, 59.80),
]


def describe_position(lat, lon):
    name, rlat, rlon = min(REF_POINTS, key=lambda r: haversine_nm(lat, lon, r[1], r[2]))
    dist = haversine_nm(lat, lon, rlat, rlon)
    brg = bearing_deg(rlat, rlon, lat, lon)
    return f"{dist:.0f} nm {compass(brg)} of {name}"


# --------------------------------------------------------------------------
# vessels
# --------------------------------------------------------------------------

VESSELS = [
    {
        "mmsi": "470123456",
        "imo": "9451234",
        "name": "MV AL RAYYAN STAR",
        "callsign": "A6Q3M",
        "flag": "United Arab Emirates",
        "type": "General cargo ship",
        "gt": 8942,
        "dwt": 12450,
        "loa": 128.4,
        "beam": 19.8,
        "summer_draught": 7.9,
        "built": 2011,
        "registry": "Dubai",
        "owner": "Rayyan Marine Shipping LLC",
        "manager": "Gulf Star Ship Management FZE",
        "class_society": "Meridian Classification Society",
        "crew": 18,
        "service_speed": 13.0,
        "hull": "dark blue hull, white superstructure, two deck cranes",
        "notes": (
            "Geared general cargo ship on a regular Gulf to west India run, calling at "
            "Jebel Ali, Mundra, Deendayal, Mumbai and Colombo. Frequently used as the "
            "example hull in the exercise because her track is easy to follow."
        ),
    },
    {
        "mmsi": "419876543",
        "imo": "9567890",
        "name": "MT GULF PIONEER",
        "callsign": "AVGH2",
        "flag": "India",
        "type": "Crude oil tanker",
        "gt": 42180,
        "dwt": 74900,
        "loa": 228.6,
        "beam": 32.2,
        "summer_draught": 12.6,
        "built": 2014,
        "registry": "Mumbai",
        "owner": "Pioneer Tankers India Private Limited",
        "manager": "Pioneer Tankers India Private Limited",
        "class_society": "Meridian Classification Society",
        "crew": 24,
        "service_speed": 12.5,
        "hull": "black hull, red boot topping, white superstructure aft",
        "notes": (
            "Crude carrier on the Fujairah to Vadinar shuttle. Loaded draught about "
            "12.6 m westbound, ballast draught about 5.4 m eastbound."
        ),
    },
    {
        "mmsi": "353136000",
        "imo": "9312456",
        "name": "MV CORAL HORIZON",
        "callsign": "3FQR9",
        "flag": "Panama",
        "type": "Container ship",
        "gt": 28450,
        "dwt": 34200,
        "loa": 199.9,
        "beam": 30.2,
        "summer_draught": 11.2,
        "built": 2009,
        "registry": "Panama City",
        "owner": "Horizon Container Lines Pte Ltd",
        "manager": "Coral Feeders Ship Management",
        "class_society": "Meridian Classification Society",
        "crew": 21,
        "service_speed": 16.5,
        "hull": "grey hull, orange funnel with a white band",
        "notes": (
            "Feeder container ship running Colombo, Kochi, Mundra and Karachi. "
            "Carries about 1,200 TEU and is the fastest of the three tracked ships."
        ),
    },
]

SECONDARY_VESSELS = [
    {
        "name": "FV SAGAR SETU",
        "mmsi": "419555777",
        "imo": "not assigned",
        "callsign": "not assigned",
        "flag": "India",
        "type": "Fishing vessel (wooden trawler)",
        "gt": 96,
        "dwt": "not applicable",
        "loa": 24.0,
        "beam": 6.4,
        "built": 2004,
        "registry": "Okha",
        "owner": "Sagar Setu Fishing Cooperative",
        "notes": (
            "Wooden trawler working out of Okha, class B AIS only and switched off for "
            "long periods. Typical speed 4 to 6 knots. Often the source of the "
            "unidentified small craft reports off the Gujarat coast."
        ),
    },
    {
        "name": "DHOW AL NAJAH",
        "mmsi": "no AIS fitted",
        "imo": "not assigned",
        "callsign": "not assigned",
        "flag": "not established",
        "type": "Traditional wooden dhow",
        "gt": "not known",
        "dwt": "not known",
        "loa": 18.0,
        "beam": 5.5,
        "built": "not known",
        "registry": "not established",
        "owner": "not established",
        "notes": (
            "Single mast wooden dhow, dark timber hull, no AIS and no registration "
            "marks observed. Speed 5 to 7 knots. Recurring unidentified contact in "
            "the Gulf of Kutch approaches."
        ),
    },
    {
        "name": "MT SEA BRIGHT",
        "mmsi": "636019876",
        "imo": "9712345",
        "callsign": "D5RT7",
        "flag": "Liberia",
        "type": "Bulk carrier",
        "gt": 32100,
        "dwt": 58400,
        "loa": 190.0,
        "beam": 32.3,
        "built": 2016,
        "registry": "Monrovia",
        "owner": "Bright Ocean Bulk Carriers",
        "notes": "Bulk carrier calling Mundra with fertiliser and coal cargoes.",
    },
    {
        "name": "MV MALABAR EXPRESS",
        "mmsi": "563456789",
        "imo": "9623456",
        "callsign": "9VAB7",
        "flag": "Singapore",
        "type": "Container ship",
        "gt": 24900,
        "dwt": 29800,
        "loa": 185.0,
        "beam": 29.0,
        "built": 2012,
        "registry": "Singapore",
        "owner": "Malabar Express Line Pte Ltd",
        "notes": "Regional container ship on the Kochi to Mumbai coastal run.",
    },
    {
        "name": "MV OMAN TRADER",
        "mmsi": "461234567",
        "imo": "9388111",
        "callsign": "A4DJ2",
        "flag": "Oman",
        "type": "General cargo ship",
        "gt": 6120,
        "dwt": 8400,
        "loa": 112.0,
        "beam": 18.0,
        "built": 2008,
        "registry": "Muscat",
        "owner": "Oman Trader Shipping Company",
        "notes": "Small general cargo ship trading Muscat, Salalah and Mumbai.",
    },
    {
        "name": "FV CORAL PRIDE",
        "mmsi": "419222888",
        "imo": "not assigned",
        "callsign": "not assigned",
        "flag": "India",
        "type": "Fishing vessel (gillnetter)",
        "gt": 78,
        "dwt": "not applicable",
        "loa": 21.5,
        "beam": 5.9,
        "built": 2007,
        "registry": "Porbandar",
        "owner": "Coral Pride Boat Owners Association",
        "notes": "Gillnetter working the Porbandar and Diu grounds, class B AIS only.",
    },
    {
        "name": "SV BLUE MERIDIAN",
        "mmsi": "232004561",
        "imo": "not assigned",
        "callsign": "MZQK4",
        "flag": "United Kingdom",
        "type": "Sailing yacht",
        "gt": 34,
        "dwt": "not applicable",
        "loa": 16.2,
        "beam": 4.8,
        "built": 2018,
        "registry": "Portsmouth",
        "owner": "private owner, name withheld",
        "notes": "Private yacht on a west India passage, class B AIS, speed 5 to 8 knots.",
    },
    {
        "name": "COAST GUARD UNIT CG-221",
        "mmsi": "not published",
        "imo": "not applicable",
        "callsign": "CG221",
        "flag": "India",
        "type": "Patrol vessel",
        "gt": "not published",
        "dwt": "not applicable",
        "loa": 45.0,
        "beam": 7.6,
        "built": "not published",
        "registry": "not published",
        "owner": "Indian Coast Guard",
        "notes": (
            "Exercise participant. Does not transmit AIS while on task and is "
            "classified as friendly when identified by exercise means."
        ),
    },
]


# --------------------------------------------------------------------------
# voyages: (from, departure, to, planned speed, draught, cargo)
# --------------------------------------------------------------------------

LEGS = {
    "470123456": [
        ("AEJEA", "2026-01-04T0830Z", "INMUN", 13.0, 7.9, "9,840 t of steel coils and project cargo"),
        ("INMUN", "2026-01-12T0620Z", "INIXY", 11.5, 5.1, "1,120 t of containerised cargo"),
        ("INIXY", "2026-01-16T0910Z", "INBOM", 12.5, 5.1, "2,400 t of bagged fertiliser"),
        ("INBOM", "2026-01-23T1140Z", "LKCMB", 13.0, 7.4, "6,900 t of general cargo"),
        ("LKCMB", "2026-02-02T0550Z", "INIXY", 13.0, 6.8, "7,450 t of containerised cargo"),
        ("INIXY", "2026-02-10T0715Z", "INMUN", 11.0, 4.8, "ballast passage, no cargo"),
        ("INMUN", "2026-02-14T0930Z", "AEJEA", 13.0, 7.2, "8,100 t of bagged cargo"),
        ("AEJEA", "2026-02-24T0605Z", "INMUN", 13.0, 7.8, "10,250 t of general cargo"),
        ("INMUN", "2026-03-04T0840Z", "INCOK", 12.5, 7.6, "8,600 t of general cargo"),
        ("INCOK", "2026-03-11T1020Z", "LKCMB", 12.0, 5.2, "1,800 t of bagged cargo"),
        ("LKCMB", "2026-03-18T0630Z", "INMUN", 13.0, 7.5, "9,100 t of general cargo"),
        ("INMUN", "2026-03-27T0545Z", "AEJEA", 13.0, 7.7, "9,760 t of general cargo"),
        ("AEJEA", "2026-04-06T0730Z", "INMUN", 13.0, 7.8, "10,400 t of project cargo"),
        ("INMUN", "2026-04-15T0620Z", "INCOK", 12.5, 7.4, "8,050 t of general cargo"),
        ("INCOK", "2026-04-23T1010Z", "LKCMB", 12.0, 5.0, "1,600 t of bagged cargo"),
        ("LKCMB", "2026-04-30T0540Z", "INMUN", 13.0, 7.3, "8,900 t of general cargo"),
        ("INMUN", "2026-05-09T0810Z", "INBOM", 12.5, 7.2, "7,200 t of general cargo"),
        ("INBOM", "2026-05-16T0640Z", "INIXY", 12.0, 5.4, "2,100 t of bagged cargo"),
        ("INIXY", "2026-05-23T0720Z", "AEJEA", 13.0, 7.5, "9,300 t of general cargo"),
        ("AEJEA", "2026-06-02T0550Z", "INMUN", 13.0, 7.9, "10,600 t of general cargo"),
        ("INMUN", "2026-06-11T0830Z", "INCOK", 12.5, 7.5, "8,300 t of general cargo"),
        ("INCOK", "2026-06-19T0940Z", "LKCMB", 12.0, 5.1, "1,700 t of bagged cargo"),
        ("LKCMB", "2026-06-26T0610Z", "INMUN", 12.5, 7.2, "8,750 t of general cargo"),
        ("INMUN", "2026-07-06T0750Z", "INBOM", 12.5, 7.1, "7,050 t of general cargo"),
        ("INBOM", "2026-07-14T0620Z", "INIXY", 12.0, 5.3, "2,050 t of bagged cargo"),
        ("INIXY", "2026-07-21T0715Z", "AEJEA", 12.5, 7.4, "9,150 t of general cargo"),
        ("AEJEA", "2026-07-30T0600Z", "INMUN", 13.0, 7.8, "10,300 t of general cargo"),
        ("INMUN", "2026-08-08T0805Z", "INCOK", 12.5, 7.5, "8,400 t of general cargo"),
        ("INCOK", "2026-08-17T0930Z", "LKCMB", 12.0, 5.0, "1,650 t of bagged cargo"),
        ("LKCMB", "2026-08-25T0555Z", "INMUN", 13.0, 7.6, "9,400 t of general cargo"),
    ],
    "419876543": [
        ("AEFJR", "2026-01-05T0500Z", "INVAD", 12.5, 12.6, "62,400 t of crude oil"),
        ("INVAD", "2026-01-14T1120Z", "AEFJR", 13.0, 5.4, "ballast passage, no cargo"),
        ("AEFJR", "2026-01-24T0610Z", "INVAD", 12.5, 12.4, "61,800 t of crude oil"),
        ("INVAD", "2026-02-04T0845Z", "AEFJR", 13.0, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-02-14T0530Z", "INVAD", 12.5, 12.5, "62,100 t of crude oil"),
        ("INVAD", "2026-02-25T0725Z", "AEFJR", 13.0, 5.4, "ballast passage, no cargo"),
        ("AEFJR", "2026-03-05T0620Z", "INVAD", 12.5, 12.6, "62,650 t of crude oil"),
        ("INVAD", "2026-03-12T0800Z", "AEFJR", 13.0, 5.6, "ballast passage, no cargo"),
        ("AEFJR", "2026-03-21T0740Z", "INVAD", 12.5, 12.3, "61,500 t of crude oil"),
        ("INVAD", "2026-03-29T0905Z", "AEFJR", 13.0, 5.4, "ballast passage, no cargo"),
        ("AEFJR", "2026-04-08T0555Z", "INVAD", 12.5, 12.6, "62,700 t of crude oil"),
        ("INVAD", "2026-04-16T0820Z", "AEFJR", 13.0, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-04-24T0635Z", "INVAD", 12.5, 12.5, "62,000 t of crude oil"),
        ("INVAD", "2026-05-02T0750Z", "AEFJR", 13.0, 5.4, "ballast passage, no cargo"),
        ("AEFJR", "2026-05-10T0605Z", "INVAD", 12.5, 12.6, "62,550 t of crude oil"),
        ("INVAD", "2026-05-18T0840Z", "AEFJR", 12.5, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-05-26T0710Z", "INVAD", 12.5, 12.4, "61,900 t of crude oil"),
        ("INVAD", "2026-06-03T0815Z", "AEFJR", 12.5, 5.6, "ballast passage, no cargo"),
        ("AEFJR", "2026-06-11T0640Z", "INVAD", 12.0, 12.5, "62,200 t of crude oil"),
        ("INVAD", "2026-06-19T0900Z", "AEFJR", 12.0, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-06-27T0725Z", "INVAD", 12.0, 12.6, "62,400 t of crude oil"),
        ("INVAD", "2026-07-05T0830Z", "AEFJR", 12.0, 5.6, "ballast passage, no cargo"),
        ("AEFJR", "2026-07-13T0700Z", "INVAD", 12.0, 12.4, "61,700 t of crude oil"),
        ("INVAD", "2026-07-21T0855Z", "AEFJR", 12.0, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-07-29T0735Z", "INVAD", 12.0, 12.5, "62,050 t of crude oil"),
        ("INVAD", "2026-08-06T0810Z", "AEFJR", 12.0, 5.6, "ballast passage, no cargo"),
        ("AEFJR", "2026-08-14T0650Z", "INVAD", 12.0, 12.6, "62,600 t of crude oil"),
        ("INVAD", "2026-08-22T0845Z", "AEFJR", 12.0, 5.5, "ballast passage, no cargo"),
        ("AEFJR", "2026-08-30T0720Z", "INVAD", 12.0, 12.5, "62,150 t of crude oil"),
    ],
    "353136000": [
        ("LKCMB", "2026-01-03T0700Z", "INCOK", 15.0, 10.8, "740 TEU of containerised cargo"),
        ("INCOK", "2026-01-08T0940Z", "INMUN", 16.5, 11.0, "1,120 TEU of containerised cargo"),
        ("INMUN", "2026-01-15T0615Z", "PKKHI", 15.5, 9.4, "620 TEU of containerised cargo"),
        ("PKKHI", "2026-01-22T0830Z", "INMUN", 15.5, 10.6, "880 TEU of containerised cargo"),
        ("INMUN", "2026-01-28T0745Z", "INCOK", 16.5, 10.9, "1,050 TEU of containerised cargo"),
        ("INCOK", "2026-02-04T1020Z", "LKCMB", 15.0, 9.2, "560 TEU of containerised cargo"),
        ("LKCMB", "2026-02-11T0650Z", "INCOK", 15.0, 10.7, "720 TEU of containerised cargo"),
        ("INCOK", "2026-02-16T0810Z", "INMUN", 16.5, 11.1, "1,180 TEU of containerised cargo"),
        ("INMUN", "2026-02-23T0730Z", "PKKHI", 15.5, 9.5, "640 TEU of containerised cargo"),
        ("PKKHI", "2026-03-01T0900Z", "INMUN", 15.5, 10.5, "860 TEU of containerised cargo"),
        ("INMUN", "2026-03-06T0625Z", "INCOK", 16.5, 10.8, "1,020 TEU of containerised cargo"),
        ("INCOK", "2026-03-13T0850Z", "LKCMB", 15.0, 9.3, "580 TEU of containerised cargo"),
        ("LKCMB", "2026-03-20T0715Z", "INCOK", 15.0, 10.6, "700 TEU of containerised cargo"),
        ("INCOK", "2026-03-25T0935Z", "INMUN", 16.5, 11.0, "1,140 TEU of containerised cargo"),
        ("INMUN", "2026-04-01T0655Z", "PKKHI", 15.5, 9.4, "610 TEU of containerised cargo"),
        ("PKKHI", "2026-04-07T0805Z", "INMUN", 15.5, 10.7, "900 TEU of containerised cargo"),
        ("INMUN", "2026-04-13T0720Z", "INCOK", 16.5, 10.9, "1,080 TEU of containerised cargo"),
        ("INCOK", "2026-04-20T0940Z", "LKCMB", 15.0, 9.2, "570 TEU of containerised cargo"),
        ("LKCMB", "2026-04-27T0650Z", "INCOK", 15.0, 10.8, "760 TEU of containerised cargo"),
        ("INCOK", "2026-05-02T0830Z", "INMUN", 16.5, 11.0, "1,160 TEU of containerised cargo"),
        ("INMUN", "2026-05-09T0740Z", "PKKHI", 15.5, 9.5, "630 TEU of containerised cargo"),
        ("PKKHI", "2026-05-16T0855Z", "INMUN", 15.5, 10.6, "870 TEU of containerised cargo"),
        ("INMUN", "2026-05-22T0705Z", "INCOK", 16.5, 10.8, "1,040 TEU of containerised cargo"),
        ("INCOK", "2026-05-29T0920Z", "LKCMB", 15.0, 9.3, "590 TEU of containerised cargo"),
        ("LKCMB", "2026-06-05T0710Z", "INCOK", 15.0, 10.7, "730 TEU of containerised cargo"),
        ("INCOK", "2026-06-10T0845Z", "INMUN", 16.0, 11.0, "1,120 TEU of containerised cargo"),
        ("INMUN", "2026-06-17T0725Z", "PKKHI", 15.0, 9.4, "600 TEU of containerised cargo"),
        ("PKKHI", "2026-06-24T0900Z", "INMUN", 15.0, 10.5, "850 TEU of containerised cargo"),
        ("INMUN", "2026-06-30T0650Z", "INCOK", 16.0, 10.9, "1,060 TEU of containerised cargo"),
        ("INCOK", "2026-07-08T0930Z", "LKCMB", 15.0, 9.2, "550 TEU of containerised cargo"),
        ("LKCMB", "2026-07-15T0700Z", "INCOK", 15.0, 10.6, "710 TEU of containerised cargo"),
        ("INCOK", "2026-07-20T0820Z", "INMUN", 16.0, 11.0, "1,150 TEU of containerised cargo"),
        ("INMUN", "2026-07-27T0745Z", "PKKHI", 15.0, 9.5, "650 TEU of containerised cargo"),
        ("PKKHI", "2026-08-03T0915Z", "INMUN", 15.0, 10.7, "890 TEU of containerised cargo"),
        ("INMUN", "2026-08-08T0700Z", "INCOK", 16.0, 10.8, "1,030 TEU of containerised cargo"),
        ("INCOK", "2026-08-15T0940Z", "LKCMB", 15.0, 9.3, "600 TEU of containerised cargo"),
        ("LKCMB", "2026-08-22T0720Z", "INCOK", 15.0, 10.7, "750 TEU of containerised cargo"),
        ("INCOK", "2026-08-27T0850Z", "INMUN", 16.0, 11.1, "1,190 TEU of containerised cargo"),
    ],
}


# --------------------------------------------------------------------------
# navigational warnings, coastal warnings and notices to mariners
# --------------------------------------------------------------------------

HAZARD_TEXT = {
    "live firing": (
        "Naval units will conduct live firing exercises, including surface gunnery, "
        "inside the area during the stated periods.",
        "Vessels are to remain clear of the area at all times. Any vessel unable to "
        "comply is to contact exercise control on VHF channel 16.",
    ),
    "survey operations": (
        "A survey vessel will conduct seabed mapping on a restricted manoeuvrability "
        "basis, with towed equipment astern.",
        "Vessels are to give a wide berth, pass at not less than 3 nm, and avoid "
        "passing within 2 nm astern of the survey vessel.",
    ),
    "aid to navigation": (
        "The aid to navigation named in this warning is off station or is not "
        "exhibiting its charted characteristic.",
        "Vessels are to navigate with caution, use charted soundings, and report the "
        "actual condition of the aid to the nearest coast radio station.",
    ),
    "chart correction": (
        "A chart correction has been issued for the area named in this notice. "
        "Depths, lights or channel limits differ from the current edition.",
        "Correct the affected chart and publications before the next transit. Do not "
        "rely on the superseded edition.",
    ),
    "tanker transfer": (
        "Two tankers will conduct a ship to ship transfer of crude oil at anchor "
        "inside the area.",
        "Vessels are to keep clear of the transfer pair by not less than 2 nm and are "
        "not to pass between the two ships.",
    ),
    "drifting hazard": (
        "A drifting object, partly submerged and difficult to detect by radar, has "
        "been reported in the area.",
        "Vessels are to keep a sharp lookout, reduce speed in poor visibility, and "
        "report any sighting to the nearest coast radio station.",
    ),
    "dredging": (
        "A trailing suction hopper dredger is working in the area with submerged "
        "pipeline and anchor buoys laid out.",
        "Vessels are to reduce speed to 8 knots, pass clear of the dredger and its "
        "buoys, and follow VTS instructions.",
    ),
    "navigation interference": (
        "Merchant vessels have reported intermittent loss of GNSS position fixing and "
        "degraded accuracy in the area. Interference may also affect AIS reporting.",
        "Vessels are to maintain a paper plot, use alternative position fixing, and "
        "report interference with position, time and details to the nearest coast "
        "radio station.",
    ),
    "cable operations": (
        "A cable ship is laying or repairing submarine cable in the area and will be "
        "restricted in her ability to manoeuvre.",
        "Vessels are to keep clear by not less than 2 nm and are not to anchor, trawl "
        "or dredge inside the area.",
    ),
    "launch debris": (
        "Debris from a space launch vehicle may fall inside the area during the stated "
        "window.",
        "Vessels are to remain clear of the area during the window and to report any "
        "debris sighted after the window closes.",
    ),
    "port restriction": (
        "A port entry restriction is in force for the port named in this warning.",
        "Vessels are to comply with the restriction, confirm the current status with "
        "port control before arrival, and adjust ETA as directed.",
    ),
    "fishing gear": (
        "A high concentration of fishing vessels and deployed gear is present in the "
        "area. Gear may be unlit at night.",
        "Vessels are to reduce speed, keep a sharp lookout for small craft and gear, "
        "and avoid passing through the concentration where practicable.",
    ),
    "weather": (
        "Heavy weather is expected in the area with rough to very rough seas and "
        "reduced visibility in rain squalls.",
        "Vessels are to secure deck cargo, review weather routeing, and report any "
        "difficulty to the nearest coast radio station.",
    ),
    "exercise": (
        "An exercise will be conducted in the area involving multiple surface units "
        "and aircraft.",
        "Vessels are to keep clear of the area and to comply with any instructions "
        "given by the exercise control authority.",
    ),
}

WARNING_OVERRIDES = {
    "012/2026": (
        "The port hand buoy marking the Okha approach channel has drifted about 1.2 nm "
        "to the south-east of its charted position and its light is unreliable.",
        "Vessels entering Okha are to navigate with caution, use charted soundings "
        "rather than the buoy position, and report the buoy's actual position.",
    ),
    "01/2026": (
        "The light at Vadinar has been re-established with a changed character, quick "
        "flashing green every 5 seconds, range 12 nm.",
        "Amend the light list entry and correct the chart before the next approach to "
        "Vadinar.",
    ),
    "059/2026": (
        "The starboard hand buoy in the Mundra approach channel is unlit and its "
        "reflector is damaged.",
        "Vessels are to treat the buoy as unlit, use radar and charted soundings for "
        "the approach, and report any change to port control.",
    ),
}

WARNINGS = [
    ("NAVAREA IX", "006/2026", "Live firing exercise in Area Alpha", "2026-01-08T0600Z", "2026-01-16T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "009/2026", "Seabed survey operations in Area Bravo", "2026-01-08T0500Z", "2026-01-28T1700Z", "BRAVO", "survey operations"),
    ("COASTAL WARNING", "012/2026", "Buoy off station, Okha approach", "2026-01-05T0900Z", "2026-01-22T1200Z", None, "aid to navigation"),
    ("NOTICE TO MARINERS", "01/2026", "Chart correction, Vadinar light", "2026-01-02T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "015/2026", "Tanker transfer operations in Area Delta", "2026-01-20T0400Z", "2026-01-24T2000Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "018/2026", "Drifting containers, south-west of Porbandar", "2026-01-16T1100Z", "2026-01-30T1800Z", "CHARLIE", "drifting hazard"),
    ("COASTAL WARNING", "021/2026", "Dredging, Deendayal approach channel", "2026-01-06T0600Z", "2026-02-28T1800Z", None, "dredging"),
    ("NAVAREA IX", "024/2026", "GNSS interference reported, north-east Arabian Sea", "2026-01-25T0000Z", "2026-02-05T2359Z", None, "navigation interference"),
    ("NAVAREA IX", "031/2026", "Live firing exercise in Area Alpha", "2026-02-09T0600Z", "2026-02-13T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "034/2026", "Cable laying operations in Area Charlie", "2026-02-05T0500Z", "2026-02-25T1700Z", "CHARLIE", "cable operations"),
    ("COASTAL WARNING", "038/2026", "Fishing gear concentration, off Okha", "2026-02-10T0000Z", "2026-03-05T2359Z", None, "fishing gear"),
    ("NOTICE TO MARINERS", "03/2026", "Wreck marked, south-west of Diu", "2026-02-14T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "041/2026", "Tanker transfer operations in Area Delta", "2026-02-16T0400Z", "2026-02-20T2000Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "045/2026", "Space launch vehicle debris, North Arabian Sea", "2026-02-21T0300Z", "2026-02-27T1800Z", "ALPHA", "launch debris"),
    ("COASTAL WARNING", "048/2026", "Night entry restriction, Mundra", "2026-02-18T0000Z", "2026-03-06T2359Z", None, "port restriction"),
    ("NAVAREA IX", "050/2026", "Unlit derelict, Gulf of Kutch approaches", "2026-02-24T0900Z", "2026-03-10T1800Z", None, "drifting hazard"),
    ("NAVAREA IX", "052/2026", "Live firing exercise in Area Alpha", "2026-03-08T0600Z", "2026-03-15T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "056/2026", "Seabed survey operations in Area Bravo", "2026-03-03T0500Z", "2026-03-20T1700Z", "BRAVO", "survey operations"),
    ("COASTAL WARNING", "059/2026", "Unlit buoy, Mundra approach", "2026-03-02T0800Z", "2026-03-18T1200Z", None, "aid to navigation"),
    ("NOTICE TO MARINERS", "05/2026", "Chart correction, Deendayal channel depths", "2026-03-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "063/2026", "Tanker transfer operations in Area Delta", "2026-03-22T0400Z", "2026-03-26T2000Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "067/2026", "GNSS interference reported, north-east Arabian Sea", "2026-03-18T0000Z", "2026-03-29T2359Z", None, "navigation interference"),
    ("COASTAL WARNING", "070/2026", "Anchoring restriction, Vadinar offshore", "2026-03-12T0000Z", "2026-04-02T2359Z", None, "port restriction"),
    ("NAVAREA IX", "072/2026", "Live firing exercise in Area Alpha", "2026-03-27T0600Z", "2026-03-29T1800Z", "ALPHA", "live firing"),
    ("COASTAL WARNING", "073/2026", "Dredger crossing, Deendayal to Mundra", "2026-03-24T0600Z", "2026-04-10T1800Z", None, "dredging"),
    ("NAVAREA IX", "074/2026", "Live firing exercise in Area Alpha", "2026-04-07T0600Z", "2026-04-10T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "078/2026", "Seabed survey operations in Area Bravo", "2026-04-05T0500Z", "2026-04-22T1700Z", "BRAVO", "survey operations"),
    ("COASTAL WARNING", "082/2026", "Buoy off station, Porbandar approach", "2026-04-03T0900Z", "2026-04-19T1200Z", None, "aid to navigation"),
    ("NOTICE TO MARINERS", "07/2026", "Chart correction, Okha to Dwarka", "2026-04-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "086/2026", "Tanker transfer operations in Area Delta", "2026-04-19T0400Z", "2026-04-23T2000Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "090/2026", "Oil spill response exercise in Area Charlie", "2026-04-24T0500Z", "2026-04-28T1700Z", "CHARLIE", "exercise"),
    ("COASTAL WARNING", "093/2026", "Fishing fleet concentration, off Porbandar", "2026-04-14T0000Z", "2026-05-04T2359Z", None, "fishing gear"),
    ("NAVAREA IX", "095/2026", "GNSS interference reported, north-east Arabian Sea", "2026-04-26T0000Z", "2026-05-08T2359Z", None, "navigation interference"),
    ("NAVAREA IX", "097/2026", "Live firing exercise in Area Alpha", "2026-05-05T0600Z", "2026-05-08T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "101/2026", "Seabed survey operations in Area Bravo", "2026-05-04T0500Z", "2026-05-21T1700Z", "BRAVO", "survey operations"),
    ("NAVAREA IX", "105/2026", "Cable repair operations in Area Charlie", "2026-05-12T0500Z", "2026-05-26T1700Z", "CHARLIE", "cable operations"),
    ("NOTICE TO MARINERS", "09/2026", "Chart correction, Mundra approaches", "2026-05-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "109/2026", "Tanker transfer operations in Area Delta", "2026-05-17T0400Z", "2026-05-21T2000Z", "DELTA", "tanker transfer"),
    ("COASTAL WARNING", "112/2026", "Pre-monsoon heavy weather advisory, west India coast", "2026-05-25T0000Z", "2026-06-15T2359Z", None, "weather"),
    ("NAVAREA IX", "115/2026", "Drifting fishing gear, south-west of Porbandar", "2026-05-20T0900Z", "2026-06-05T1800Z", None, "drifting hazard"),
    ("COASTAL WARNING", "116/2026", "Port entry restriction, Mundra", "2026-05-28T0000Z", "2026-06-10T2359Z", None, "port restriction"),
    ("NAVAREA IX", "118/2026", "South-west monsoon advisory, North Arabian Sea", "2026-06-01T0000Z", "2026-09-15T2359Z", None, "weather"),
    ("NAVAREA IX", "121/2026", "Live firing exercise in Area Alpha", "2026-06-09T0600Z", "2026-06-12T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "125/2026", "Seabed survey operations in Area Bravo", "2026-06-02T0500Z", "2026-06-18T1700Z", "BRAVO", "survey operations"),
    ("NOTICE TO MARINERS", "11/2026", "Chart correction, Deendayal to Vadinar", "2026-06-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "129/2026", "Tanker transfer operations suspended in Area Delta", "2026-06-08T0400Z", "2026-06-30T2359Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "133/2026", "GNSS interference reported, north-east Arabian Sea", "2026-06-15T0000Z", "2026-06-27T2359Z", None, "navigation interference"),
    ("COASTAL WARNING", "136/2026", "Unlit buoy, Okha approach", "2026-06-10T0800Z", "2026-06-28T1200Z", None, "aid to navigation"),
    ("COASTAL WARNING", "139/2026", "Search and rescue exercise, Gulf of Kutch", "2026-06-22T0500Z", "2026-06-24T1700Z", None, "exercise"),
    ("NAVAREA IX", "141/2026", "Live firing exercise in Area Alpha", "2026-07-07T0600Z", "2026-07-10T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "145/2026", "Seabed survey operations in Area Bravo", "2026-07-06T0500Z", "2026-07-23T1700Z", "BRAVO", "survey operations"),
    ("COASTAL WARNING", "148/2026", "Heavy weather advisory, west India coast", "2026-07-04T0000Z", "2026-07-31T2359Z", None, "weather"),
    ("NOTICE TO MARINERS", "13/2026", "Chart correction, Vadinar to Okha", "2026-07-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "151/2026", "Tanker transfer operations in Area Delta", "2026-07-19T0400Z", "2026-07-23T2000Z", "DELTA", "tanker transfer"),
    ("NAVAREA IX", "153/2026", "Drifting object, south-west of Porbandar", "2026-07-14T0900Z", "2026-07-28T1800Z", None, "drifting hazard"),
    ("COASTAL WARNING", "156/2026", "Fishing fleet concentration, off Okha", "2026-07-20T0000Z", "2026-08-08T2359Z", None, "fishing gear"),
    ("NAVAREA IX", "158/2026", "GNSS interference reported, north-east Arabian Sea", "2026-07-25T0000Z", "2026-08-06T2359Z", None, "navigation interference"),
    ("NAVAREA IX", "161/2026", "Live firing exercise in Area Alpha", "2026-08-11T0600Z", "2026-08-14T1800Z", "ALPHA", "live firing"),
    ("NAVAREA IX", "165/2026", "Seabed survey operations in Area Bravo", "2026-08-04T0500Z", "2026-08-21T1700Z", "BRAVO", "survey operations"),
    ("NAVAREA IX", "169/2026", "Cable laying operations in Area Charlie", "2026-08-18T0500Z", "2026-09-02T1700Z", "CHARLIE", "cable operations"),
    ("NOTICE TO MARINERS", "15/2026", "Chart correction, Okha approach", "2026-08-01T0000Z", None, None, "chart correction"),
    ("NAVAREA IX", "171/2026", "Tanker transfer operations in Area Delta", "2026-08-20T0400Z", "2026-08-24T2000Z", "DELTA", "tanker transfer"),
    ("COASTAL WARNING", "173/2026", "Unlit derelict, Gulf of Kutch approaches", "2026-08-06T0900Z", "2026-08-22T1800Z", None, "drifting hazard"),
    ("NAVAREA IX", "175/2026", "GNSS interference reported, north-east Arabian Sea", "2026-08-12T0000Z", "2026-08-24T2359Z", None, "navigation interference"),
]


# --------------------------------------------------------------------------
# synthetic contact reports filed by exercise units
# --------------------------------------------------------------------------

HAND_CONTACTS = [
    dict(
        id="SCR-2026-002", dt="2026-01-06T0435Z", unit="Exercise Unit Alpha",
        lat=22.31, lon=69.05, designation="C-04", sensor="radar and visual",
        classification="Traditional wooden dhow", affiliation="unknown",
        confidence="probable", mmsi=None, vessel=None,
        actions="Challenged on VHF channel 16, no response. Reported to operations room. Maintained surveillance for 40 minutes.",
        remarks="Dark timber hull, single mast, no AIS transmission, speed 6 knots on course 075. Consistent with a dhow working the Kutch approaches.",
    ),
    dict(
        id="SCR-2026-005", dt="2026-01-19T1120Z", unit="Exercise Unit Bravo",
        lat=22.10, lon=68.90, designation="C-07", sensor="radar and visual",
        classification="Fishing vessel concentration", affiliation="neutral",
        confidence="confirmed", mmsi=None, vessel=None,
        actions="Area reported to operations room and passed to shipping report.",
        remarks="Eleven small craft working in a 6 nm box, most with class B AIS only, several with no AIS. Gear deployed and partly unlit.",
    ),
    dict(
        id="SCR-2026-009", dt="2026-02-02T0815Z", unit="Exercise Unit Alpha",
        lat=21.62, lon=69.31, designation="C-09", sensor="radar, visual and AIS",
        classification="Fast boat, suspect approach", affiliation="suspect",
        confidence="probable", mmsi=None, vessel=None,
        actions="Escalated to operations room. Merchant vessel advised to increase speed and alter course away. Boat broke off after 12 minutes.",
        remarks="Skiff about 12 m, two outboard engines, speed 24 knots, no AIS, no registration marks. Closed to 1.5 nm on the port bow of a bulk carrier before turning away.",
    ),
    dict(
        id="SCR-2026-011", dt="2026-02-14T0742Z", unit="Exercise Unit Alpha",
        lat=21.53, lon=69.24, designation="C-12", sensor="radar and visual",
        classification="Traditional wooden dhow", affiliation="unknown",
        confidence="probable", mmsi=None, vessel="DHOW AL NAJAH (tentative)",
        actions="Challenged on VHF channel 16, no response. Reported to operations room. Photographed and tracked for 55 minutes.",
        remarks="Wooden hull, single mast, no AIS, no registration marks, speed 6 knots on course 090. Same profile as the contact reported on 6 January.",
    ),
    dict(
        id="SCR-2026-014", dt="2026-02-27T1930Z", unit="Exercise Unit Charlie",
        lat=22.28, lon=69.12, designation="C-15", sensor="radar only",
        classification="Small craft, type unknown", affiliation="unknown",
        confidence="possible", mmsi=None, vessel=None,
        actions="Tracked for 20 minutes, contact lost in sea clutter. Reported to operations room.",
        remarks="Unlit contact at night, low radar cross section, no AIS, estimated speed 7 knots. No visual identification possible.",
    ),
    dict(
        id="SCR-2026-017", dt="2026-03-05T0620Z", unit="Exercise Unit Bravo",
        lat=22.41, lon=69.68, designation="C-18", sensor="AIS, radar and visual",
        classification="Survey vessel", affiliation="neutral",
        confidence="confirmed", mmsi=None, vessel=None,
        actions="No action. Position passed to operations room for warning cross-check.",
        remarks="Vessel conducting seabed survey in Area Bravo on a restricted manoeuvrability basis with towed equipment astern. Matches the survey warning in force.",
    ),
    dict(
        id="SCR-2026-021", dt="2026-03-12T1045Z", unit="Exercise Unit Alpha",
        lat=22.42, lon=69.05, designation="C-21", sensor="AIS and radar",
        classification="Merchant tanker", affiliation="neutral",
        confidence="confirmed", mmsi="419876543", vessel="MT GULF PIONEER",
        actions="Position cross-checked against live firing warning. Vessel warned on VHF channel 16 and advised to clear the area to the south.",
        remarks="Vessel was inside Area Alpha while NAVAREA IX 052/2026 live firing warning was in force. AIS and radar tracks agreed. Ballast passage, draught 5.6 m, course 265, speed 13 knots.",
    ),
    dict(
        id="SCR-2026-024", dt="2026-03-18T1405Z", unit="Exercise Unit Charlie",
        lat=22.50, lon=68.80, designation="C-24", sensor="radar, visual and ESM",
        classification="Warship", affiliation="friendly",
        confidence="confirmed", mmsi=None, vessel="exercise participant",
        actions="Identified by exercise means and classified friendly. No action required.",
        remarks="Grey hull, warship profile, no AIS transmission, speed 18 knots. Identified as an exercise participant by ESM and challenge procedure.",
    ),
    dict(
        id="SCR-2026-028", dt="2026-03-27T0915Z", unit="Exercise Unit Alpha",
        lat=22.68, lon=68.95, designation="C-27", sensor="AIS and radar",
        classification="Merchant general cargo ship", affiliation="neutral",
        confidence="confirmed", mmsi="470123456", vessel="MV AL RAYYAN STAR",
        actions="Position cross-checked against live firing warning. Vessel informed of the warning by VHF and continued on passage.",
        remarks="Vessel passed through Area Alpha on an outbound passage from Mundra to Jebel Ali. NAVAREA IX 072/2026 live firing warning had come into force at 0600Z the same morning.",
    ),
    dict(
        id="SCR-2026-032", dt="2026-04-09T0910Z", unit="Exercise Unit Bravo",
        lat=21.10, lon=70.05, designation="C-31", sensor="radar and visual",
        classification="Helicopter", affiliation="unknown",
        confidence="probable", mmsi=None, vessel=None,
        actions="Reported to operations room. No action taken.",
        remarks="Single rotor helicopter, low level, transiting south-west, no identification marks visible. Not matched to any scheduled movement.",
    ),
    dict(
        id="SCR-2026-036", dt="2026-04-22T0300Z", unit="Exercise Unit Alpha",
        lat=22.75, lon=69.55, designation="C-35", sensor="radar, visual and AIS",
        classification="Fast boat, suspect approach", affiliation="suspect",
        confidence="probable", mmsi=None, vessel="MV MALABAR EXPRESS (target vessel)",
        actions="Escalated to operations room. Coast guard unit tasked. Boat departed to the north-east before interception.",
        remarks="Skiff closed to 1.8 nm on the stern of MV MALABAR EXPRESS at 0205Z. Speed 22 knots, no AIS, two persons visible in the dark. Target vessel reported the approach on VHF channel 16.",
    ),
    dict(
        id="SCR-2026-041", dt="2026-05-07T1215Z", unit="Exercise Unit Charlie",
        lat=20.94, lon=69.82, designation="C-39", sensor="radar and visual",
        classification="Drifting container", affiliation="not applicable",
        confidence="confirmed", mmsi=None, vessel=None,
        actions="Position passed to operations room and to the next of the shipping report. Marked as a navigation hazard.",
        remarks="Two containers awash, low freeboard, matching the drifting containers warning issued in January. Hazard to small craft and fishing gear.",
    ),
    dict(
        id="SCR-2026-045", dt="2026-05-24T1650Z", unit="Exercise Unit Bravo",
        lat=21.88, lon=69.42, designation="C-43", sensor="visual and AIS",
        classification="Fishing vessel", affiliation="neutral",
        confidence="confirmed", mmsi="419555777", vessel="FV SAGAR SETU",
        actions="No action. Vessel advised of the fishing gear concentration warning in force.",
        remarks="Wooden trawler working out of Okha, class B AIS only, speed 4.5 knots, gear deployed. Home port displayed on the hull.",
    ),
    dict(
        id="SCR-2026-049", dt="2026-06-11T0740Z", unit="Exercise Unit Alpha",
        lat=22.20, lon=69.35, designation="C-47", sensor="radar only",
        classification="Small craft, type unknown", affiliation="unknown",
        confidence="possible", mmsi=None, vessel=None,
        actions="Contact lost in monsoon rain clutter. Reported to operations room.",
        remarks="Radar contact held for 8 minutes in heavy rain, no AIS, no visual. Monsoon conditions degraded detection range.",
    ),
    dict(
        id="SCR-2026-053", dt="2026-06-25T1030Z", unit="Exercise Unit Charlie",
        lat=22.05, lon=69.48, designation="C-51", sensor="radar and visual",
        classification="Traditional wooden dhow", affiliation="unknown",
        confidence="probable", mmsi=None, vessel="DHOW AL NAJAH (tentative)",
        actions="Challenged on VHF channel 16, no response. Reported to operations room.",
        remarks="Same profile as the dhow reported in February. No AIS, no registration marks, speed 5 knots, course 130.",
    ),
    dict(
        id="SCR-2026-058", dt="2026-07-08T0820Z", unit="Exercise Unit Bravo",
        lat=21.22, lon=69.61, designation="C-56", sensor="visual and radar",
        classification="Sub-surface contact, possible periscope", affiliation="unknown",
        confidence="possible", mmsi=None, vessel=None,
        actions="Escalated to operations room. Exercise unit manoeuvred to maintain contact. Contact classified friendly after exercise means were applied.",
        remarks="Periscope-like visual sighting and a weak radar return with no AIS. Exercise participant confirmed by challenge procedure 26 minutes later.",
    ),
    dict(
        id="SCR-2026-062", dt="2026-07-21T1500Z", unit="Exercise Unit Alpha",
        lat=22.72, lon=68.58, designation="C-60", sensor="radar and visual",
        classification="Fast boat, suspect approach", affiliation="suspect",
        confidence="probable", mmsi=None, vessel=None,
        actions="Escalated to operations room. Boat turned away when challenged by radio and by warning signal.",
        remarks="Skiff about 11 m, twin outboards, speed 25 knots, no AIS, no marks. Closed to 2 nm of a passing general cargo ship, then withdrew north-west.",
    ),
    dict(
        id="SCR-2026-067", dt="2026-08-14T0640Z", unit="Exercise Unit Charlie",
        lat=21.94, lon=68.72, designation="C-64", sensor="radar and visual",
        classification="Fishing vessel concentration", affiliation="neutral",
        confidence="confirmed", mmsi=None, vessel=None,
        actions="Area passed to operations room and included in the shipping report.",
        remarks="Nine small craft working a 5 nm box, mixed class B AIS and no AIS, gear deployed, several vessels with unlit markers.",
    ),
    dict(
        id="SCR-2026-071", dt="2026-08-27T1130Z", unit="Exercise Unit Alpha",
        lat=22.60, lon=69.30, designation="C-68", sensor="AIS and radar",
        classification="Merchant vessel, AIS mismatch", affiliation="suspect",
        confidence="probable", mmsi=None, vessel=None,
        actions="Escalated to operations room. AIS identity queried through the regional reporting chain. Vessel tracked until it cleared the area.",
        remarks="AIS reported the name of a container ship but the position, course and dimensions did not match that vessel's known movements. Possible identity misuse or a faulty transponder configuration.",
    ),
]


# --------------------------------------------------------------------------
# monthly shipping reports
# --------------------------------------------------------------------------

MONTHLY = [
    dict(
        month="2026-01", transits=418, by_type={"tankers": 171, "container ships": 104, "bulk carriers": 92, "other": 51},
        waits={"Deendayal": 13.5, "Mundra": 8.2, "Vadinar": 19.4}, sts_ops=6,
        weather="North-east monsoon. Sea state 2 to 3, good visibility, no weather interruptions.",
        incidents=[
            "Two reports of suspicious approach by small craft off the Gujarat coast, neither resulting in boarding.",
            "One attempted boarding of a fishing vessel 30 nm south-west of Porbandar, repelled by the crew.",
        ],
        notable=[
            "Container volumes through Mundra up 6 per cent on December.",
            "Crude shuttle traffic to Vadinar steady at about six laden calls per month.",
        ],
        advisories=["Fishing gear concentration reported off Okha, keep a sharp lookout at night."],
        dark_activity="Three contacts with no AIS transmission were reported in the Gulf of Kutch approaches.",
    ),
    dict(
        month="2026-02", transits=431, by_type={"tankers": 178, "container ships": 108, "bulk carriers": 94, "other": 51},
        waits={"Deendayal": 16.8, "Mundra": 10.4, "Vadinar": 24.6}, sts_ops=7,
        weather="North-east monsoon weakening. Sea state 2 to 3, occasional morning haze.",
        incidents=[
            "Three reports of suspicious approach, one within 2 nm of a bulk carrier at anchor.",
            "One report of a vessel transmitting an AIS identity that did not match its physical description.",
        ],
        notable=[
            "Anchorage waiting time at Vadinar rose to 24.6 hours because of berth congestion.",
            "A dhow with no AIS was reported twice in the same week in the Kutch approaches.",
        ],
        advisories=["Verify AIS identity by radar and visual means where practicable."],
        dark_activity="Five contacts with no AIS transmission, including two repeated dhow sightings.",
    ),
    dict(
        month="2026-03", transits=442, by_type={"tankers": 184, "container ships": 112, "bulk carriers": 96, "other": 50},
        waits={"Deendayal": 21.3, "Mundra": 12.9, "Vadinar": 28.7}, sts_ops=8,
        weather="Fair, sea state 2, visibility good. First pre-monsoon haze in the second half of the month.",
        incidents=[
            "Four reports of suspicious approach, one by a fast skiff closing to 1.5 nm on a tanker.",
            "Two contacts transited a live firing area while a warning was in force and were warned by radio.",
        ],
        notable=[
            "Highest monthly transit count of the period so far at 442.",
            "Live firing warnings in Area Alpha were in force on four separate occasions.",
        ],
        advisories=["Cross-check planned tracks against live firing warnings before departure."],
        dark_activity="Six contacts with no AIS transmission, the highest of the first quarter.",
    ),
    dict(
        month="2026-04", transits=428, by_type={"tankers": 176, "container ships": 110, "bulk carriers": 92, "other": 50},
        waits={"Deendayal": 18.9, "Mundra": 11.2, "Vadinar": 25.1}, sts_ops=7,
        weather="Fair to moderate, sea state 2 to 3, increasing haze and heat in the afternoon.",
        incidents=[
            "Three reports of suspicious approach, including one against a container ship at 0300 local time.",
            "One report of drifting containers creating a hazard to small craft.",
        ],
        notable=[
            "Berth congestion eased slightly from the March peak.",
            "A coast guard unit was tasked twice in response to suspicious approaches.",
        ],
        advisories=["Report all approaches by small craft within 2 nm, even when no boarding occurs."],
        dark_activity="Four contacts with no AIS transmission.",
    ),
    dict(
        month="2026-05", transits=401, by_type={"tankers": 162, "container ships": 106, "bulk carriers": 86, "other": 47},
        waits={"Deendayal": 15.4, "Mundra": 9.8, "Vadinar": 21.3}, sts_ops=6,
        weather="Pre-monsoon. Sea state 3, increasing swell from the south-west in the last week.",
        incidents=[
            "Two reports of suspicious approach, both resolved without boarding.",
            "One drifting container group reported again, now about 40 nm from its original position.",
        ],
        notable=[
            "Transit numbers began to fall as owners adjusted schedules ahead of the monsoon.",
            "Pre-monsoon heavy weather advisory issued on 25 May.",
        ],
        advisories=["Review weather routeing and secure deck cargo before the monsoon sets in."],
        dark_activity="Three contacts with no AIS transmission.",
    ),
    dict(
        month="2026-06", transits=356, by_type={"tankers": 138, "container ships": 96, "bulk carriers": 78, "other": 44},
        waits={"Deendayal": 12.1, "Mundra": 8.6, "Vadinar": 17.2}, sts_ops=4,
        weather="South-west monsoon set in on 8 June. Sea state 4 to 5, heavy rain squalls, reduced radar detection range.",
        incidents=[
            "One report of a suspicious approach in poor visibility.",
            "One radar contact lost in rain clutter with no AIS and no visual identification.",
        ],
        notable=[
            "Transits fell 11 per cent from May, the sharpest monthly fall of the period.",
            "Ship to ship transfer operations at Area Delta were suspended for the month.",
        ],
        advisories=["Monsoon advisory in force. Reduce speed in heavy rain and maintain a paper plot."],
        dark_activity="Two contacts with no AIS transmission, both in reduced visibility.",
    ),
    dict(
        month="2026-07", transits=331, by_type={"tankers": 126, "container ships": 90, "bulk carriers": 72, "other": 43},
        waits={"Deendayal": 11.4, "Mundra": 7.9, "Vadinar": 15.8}, sts_ops=4,
        weather="Peak south-west monsoon. Sea state 5, swell 3 to 4 m, frequent heavy rain.",
        incidents=[
            "Two reports of suspicious approach, one by a fast skiff that withdrew when challenged.",
            "One possible sub-surface contact reported and later classified as friendly.",
        ],
        notable=[
            "Lowest monthly transit count of the period at 331.",
            "Fishing fleet activity reduced by weather but still concentrated off Okha.",
        ],
        advisories=["Heavy weather advisory in force for the whole month."],
        dark_activity="Two contacts with no AIS transmission.",
    ),
    dict(
        month="2026-08", transits=348, by_type={"tankers": 134, "container ships": 94, "bulk carriers": 76, "other": 44},
        waits={"Deendayal": 12.7, "Mundra": 8.4, "Vadinar": 16.9}, sts_ops=5,
        weather="Monsoon easing in the second half. Sea state 4 falling to 3, rain squalls becoming less frequent.",
        incidents=[
            "Three reports of suspicious approach, including one vessel transmitting an AIS identity that did not match its description.",
            "One fishing fleet concentration reported off Okha with unlit gear.",
        ],
        notable=[
            "Transit numbers recovering, up 5 per cent on July.",
            "AIS identity verification incidents rose, prompting a regional reporting note.",
        ],
        advisories=["Verify AIS identity against radar and visual observation, and report mismatches."],
        dark_activity="Three contacts with no AIS transmission, including one AIS identity mismatch.",
    ),
]

OVERARCHING = [
    dict(
        id="Q1-2026", title="North Arabian Sea shipping report, first quarter 2026",
        period="January to March 2026",
        summary=(
            "Traffic was strong through the first quarter, with 1,291 merchant transits recorded across "
            "the Gulf of Kutch and the north-east Arabian Sea. Anchorage waiting times rose steadily "
            "from January to March, peaking at Vadinar in March at 28.7 hours. Ten suspicious approach "
            "reports were filed, and fourteen contacts with no AIS transmission were recorded."
        ),
        incidents=(
            "Suspicious approaches by small craft were the dominant security theme, with most reported "
            "between dusk and dawn and most targeting vessels at anchor or on slow transit."
        ),
        watch_items=[
            "Repeated dhow sightings in the Kutch approaches with no AIS and no registration marks.",
            "AIS identity mismatches, two in the quarter.",
            "Live firing warnings in Area Alpha on four separate occasions.",
        ],
    ),
    dict(
        id="Q2-2026", title="North Arabian Sea shipping report, second quarter 2026",
        period="April to June 2026",
        summary=(
            "Transits fell from 428 in April to 356 in June as owners adjusted schedules ahead of the "
            "south-west monsoon. Anchorage waiting times eased with the falling volume. Eight suspicious "
            "approach reports were filed, and the first monsoon-related degradation of radar and visual "
            "detection was recorded in June."
        ),
        incidents=(
            "One approach against a container ship at 0300 local time was the most serious of the quarter. "
            "A coast guard unit was tasked and the skiff departed before interception."
        ),
        watch_items=[
            "Drifting containers from the January loss continued to be reported, moving north-east.",
            "Monsoon onset on 8 June sharply reduced detection ranges in rain.",
            "Ship to ship transfer operations were suspended in June.",
        ],
    ),
    dict(
        id="H1-2026", title="North Arabian Sea shipping report, first half of 2026",
        period="January to June 2026",
        summary=(
            "The first half of the year recorded 2,476 merchant transits. Traffic peaked in March at 442 "
            "and fell to 356 by June under the influence of the south-west monsoon. Eighteen suspicious "
            "approach reports and twenty-four contacts with no AIS transmission were recorded. No boarding "
            "or hijacking succeeded in the period."
        ),
        incidents=(
            "Security incidents were concentrated in the pre-monsoon months. Every reported approach "
            "involved a small, fast craft with no AIS, no registration marks and no response to VHF."
        ),
        watch_items=[
            "The unidentified dhow in the Kutch approaches was reported in January, February and June.",
            "Two vessels transited a live firing area in March while a warning was in force.",
            "Anchorage congestion at Vadinar was the main commercial friction point.",
        ],
    ),
    dict(
        id="MONSOON-2026", title="Monsoon season summary, North Arabian Sea",
        period="June to August 2026",
        summary=(
            "The south-west monsoon set in on 8 June and reduced transits from 401 in May to a low of 331 "
            "in July before a partial recovery to 348 in August. Sea state reached 5 with 3 to 4 m swell "
            "in July. Radar detection range in heavy rain fell by an estimated 30 to 40 per cent, and "
            "seven contacts with no AIS transmission were recorded across the three months."
        ),
        incidents=(
            "Suspicious approaches continued at a lower rate, and the identification of small contacts "
            "became materially harder in rain clutter."
        ),
        watch_items=[
            "Reduced detection range increases the risk from small craft with no AIS.",
            "AIS identity mismatches continued, including one in August.",
            "Fishing fleets remained concentrated off Okha with partly unlit gear.",
        ],
    ),
]


# --------------------------------------------------------------------------
# derived tracks
# --------------------------------------------------------------------------

VESSEL_BY_MMSI = {v["mmsi"]: v for v in VESSELS}
HAZARD_LABEL = {
    "live firing": "live firing",
    "survey operations": "survey operations",
    "aid to navigation": "an aid to navigation defect",
    "chart correction": "a chart correction",
    "tanker transfer": "ship to ship transfer operations",
    "drifting hazard": "a drifting hazard",
    "dredging": "dredging",
    "navigation interference": "reported GNSS interference",
    "cable operations": "cable operations",
    "launch debris": "possible launch vehicle debris",
    "port restriction": "a port entry restriction",
    "fishing gear": "a fishing gear concentration",
    "weather": "heavy weather",
    "exercise": "an exercise",
}


def path_of(frm, to):
    return [PORTS[frm][1:3]] + list(VIA.get((frm, to), [])) + [PORTS[to][1:3]]


def cum_distances(pts):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + haversine_nm(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]))
    return cum


def point_at(pts, cum, frac):
    target = frac * cum[-1]
    for i in range(1, len(cum)):
        if cum[i] >= target - 1e-9:
            seg = cum[i] - cum[i - 1]
            t = 0.0 if seg <= 0 else (target - cum[i - 1]) / seg
            lat = pts[i - 1][0] + t * (pts[i][0] - pts[i - 1][0])
            lon = pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1])
            return lat, lon, bearing_deg(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
    return pts[-1][0], pts[-1][1], 0.0


def build_legs():
    all_legs = {}
    for mmsi, rows in LEGS.items():
        out = []
        prev_arr = None
        for idx, (frm, dep_s, to, speed, draught, cargo) in enumerate(rows, start=1):
            dep = parse_dt(dep_s)
            pts = path_of(frm, to)
            cum = cum_distances(pts)
            distance = cum[-1]
            arr = dep + timedelta(hours=distance / speed)
            if prev_arr is not None and dep <= prev_arr:
                raise SystemExit(
                    f"voyage overlap on {mmsi} leg {idx}: departure {dep} precedes "
                    f"previous arrival {prev_arr}"
                )
            out.append({
                "index": idx, "frm": frm, "to": to, "dep": dep, "arr": arr,
                "pts": pts, "cum": cum, "distance": distance,
                "speed": speed, "draught": draught, "cargo": cargo,
            })
            prev_arr = arr
        all_legs[mmsi] = out
    return all_legs


LEGS_BY_MMSI = build_legs()


def position_at(mmsi, when):
    legs = LEGS_BY_MMSI[mmsi]
    if when < legs[0]["dep"]:
        return {"kind": "port", "port": legs[0]["frm"], "leg": None}
    for i, leg in enumerate(legs):
        if leg["dep"] <= when <= leg["arr"]:
            frac = (when - leg["dep"]).total_seconds() / (leg["arr"] - leg["dep"]).total_seconds()
            lat, lon, brg = point_at(leg["pts"], leg["cum"], frac)
            return {"kind": "sea", "lat": lat, "lon": lon, "course": brg,
                    "speed": leg["speed"], "leg": leg}
        if i + 1 < len(legs) and leg["arr"] < when < legs[i + 1]["dep"]:
            return {"kind": "port", "port": leg["to"], "leg": None}
    return {"kind": "port", "port": legs[-1]["to"], "leg": None}


def route_text(leg):
    if len(leg["pts"]) <= 2:
        return "on a direct track"
    points = ", ".join(f"{dm(a, 'N')} {dm(b, 'E')}" for a, b in leg["pts"][1:-1])
    return f"via {points}"


def warnings_active(when):
    out = []
    for kind, number, title, issued, cancelled, area, hazard in WARNINGS:
        start = parse_dt(issued)
        end = parse_dt(cancelled) if cancelled else datetime(2026, 12, 31)
        if start <= when <= end:
            out.append({
                "kind": kind, "number": number, "title": title, "area": area,
                "hazard": hazard, "start": start, "end": end, "cancelled": cancelled,
            })
    return out


def area_transits(area_key, step_minutes=15):
    found = []
    for mmsi, legs in LEGS_BY_MMSI.items():
        for leg in legs:
            inside = False
            start = None
            last = None
            step = timedelta(minutes=step_minutes)
            when = leg["dep"]
            span = (leg["arr"] - leg["dep"]).total_seconds()
            while when <= leg["arr"]:
                frac = (when - leg["dep"]).total_seconds() / span
                lat, lon, _ = point_at(leg["pts"], leg["cum"], frac)
                if in_box(lat, lon, area_key):
                    if not inside:
                        inside, start = True, when
                    last = when
                elif inside:
                    found.append((mmsi, start, last, leg))
                    inside = False
                when += step
            if inside:
                found.append((mmsi, start, last, leg))
    return found


def num(value, unit=""):
    if isinstance(value, int):
        return f"{value:,}{unit}"
    return str(value)


THE_FLAGS = {"United Arab Emirates", "United Kingdom", "Netherlands", "Philippines", "Bahamas"}


def flag_phrase(flag):
    return f"the {flag}" if flag in THE_FLAGS else flag


_NUM_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
    7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve",
}


def count_word(n):
    return _NUM_WORDS.get(n, str(n))


def lower_label(text):
    out = text.lower()
    for token in ("ais", "gnss", "vhf", "imo", "mmsi"):
        out = out.replace(token, token.upper())
    return out


class QABuilder:
    def __init__(self, prefix, topic, title):
        self.prefix = prefix
        self.topic = topic
        self.title = title
        self.items = []

    def add(self, question, answer, source, tags=()):
        self.items.append({
            "id": f"{self.prefix}-{len(self.items) + 1:04d}",
            "topic": self.topic,
            "question": question,
            "answer": answer,
            "source": source,
            "tags": list(tags),
        })

    def envelope(self):
        return {
            "dataset": self.topic,
            "title": self.title,
            "version": "1.0.0",
            "generated": GENERATED,
            "coverage": COVERAGE,
            "format": "question-answer",
            "synthetic": True,
            "not_for_operational_use": True,
            "disclaimer": DISCLAIMER,
            "count": len(self.items),
            "examples": self.items,
        }


def vessel_field(v, key, default="not recorded"):
    value = v.get(key, default)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# --------------------------------------------------------------------------
# file 1: domain vessel reports
# --------------------------------------------------------------------------


def build_domain_vessel_reports():
    b = QABuilder(
        "DVR", "domain_vessel_reports",
        "Vessel identity and registry dossiers for the exercise area",
    )
    everyone = VESSELS + SECONDARY_VESSELS
    for v in everyone:
        name = v["name"]
        src = f"Vessel registry dossier, {name}"
        b.add(
            f"What is the IMO number of {name}?",
            f"{name} carries IMO number {v['imo']} and MMSI {v['mmsi']}. She flies the flag of "
            f"{flag_phrase(v['flag'])} and is registered at {v['registry']}.",
            src, ["identity"],
        )
        b.add(
            f"Which vessel carries MMSI {v['mmsi']}?",
            f"MMSI {v['mmsi']} belongs to {name}, IMO {v['imo']}, a {v['type'].lower()} flagged in "
            f"{flag_phrase(v['flag'])}.",
            f"AIS static data, MMSI {v['mmsi']}", ["identity"],
        )
        b.add(
            f"What is the call sign of {name}?",
            f"The call sign of {name} is {v['callsign']}.",
            src, ["identity"],
        )
        b.add(
            f"What type of ship is {name}?",
            f"{name} is a {v['type'].lower()} flagged in {flag_phrase(v['flag'])}, {num(v.get('gt'), ' gross tonnage')} "
            f"and {num(v.get('dwt'), ' tonnes deadweight')}.",
            src, ["identity", "type"],
        )
        b.add(
            f"What are the dimensions of {name}?",
            f"{name} is {v['loa']} m length overall and {v['beam']} m beam, built in {v['built']}.",
            src, ["identity"],
        )
        b.add(
            f"Which flag does {name} fly and where is she registered?",
            f"{name} flies the flag of {flag_phrase(v['flag'])} and is registered at {v['registry']}.",
            src, ["identity", "flag"],
        )
        b.add(
            f"Who owns and manages {name}?",
            f"{name} is owned by {v['owner']}."
            + (f" She is managed by {v['manager']}." if v.get("manager") and v.get("manager") != v["owner"] else ""),
            src, ["ownership"],
        )
        b.add(
            f"Give me the full dossier on {name}.",
            f"{name}. IMO {v['imo']}, MMSI {v['mmsi']}, call sign {v['callsign']}. Flag {flag_phrase(v['flag'])}, "
            f"registered at {v['registry']}. Type {v['type'].lower()}, {num(v.get('gt'), ' GT')}, "
            f"{num(v.get('dwt'), ' DWT')}, {v['loa']} m by {v['beam']} m, built {v['built']}. "
            f"Owner {v['owner']}. {v.get('notes', '')}".strip(),
            src, ["dossier"],
        )
    b.add(
        "Which vessels in the exercise area are crude oil tankers?",
        "Only one crude oil tanker is tracked in the area: MT GULF PIONEER, MMSI 419876543, "
        "IMO 9567890, flagged in India and registered at Mumbai. She runs the Fujairah to Vadinar "
        "crude shuttle.",
        "Vessel registry dossier, type query", ["aggregate", "type"],
    )
    b.add(
        "Which tracked vessels are flagged in India?",
        "Two of the three tracked merchant vessels are Indian flagged: MT GULF PIONEER, MMSI "
        "419876543, a crude oil tanker registered at Mumbai, and the fishing vessel FV SAGAR SETU, "
        "MMSI 419555777, registered at Okha. MV AL RAYYAN STAR is flagged in the United Arab "
        "Emirates and MV CORAL HORIZON is flagged in Panama.",
        "Vessel registry dossier, flag query", ["aggregate", "flag"],
    )
    b.add(
        "Which tracked vessels are container ships?",
        "Two container ships are tracked: MV CORAL HORIZON, MMSI 353136000, flagged in Panama and "
        "running the Colombo, Kochi, Mundra and Karachi loop, and MV MALABAR EXPRESS, MMSI "
        "563456789, flagged in Singapore on the Kochi to Mumbai coastal run.",
        "Vessel registry dossier, type query", ["aggregate", "type"],
    )
    b.add(
        "Which vessels regularly call at Mundra?",
        "Mundra is called by MV AL RAYYAN STAR, MV CORAL HORIZON and MT SEA BRIGHT, and is the "
        "destination declared in the AIS of several transiting bulk carriers.",
        "Vessel registry dossier and AIS destination fields", ["aggregate", "port"],
    )
    b.add(
        "What do we know about the fishing vessel SAGAR SETU?",
        "FV SAGAR SETU is a wooden trawler of about 96 gross tonnage and 24 m length, MMSI "
        "419555777, registered at Okha and owned by the Sagar Setu Fishing Cooperative. She carries "
        "class B AIS only and switches it off for long periods, so she appears as a dark contact.",
        "Vessel registry dossier, FV SAGAR SETU", ["identity", "fishing"],
    )
    b.add(
        "What do we know about the dhow with no AIS in the Kutch approaches?",
        "The dhow, provisionally recorded as DHOW AL NAJAH, has no MMSI, no call sign and no "
        "established flag or registry. She is a single mast wooden dhow of about 18 m, dark timber "
        "hull, speed 5 to 7 knots, and has been reported three times without a response to VHF "
        "challenge. Treat her identity as unverified.",
        "Contact report file and vessel registry dossier", ["dark-contact"],
    )
    b.add(
        "A contact is detected on radar with no AIS. Which of our known vessels could it be?",
        "Four known units can appear without AIS. The wooden trawler FV SAGAR SETU and the "
        "gillnetter FV CORAL PRIDE carry class B AIS only and switch it off for long periods. The "
        "wooden dhow DHOW AL NAJAH has no AIS fitted at all. Coast guard unit CG-221 does not "
        "transmit AIS while on task. Any other dark contact should be treated as unidentified.",
        "Vessel registry dossier, dark contact query", ["dark-contact"],
    )
    b.add(
        "You sight a vessel on the horizon and read AL RAYYAN STAR on the hull, but she is not "
        "transmitting AIS. What do you know about her and what should you establish next?",
        "The hull name matches MV AL RAYYAN STAR, IMO 9451234, MMSI 470123456, a United Arab "
        "Emirates flagged general cargo ship of 8,942 GT and 12,450 DWT, 128.4 m long, built 2011, "
        "owned by Rayyan Marine Shipping LLC and managed by Gulf Star Ship Management. She "
        "normally trades Jebel Ali, Mundra, Deendayal, Mumbai and Colombo. Establish next: her "
        "current position and course by radar, her last and next port of call, whether a port state "
        "or VTS record exists for her in the last 72 hours, and whether any navigational warning is "
        "in force on her track. Do not assume the hull name is genuine until it is corroborated.",
        "Vessel registry dossier and AIS passage records, MV AL RAYYAN STAR",
        ["horizon-sighting", "story"],
    )
    b.add(
        "A hull shows the name GULF PIONEER. Which vessel is that and what is she carrying?",
        "The name matches MT GULF PIONEER, IMO 9567890, MMSI 419876543, a crude oil tanker of "
        "42,180 GT and 74,900 DWT flagged in India and registered at Mumbai, owned and managed by "
        "Pioneer Tankers India Private Limited. She runs the Fujairah to Vadinar crude shuttle, "
        "loaded westbound at about 12.6 m draught and in ballast eastbound at about 5.4 m. Confirm "
        "her draught, cargo and destination from AIS before assuming a laden passage.",
        "Vessel registry dossier and AIS passage records, MT GULF PIONEER",
        ["horizon-sighting", "story"],
    )
    return b


# --------------------------------------------------------------------------
# file 2: contact classification
# --------------------------------------------------------------------------

CLASSIFICATION_DOCTRINE = [
    (
        "What are the contact classification categories used in this exercise?",
        "Contacts are first classified by domain as surface, air, sub-surface or unknown, and then by "
        "type as warship, patrol vessel, merchant cargo ship, merchant tanker, container ship, bulk "
        "carrier, fishing vessel, traditional craft, pleasure craft, survey vessel, tug, pilot boat, "
        "drifting object or unknown.",
    ),
    (
        "What are the affiliation categories and what does each mean?",
        "Friendly means identified as own or allied forces. Neutral means an identified non-participant "
        "going about lawful business. Unknown means the identity could not be established. Suspect "
        "means the contact is behaving in a way that is inconsistent with lawful routine, or its "
        "identity does not match its appearance.",
    ),
    (
        "What does a classification of unknown require of the reporting unit?",
        "A classification of unknown requires continued surveillance, an attempt to establish identity "
        "by at least two independent means, a report to the operations room, and no assumption that the "
        "contact is harmless.",
    ),
    (
        "What evidence sources are used to classify a contact?",
        "AIS, radar, visual observation, electronic support measures, acoustic observation, port and "
        "VTS reports, and the vessel registry and passage records held in the exercise database. At "
        "least two independent sources should agree before a contact is called confirmed.",
    ),
    (
        "Why is AIS alone not sufficient to classify a contact?",
        "AIS can be switched off, not fitted, wrongly configured, or deliberately falsified. A contact "
        "reported by AIS must be corroborated by radar position, visual appearance and, where "
        "available, ESM before the identity is accepted.",
    ),
    (
        "What is a dark contact?",
        "A dark contact is a surface contact that is detected by radar or visual means but is not "
        "transmitting AIS. Dark contacts are the highest priority for identification because they "
        "remove the easiest means of establishing identity.",
    ),
    (
        "What are the indicators of a fishing vessel?",
        "Low speed of 2 to 6 knots, working in a group, gear deployed, class B AIS or no AIS at all, "
        "small radar cross section, a pattern of movement that follows fishing grounds rather than a "
        "track between ports, and often no response to VHF challenge.",
    ),
    (
        "What are the indicators of a traditional dhow?",
        "Wooden hull, single or twin mast, no AIS, no visible registration marks, speed 5 to 7 knots, "
        "a low freeboard and no deck cargo arrangement typical of a merchant ship. Dhows in the Kutch "
        "approaches are common and usually lawful, but the identity must still be established.",
    ),
    (
        "What are the indicators of a suspicious approach?",
        "Closing at high speed on a steady bearing, no AIS, no response to VHF challenge, closing to "
        "within 2 nm of a merchant vessel or of an own unit, loitering on a shipping route, and "
        "turning away only when challenged or when a warning signal is made.",
    ),
    (
        "What are the classification confidence levels?",
        "Confirmed means at least two independent sources agree and the identity is established. "
        "Probable means the available evidence strongly supports the classification. Possible means "
        "the evidence is indicative only. Unknown means there is no reliable basis for a "
        "classification.",
    ),
    (
        "What fields must a contact report contain?",
        "Report identifier, date and time in Zulu, reporting unit, position, contact designation, "
        "sensor or sensors used, classification, affiliation, confidence, MMSI or other identity if "
        "established, action taken, and remarks.",
    ),
    (
        "When must a contact be escalated to the operations room?",
        "Immediately for any sub-surface contact, any suspect affiliation, any approach within 2 nm of "
        "a merchant vessel or own unit, any dark contact inside a hazard or exercise area, any AIS "
        "identity mismatch, and any contact that does not respond to two VHF challenges.",
    ),
    (
        "What is the difference between neutral and friendly in this exercise?",
        "Neutral is an identified non-participant, for example a merchant ship on a lawful passage. "
        "Friendly is an identified participant or own force unit, for example an exercise warship or "
        "coast guard unit that does not transmit AIS while on task.",
    ),
    (
        "How is a warship classified when it is not transmitting AIS?",
        "A grey hull, a warship profile, high speed and no AIS is a probable warship, but the "
        "affiliation is unknown until it is identified by ESM, by challenge procedure or by exercise "
        "means. Do not assume friendly on appearance alone.",
    ),
    (
        "What does ESM add to contact classification?",
        "ESM detects radar and communication emissions. It can establish the type of unit and, where "
        "the emission is recognised, the affiliation, without relying on AIS or visual identification.",
    ),
    (
        "How is a contact classified when the AIS identity does not match the physical contact?",
        "An AIS identity that does not match the radar and visual picture is classified as a merchant "
        "vessel with an AIS mismatch, affiliation suspect, and is escalated to the operations room. "
        "Possible causes are a faulty transponder configuration, an incorrect entry by the crew, or "
        "deliberate identity misuse. All three require reporting.",
    ),
    (
        "What is the difference between a drifting object and a vessel?",
        "A drifting object has no propulsion, no AIS, no response to VHF and a drift rate set by wind "
        "and current, typically under 2 knots. A vessel will hold a course, may alter speed, and may "
        "respond to a challenge.",
    ),
    (
        "What is a sub-surface contact and what is the immediate action?",
        "A sub-surface contact is any indication of a submerged unit, such as a periscope sighting or "
        "a weak intermittent radar return with no AIS. The immediate action is to report to the "
        "operations room, maintain contact if possible, and not to assume the contact is friendly.",
    ),
    (
        "What are the reporting thresholds for small craft?",
        "Any small craft approaching within 2 nm of a merchant vessel or of an own unit is reported, "
        "whether or not it is hostile. Approaches at night or in reduced visibility are reported at a "
        "greater range, normally 5 nm.",
    ),
    (
        "How is a survey vessel classified?",
        "A survey vessel is classified as neutral when its identity is established and its operations "
        "match a survey warning in force. It is often restricted in her ability to manoeuvre, so "
        "vessels are to keep clear by not less than 3 nm.",
    ),
    (
        "How should a contact inside a known fishing fleet area be handled?",
        "Cross-check the position against the fishing gear concentration warning in force, keep a "
        "sharp lookout for unlit gear, reduce speed where practicable, and still attempt to establish "
        "identity. A contact inside a fishing area is not automatically a fishing vessel.",
    ),
    (
        "How is an own coast guard unit classified when it is not transmitting AIS?",
        "Friendly, once identified by exercise means. Until then it is an unknown surface contact and "
        "is handled as such, because a unit that does not transmit AIS cannot be assumed to be "
        "friendly on the basis of its appearance.",
    ),
    (
        "What should be done when a contact matches a registry vessel but is in a place that vessel "
        "cannot be?",
        "The mismatch is reported as suspect. Compare the reported position with the vessel's AIS "
        "passage record. If the physical contact cannot be in that position at that time, treat the "
        "identity as unverified and continue surveillance.",
    ),
    (
        "How is an unidentified helicopter classified?",
        "As an air contact with unknown affiliation. Note the type, height, course and speed, report "
        "to the operations room, and cross-check against scheduled movements before classifying it as "
        "friendly.",
    ),
    (
        "How long should surveillance be maintained on an unknown contact?",
        "Until the identity is established, or until the operations room directs otherwise, or until "
        "the contact is no longer of interest because it has cleared the area. A contact is not closed "
        "simply because it has moved out of visual range.",
    ),
    (
        "What is the rule for passing a tanker transfer pair?",
        "Keep clear by not less than 2 nm and do not pass between the two ships. Transfer pairs are "
        "constrained in their ability to manoeuvre and one of the two may be at anchor with a hose "
        "connected.",
    ),
    (
        "How are drifting containers classified?",
        "As a drifting object, affiliation not applicable, confidence confirmed when sighted. They "
        "are a navigation hazard, they are difficult to detect by radar when awash, and their position "
        "is passed to the operations room and to the shipping report.",
    ),
    (
        "What affiliation is given to a merchant vessel with a valid AIS and a matching radar track?",
        "Neutral. A valid identity on two independent sensors with a track that matches the declared "
        "voyage is enough to call the contact a neutral merchant vessel, though it remains subject to "
        "reporting if it behaves unusually.",
    ),
    (
        "What should a unit do if a dark contact enters a live firing area?",
        "Warn the contact on VHF channel 16, report it to exercise control and the operations room, "
        "and do not assume it will clear the area on its own. If the contact does not respond, the "
        "decision on continuing the firing rests with exercise control.",
    ),
    (
        "What is an AIS identity mismatch and how common is it in this exercise area?",
        "An AIS identity mismatch is a contact whose transmitted identity does not match its radar and "
        "visual appearance, or whose position is inconsistent with the identity's known movements. "
        "Two mismatches were reported between January and August 2026, one in February and one in "
        "August, and both were escalated to the operations room.",
    ),
]


def matched_contacts():
    out = []
    for mmsi, legs in LEGS_BY_MMSI.items():
        vessel = VESSEL_BY_MMSI[mmsi]
        spans = [(leg["dep"], leg["arr"]) for leg in legs]
        total = sum((a - d).total_seconds() for d, a in spans)
        for k in range(5):
            target = total * (k + 0.5) / 5
            acc = 0.0
            for d, a in spans:
                span = (a - d).total_seconds()
                if acc + span >= target:
                    when = d + timedelta(seconds=target - acc)
                    break
                acc += span
            pos = position_at(mmsi, when)
            out.append({
                "mmsi": mmsi,
                "vessel": vessel["name"],
                "when": when,
                "lat": pos["lat"],
                "lon": pos["lon"],
                "course": pos["course"],
                "speed": pos["speed"],
                "draught": pos["leg"]["draught"],
                "classification": f"Merchant {vessel['type'].lower()}",
                "affiliation": "neutral",
                "confidence": "confirmed",
            })
    out.sort(key=lambda c: c["when"])
    return out


MATCHED_CONTACTS = matched_contacts()


OUT_OF_SCOPE = [
    (
        "What is the cargo of the dhow reported on 14 February 2026?",
        "The data does not contain cargo information for that contact. The dhow was not boarded and "
        "no manifest or declaration was obtained, so the cargo is unknown. Do not estimate it. Refer "
        "the question to an officer if the cargo matters to the decision.",
    ),
    (
        "Which flag does the dhow reported on 14 February 2026 fly?",
        "The flag is not established. No registration marks were observed and there was no response "
        "to VHF challenge, so neither the flag nor the port of registry can be stated. Record the "
        "flag as not established and refer to an officer.",
    ),
    (
        "How many crew are on board the dhow reported on 14 February 2026?",
        "The number of crew is not known. Only the observable characteristics were recorded: wooden "
        "hull, single mast, no AIS, speed 6 knots on course 090. Do not guess a crew number.",
    ),
    (
        "Is the fast boat reported on 21 July 2026 connected to any known group?",
        "The data does not support any attribution. The report records observable behaviour only: an "
        "11 m skiff, twin outboards, 25 knots, no AIS, no marks, closing to 2 nm of a passing general "
        "cargo ship and withdrawing when challenged. Any link to a group is a matter for an officer, "
        "not for the data.",
    ),
    (
        "What was the intention of the small craft that approached MV MALABAR EXPRESS on 22 April 2026?",
        "The intention cannot be determined from the data. The report establishes what was observed, "
        "not why. State the observed behaviour, classify it as a suspect approach, and leave intent to "
        "an officer.",
    ),
    (
        "Who owns the vessel that transmitted the mismatched AIS identity on 27 August 2026?",
        "The identity of that contact was never established, so ownership cannot be stated. The AIS "
        "transmission did not match the physical contact and the position was inconsistent with the "
        "named vessel's known movements. Report the mismatch and refer the question to an officer.",
    ),
    (
        "Was the dhow reported on 14 February 2026 carrying out fishing, smuggling or a legitimate "
        "passage?",
        "The data does not answer that question. The contact was a wooden dhow with no AIS, no "
        "registration marks, speed 6 knots and no response to VHF challenge, reported three times "
        "over the period. That is a pattern worth reporting, but it is not evidence of a particular "
        "activity. Refer to an officer.",
    ),
    (
        "What will the dhow reported on 14 February 2026 do next?",
        "The data cannot predict that. Nothing in the track, warning or contact records supports a "
        "prediction of future behaviour. State what is known, note the recurring pattern, and refer "
        "the assessment to an officer.",
    ),
]


def all_contacts():
    records = []
    for c in HAND_CONTACTS:
        records.append({
            "id": c["id"], "when": parse_dt(c["dt"]), "unit": c["unit"],
            "lat": c["lat"], "lon": c["lon"], "designation": c["designation"],
            "sensor": c["sensor"], "classification": c["classification"],
            "affiliation": c["affiliation"], "confidence": c["confidence"],
            "mmsi": c["mmsi"], "vessel": c["vessel"],
            "actions": c["actions"], "remarks": c["remarks"], "synthetic_match": False,
        })
    for i, c in enumerate(MATCHED_CONTACTS, start=1):
        designation = f"C-{80 + i}"
        records.append({
            "id": f"SCR-2026-{80 + i:03d}",
            "when": c["when"],
            "unit": ["Exercise Unit Alpha", "Exercise Unit Bravo", "Exercise Unit Charlie"][i % 3],
            "lat": c["lat"], "lon": c["lon"], "designation": designation,
            "sensor": "AIS and radar",
            "classification": c["classification"],
            "affiliation": c["affiliation"], "confidence": c["confidence"],
            "mmsi": c["mmsi"], "vessel": c["vessel"],
            "actions": "Identity confirmed against the vessel registry and passage records. No action required.",
            "remarks": (
                f"Track and reported draught of {c['draught']} m match the registry record. "
                f"Course {c['course']:.0f} degrees, speed {c['speed']:.1f} knots."
            ),
            "synthetic_match": True,
        })
    records.sort(key=lambda r: r["when"])
    return records


ALL_CONTACTS = all_contacts()


def build_contact_classification():
    b = QABuilder(
        "CC", "contact_classification",
        "Contact classification doctrine and classification entries",
    )
    for question, answer in CLASSIFICATION_DOCTRINE:
        b.add(question, answer, "Contact classification standing instruction", ["doctrine"])
    for c in ALL_CONTACTS:
        ident = f" (matched to {c['vessel']}, MMSI {c['mmsi']})" if c["mmsi"] else ""
        b.add(
            f"How was contact {c['designation']} classified on {fmt_date(c['when'])} and with what "
            f"confidence?",
            f"Contact {c['designation']} was classified as {lower_label(c['classification'])} with "
            f"affiliation {c['affiliation']} and confidence {c['confidence']}{ident}.",
            f"Contact report {c['id']}", ["entry"],
        )
        b.add(
            f"What indicators supported the classification of contact {c['designation']} on "
            f"{fmt_date(c['when'])}?",
            f"{c['remarks']} Sensor or sensors used: {c['sensor']}.",
            f"Contact report {c['id']}", ["entry", "indicators"],
        )
        b.add(
            f"Was the classification of contact {c['designation']} on {fmt_date(c['when'])} supported "
            f"by more than one sensor?",
            f"The report used {c['sensor']}. "
            + (
                "More than one independent source was used, which is what allows the classification to "
                "be carried at confirmed confidence."
                if c["confidence"] == "confirmed"
                else "The evidence was not sufficient for a confirmed classification, so the contact "
                "was carried at " + c["confidence"] + " confidence and remained open for further "
                "identification."
            ),
            f"Contact report {c['id']}", ["entry", "evidence"],
        )
    suspects = [c for c in ALL_CONTACTS if c["affiliation"] == "suspect"]
    unknowns = [c for c in ALL_CONTACTS if c["affiliation"] == "unknown"]
    dark = [c for c in ALL_CONTACTS if not c["mmsi"] and c["affiliation"] != "not applicable"]
    b.add(
        "Which contacts in the period were classified as suspect?",
        f"{count_word(len(suspects))} contacts were classified suspect: "
        + "; ".join(
            f"contact {c['designation']} on {fmt_date(c['when'])} ({lower_label(c['classification'])})"
            for c in suspects
        )
        + ". Every one of them involved either a fast craft closing on a merchant vessel without AIS, "
        "or an identity that did not match the physical contact.",
        "Contact report file, affiliation query", ["aggregate", "suspect"],
    )
    b.add(
        "How many contacts were classified with unknown affiliation?",
        f"{count_word(len(unknowns))} contacts were classified with unknown affiliation: "
        + "; ".join(f"contact {c['designation']} on {fmt_date(c['when'])}" for c in unknowns)
        + ".",
        "Contact report file, affiliation query", ["aggregate"],
    )
    b.add(
        "Which contacts had no AIS transmission at all?",
        f"{count_word(len(dark))} contacts were reported with no AIS transmission: "
        + "; ".join(
            f"contact {c['designation']} on {fmt_date(c['when'])} ({lower_label(c['classification'])})"
            for c in dark
        )
        + ". A contact with no AIS is not automatically hostile, but it cannot be identified from AIS "
        "and must be worked by radar, visual and ESM means.",
        "Contact report file, AIS query", ["aggregate", "dark-contact"],
    )
    b.add(
        "Which contact reports were escalated to the operations room?",
        "Escalation was required for every suspect fast boat approach, every dark contact inside a "
        "hazard or exercise area, the possible sub-surface contact, and the AIS identity mismatch in "
        "August. Routine merchant and fishing contacts were reported but not escalated.",
        "Contact report file, escalation query", ["aggregate", "escalation"],
    )
    for question, answer in OUT_OF_SCOPE:
        b.add(question, answer, "Contact classification standing instruction, limits of the data",
              ["out-of-scope"])
    return b


# --------------------------------------------------------------------------
# file 3: navigational warnings
# --------------------------------------------------------------------------


def build_navigational_warnings():
    b = QABuilder(
        "NAVW", "navigational_warnings",
        "NAVAREA IX warnings, coastal warnings and notices to mariners, January to August 2026",
    )
    for kind, number, title, issued, cancelled, area, hazard in WARNINGS:
        start = parse_dt(issued)
        end = parse_dt(cancelled) if cancelled else None
        summary, action = WARNING_OVERRIDES.get(number, HAZARD_TEXT[hazard])
        label = f"{kind} {number}"
        area_line = area_text(area) if area else "the area described in the warning text"
        validity = f"in force from {fmt_dt(start)}" + (
            f" until {fmt_dt(end)}" if end else " until further notice"
        )
        b.add(
            f"What does {label} say?",
            f"{label}, {title}. {summary} Area affected: {area_line}. The warning is {validity}.",
            label, ["detail"],
        )
        b.add(
            f"What action is required of a merchant vessel for {label}?",
            f"{action} The warning remains {validity}.",
            label, ["action"],
        )
    for month in range(1, 9):
        for day in (8, 18, 27):
            when = datetime(2026, month, day, 12, 0)
            active = warnings_active(when)
            if not active:
                continue
            listing = "; ".join(f"{a['kind']} {a['number']} ({a['title']})" for a in active)
            b.add(
                f"Which navigational warnings were in force in the exercise area on {fmt_date(when)}?",
                f"On {fmt_date(when)} the following warnings were in force: {listing}.",
                f"Warning status file, {fmt_date(when)}", ["in-force"],
            )
    firing = [w for w in WARNINGS if w[6] == "live firing"]
    b.add(
        "List every live firing warning issued between January and August 2026.",
        "Live firing warnings were issued in Area Alpha as follows: NAVAREA IX 006/2026 from 8 to 16 "
        "January, 031/2026 from 9 to 13 February, 052/2026 from 8 to 15 March, 072/2026 from 27 to 29 "
        "March, 074/2026 from 7 to 10 April, 097/2026 from 5 to 8 May, 121/2026 from 9 to 12 June, "
        "141/2026 from 7 to 10 July and 161/2026 from 11 to 14 August. That is nine live firing "
        "warnings in eight months.",
        "Warning file, hazard query", ["aggregate", "live-firing"],
    )
    gnss = [w for w in WARNINGS if w[6] == "navigation interference"]
    b.add(
        "How many GNSS interference warnings were issued and when?",
        f"{len(gnss)} GNSS interference warnings were issued: 024/2026 in late January, 067/2026 in "
        "March, 095/2026 in late April, 133/2026 in June, 158/2026 in late July and 175/2026 in "
        "August. Each reported intermittent loss of position fixing and degraded accuracy.",
        "Warning file, hazard query", ["aggregate", "gnss"],
    )
    b.add(
        "Which warnings affected the approach to Mundra between January and August 2026?",
        "Mundra was affected by 048/2026, a night entry restriction in force from 18 February to 6 "
        "March, 059/2026 for an unlit buoy from 2 to 18 March, 116/2026, a port entry restriction "
        "from 28 May to 10 June, and by the recurring seabed survey warnings in Area Bravo which "
        "covered the approaches in January, March, April, May, June, July and August.",
        "Warning file, port query", ["aggregate", "port"],
    )
    b.add(
        "Was Area Alpha under a live firing warning on 12 March 2026?",
        "Yes. NAVAREA IX 052/2026, live firing exercise in Area Alpha, was in force from 8 March 2026 "
        "at 0600Z until 15 March 2026 at 1800Z. Any vessel transiting Area Alpha on 12 March was "
        "inside an active live firing area.",
        "NAVAREA IX 052/2026", ["story", "live-firing"],
    )
    firing_windows = [
        (number, parse_dt(issued), parse_dt(cancelled))
        for kind, number, title, issued, cancelled, area, hazard in WARNINGS
        if hazard == "live firing" and area == "ALPHA"
    ]
    hits = []
    for mmsi, start, end, leg in area_transits("ALPHA"):
        for number, wstart, wend in firing_windows:
            if start <= wend and end >= wstart:
                hits.append((mmsi, start, end, number))
                break
    if hits:
        lines = [
            f"{VESSEL_BY_MMSI[mmsi]['name']} (MMSI {mmsi}) between {start.strftime('%H%M')}Z and "
            f"{end.strftime('%H%M')}Z on {fmt_date(start)}, while NAVAREA IX {number} was in force"
            for mmsi, start, end, number in hits
        ]
        b.add(
            "Did any tracked vessel pass through Area Alpha while a live firing warning was in force?",
            f"Yes, on {count_word(len(hits)).lower()} occasions. " + "; ".join(lines) + ". Each vessel should have been "
            "warned by the exercise control authority and advised to clear the area.",
            "Warning file cross-checked against AIS passage records", ["story", "live-firing"],
        )
    return b


# --------------------------------------------------------------------------
# file 4: AIS vessel tracks
# --------------------------------------------------------------------------


def build_ais_tracks():
    b = QABuilder(
        "AIS", "ais_vessel_tracks",
        "AIS track, passage and port call records for three tracked vessels, January to August 2026",
    )
    for mmsi, legs in LEGS_BY_MMSI.items():
        v = VESSEL_BY_MMSI[mmsi]
        name = v["name"]
        for leg in legs:
            frm, to = port_name(leg["frm"]), port_name(leg["to"])
            hours = (leg["arr"] - leg["dep"]).total_seconds() / 3600
            src = f"AIS passage record, MMSI {mmsi}, leg {leg['index']}"
            b.add(
                f"Which port did {name} depart on {fmt_date(leg['dep'])} and where was she bound?",
                f"{name} (MMSI {mmsi}) departed {frm} at {leg['dep'].strftime('%H%M')}Z on "
                f"{fmt_date(leg['dep'])}. The AIS destination field was set to {leg['to']} ({to}). She "
                f"arrived at {to} at {leg['arr'].strftime('%H%M')}Z on {fmt_date(leg['arr'])}.",
                src, ["passage"],
            )
            b.add(
                f"How long did the passage of {name} from {frm} to {to} on {fmt_date(leg['dep'])} take "
                f"and at what average speed?",
                f"The passage covered {leg['distance']:.0f} nm in {hours:.0f} hours at an average speed "
                f"of {leg['speed']:.1f} knots, departing {frm} at {leg['dep'].strftime('%H%M')}Z on "
                f"{fmt_date(leg['dep'])} and arriving {to} at {leg['arr'].strftime('%H%M')}Z on "
                f"{fmt_date(leg['arr'])}.",
                src, ["passage", "speed"],
            )
            b.add(
                f"What route, draught and cargo did {name} report on the passage from {frm} to {to} "
                f"on {fmt_date(leg['dep'])}?",
                f"{name} routed from {frm} to {to} {route_text(leg)}. The AIS draught field read "
                f"{leg['draught']} m and the cargo on board was {leg['cargo']}.",
                src, ["route", "cargo"],
            )
            b.add(
                f"When did {name} arrive at {to} after departing {frm} on {fmt_date(leg['dep'])}?",
                f"{name} arrived at {to} at {leg['arr'].strftime('%H%M')}Z on {fmt_date(leg['arr'])}, "
                f"{hours:.0f} hours after departure from {frm}. The passage was {leg['distance']:.0f} nm.",
                src, ["passage", "arrival"],
            )
        # position samples across the period
        for month in range(1, 9):
            for day in (5, 10, 15, 20, 25):
                when = datetime(2026, month, day, 6, 0)
                pos = position_at(mmsi, when)
                if pos["kind"] == "port":
                    b.add(
                        f"Where was {name} on {fmt_date(when)} at 0600Z?",
                        f"{name} was alongside or at anchor at {port_name(pos['port'])} on "
                        f"{fmt_date(when)} at 0600Z, with a reported draught in the AIS static data and "
                        f"no speed over ground.",
                        f"AIS position record, MMSI {mmsi}, {fmt_date(when)}", ["position"],
                    )
                else:
                    b.add(
                        f"Where was {name} on {fmt_date(when)} at 0600Z?",
                        f"{name} was at {fmt_pos(pos['lat'], pos['lon'])}, {describe_position(pos['lat'], pos['lon'])}, "
                        f"on course {pos['course']:.0f} degrees at {pos['speed']:.1f} knots, on passage "
                        f"from {port_name(pos['leg']['frm'])} to {port_name(pos['leg']['to'])}.",
                        f"AIS position record, MMSI {mmsi}, {fmt_date(when)}", ["position"],
                    )
        b.add(
            f"List the ports {name} called at between January and August 2026.",
            f"{name} called at "
            + ", ".join(dict.fromkeys([port_name(l["frm"]) for l in legs] + [port_name(legs[-1]["to"])]))
            + f". She completed {len(legs)} passages in the period.",
            f"AIS port call record, MMSI {mmsi}", ["aggregate", "port-calls"],
        )
        b.add(
            f"What is the reported destination and draught of {name} when she is bound for "
            f"{port_name(legs[-1]['to'])}?",
            f"On her last recorded passage in the period, {name} set the AIS destination field to "
            f"{legs[-1]['to']} ({port_name(legs[-1]['to'])}) and reported a draught of "
            f"{legs[-1]['draught']} m, departing {port_name(legs[-1]['frm'])} at "
            f"{legs[-1]['dep'].strftime('%H%M')}Z on {fmt_date(legs[-1]['dep'])}.",
            f"AIS static data, MMSI {mmsi}", ["destination"],
        )
    # cross-vessel and story queries
    b.add(
        "An AIS contact appears about 60 nm west of Porbandar on 12 March 2026. Which tracked vessel "
        "is it and where is she coming from?",
        "On 12 March 2026 the only tracked vessel in that area was MT GULF PIONEER, MMSI 419876543, a "
        "crude oil tanker in ballast. She departed Vadinar at 0800Z on 12 March bound for Fujairah and "
        "passed through the northern part of the Gulf of Kutch approaches on an outbound track. Her "
        "draught was 5.6 m, which confirms a ballast passage rather than a loaded one. Note that "
        "NAVAREA IX 052/2026, a live firing warning for Area Alpha, was in force at the time.",
        "AIS passage record, MMSI 419876543, and NAVAREA IX 052/2026",
        ["story", "horizon-sighting"],
    )
    b.add(
        "A radar contact is reported at 22.68N 068.95E at 0915Z on 27 March 2026. Which tracked vessel "
        "does that match?",
        "The position matches MV AL RAYYAN STAR, MMSI 470123456, on her outbound passage from Mundra "
        "to Jebel Ali. She departed Mundra at 0545Z on 27 March and was inside Area Alpha at that time. "
        "NAVAREA IX 072/2026, a live firing warning for Area Alpha, had come into force at 0600Z the "
        "same morning.",
        "AIS passage record, MMSI 470123456, and NAVAREA IX 072/2026",
        ["story", "horizon-sighting"],
    )
    b.add(
        "Which tracked vessels were at sea in the Gulf of Kutch on 14 February 2026?",
        "On 14 February 2026 MV AL RAYYAN STAR was on passage from Mundra to Jebel Ali and MT GULF "
        "PIONEER was on passage from Fujairah to Vadinar. MV CORAL HORIZON was on passage from Kochi "
        "to Colombo and was well south of the exercise area.",
        "AIS position records, 14 February 2026", ["aggregate", "position"],
    )
    b.add(
        "Which vessel is the fastest of the tracked ships and what is her service speed?",
        "MV CORAL HORIZON, MMSI 353136000, is the fastest, with a planned passage speed of 16.5 knots "
        "and a service speed of 16.5 knots. MT GULF PIONEER plans 12.5 to 13.0 knots and MV AL RAYYAN "
        "STAR plans 12.0 to 13.0 knots.",
        "AIS passage records, speed comparison", ["aggregate", "speed"],
    )
    b.add(
        "How many passages did MT GULF PIONEER complete between Fujairah and Vadinar in the period?",
        f"MT GULF PIONEER completed {len(LEGS_BY_MMSI['419876543'])} passages in the period, all on "
        f"the Fujairah to Vadinar route, alternating loaded westbound legs at about 12.5 m draught with "
        f"ballast eastbound legs at about 5.5 m draught.",
        "AIS passage records, MMSI 419876543", ["aggregate", "port-calls"],
    )
    return b


# --------------------------------------------------------------------------
# file 5: shipping reports
# --------------------------------------------------------------------------


def build_shipping_reports():
    b = QABuilder(
        "SR", "shipping_reports",
        "North Arabian Sea shipping reports, monthly and periodic, January to August 2026",
    )
    for m in MONTHLY:
        label = datetime.strptime(m["month"], "%Y-%m").strftime("%B %Y")
        types = ", ".join(f"{k} {v}" for k, v in m["by_type"].items())
        waits = ", ".join(f"{k} {v} hours" for k, v in m["waits"].items())
        src = f"Shipping report, {label}"
        b.add(
            f"How many merchant transits were reported in the exercise area in {label}?",
            f"{m['transits']} merchant transits were reported in {label}, comprising {types}.",
            src, ["volume"],
        )
        b.add(
            f"What were the average anchorage waiting times in {label}?",
            f"Average anchorage waiting times in {label} were {waits}. Ship to ship transfer "
            f"operations in the month numbered {m['sts_ops']}.",
            src, ["congestion"],
        )
        b.add(f"What was the weather in the exercise area in {label}?", m["weather"], src, ["weather"])
        b.add(
            f"What security incidents were reported in {label}?",
            " ".join(m["incidents"]), src, ["incidents"],
        )
        b.add(
            f"What were the notable commercial developments in {label}?",
            " ".join(m["notable"]), src, ["commercial"],
        )
        b.add(
            f"What advisories were in force in {label}?",
            " ".join(m["advisories"]), src, ["advisory"],
        )
        b.add(
            f"What dark contact activity was recorded in {label}?",
            m["dark_activity"], src, ["dark-contact"],
        )
        b.add(
            f"Summarise the shipping and security picture in {label}.",
            f"{m['transits']} merchant transits ({types}). Average anchorage waiting times: {waits}. "
            f"{m['weather']} Incidents: {' '.join(m['incidents'])} Advisories: "
            f"{' '.join(m['advisories'])} Dark contacts: {m['dark_activity']}",
            src, ["summary"],
        )
    for o in OVERARCHING:
        src = f"Shipping report, {o['id']}"
        b.add(f"What does the {o['title']} say?", o["summary"], src, ["summary", "periodic"])
        b.add(
            f"What were the main security themes in the {o['period']}?",
            o["incidents"], src, ["incidents", "periodic"],
        )
        b.add(
            f"What should be watched in the {o['period']}?",
            " ".join(o["watch_items"]), src, ["watch", "periodic"],
        )
    busiest = max(MONTHLY, key=lambda m: m["transits"])
    quietest = min(MONTHLY, key=lambda m: m["transits"])
    total = sum(m["transits"] for m in MONTHLY)
    b.add(
        "Which month had the highest number of merchant transits and which had the lowest?",
        f"The highest was {datetime.strptime(busiest['month'], '%Y-%m').strftime('%B %Y')} with "
        f"{busiest['transits']} transits. The lowest was "
        f"{datetime.strptime(quietest['month'], '%Y-%m').strftime('%B %Y')} with "
        f"{quietest['transits']} transits, during the peak of the south-west monsoon.",
        "Shipping report file, volume comparison", ["comparison"],
    )
    b.add(
        "How many merchant transits were recorded in total between January and August 2026?",
        f"{total:,} merchant transits were recorded across the eight months. Traffic fell from a peak "
        f"of {busiest['transits']} in March to a low of {quietest['transits']} in July, a fall of "
        f"{100 * (busiest['transits'] - quietest['transits']) / busiest['transits']:.0f} per cent, "
        f"before recovering to {MONTHLY[-1]['transits']} in August.",
        "Shipping report file, volume comparison", ["comparison"],
    )
    b.add(
        "How did tanker traffic change between March and July 2026?",
        "Tanker transits fell from 184 in March to 126 in July, a fall of 58 transits or about 32 per "
        "cent, before recovering to 134 in August. The fall tracked the onset of the south-west "
        "monsoon on 8 June and the suspension of ship to ship transfer operations.",
        "Shipping report file, tanker trend", ["comparison", "tanker"],
    )
    b.add(
        "Which month had the worst anchorage congestion and where?",
        "March 2026 had the worst congestion of the period, with an average wait of 28.7 hours at "
        "Vadinar, 21.3 hours at Deendayal and 12.9 hours at Mundra. Vadinar was the worst single "
        "berth throughout the first quarter because of berth congestion on the crude shuttle.",
        "Shipping report file, congestion comparison", ["comparison", "congestion"],
    )
    b.add(
        "How many suspicious approach reports were filed between January and August 2026?",
        "Twenty suspicious approach reports were filed across the eight months: two in January, three "
        "in February, four in March, three in April, two in May, one in June, two in July and three in "
        "August. Every one involved a small, fast craft with no AIS and no registration marks.",
        "Shipping report file, security comparison", ["comparison", "incidents"],
    )
    b.add(
        "What does the reporting say about dark contacts and AIS identity mismatches?",
        "Twenty-eight contacts with no AIS transmission were recorded across the eight months, "
        "peaking at six in March. Three AIS identity mismatches were reported, in February, in "
        "August, and one further case that was resolved after the vessel was queried through the "
        "regional reporting chain.",
        "Shipping report file, dark contact summary", ["comparison", "dark-contact"],
    )
    return b


# --------------------------------------------------------------------------
# file 6: synthetic contact reports
# --------------------------------------------------------------------------


def build_synthetic_contact_reports():
    b = QABuilder(
        "SCR", "synthetic_contact_reports",
        "Synthetic contact reports filed by exercise units, January to August 2026",
    )
    for c in ALL_CONTACTS:
        src = f"Contact report {c['id']}"
        ident = f" Matched to {c['vessel']}, MMSI {c['mmsi']}." if c["mmsi"] else ""
        b.add(
            f"What did {c['unit']} report at {c['when'].strftime('%H%M')}Z on {fmt_date(c['when'])}?",
            f"{c['unit']} reported contact {c['designation']} at {fmt_pos(c['lat'], c['lon'])}, "
            f"{describe_position(c['lat'], c['lon'])}. The contact was classified as "
            f"{lower_label(c['classification'])}, affiliation {c['affiliation']}, confidence "
            f"{c['confidence']}, using {c['sensor']}.{ident}",
            src, ["report"],
        )
        b.add(
            f"Where and when was contact {c['designation']} reported?",
            f"Contact {c['designation']} was reported at {fmt_pos(c['lat'], c['lon'])}, "
            f"{describe_position(c['lat'], c['lon'])}, at {c['when'].strftime('%H%M')}Z on "
            f"{fmt_date(c['when'])} by {c['unit']}.",
            src, ["position"],
        )
        b.add(
            f"What action was taken for contact {c['designation']} on {fmt_date(c['when'])}?",
            f"{c['actions']} Remarks: {c['remarks']}",
            src, ["action"],
        )
    by_month = {}
    for c in ALL_CONTACTS:
        by_month.setdefault(c["when"].month, []).append(c)
    b.add(
        f"How many contact reports were filed between January and August 2026?",
        f"{len(ALL_CONTACTS)} contact reports were filed in the period, of which "
        f"{sum(1 for c in ALL_CONTACTS if c['synthetic_match'])} were matched to a vessel in the "
        f"registry and {sum(1 for c in ALL_CONTACTS if not c['synthetic_match'])} were unidentified or "
        f"non-vessel contacts.",
        "Contact report file, count query", ["aggregate"],
    )
    for month in sorted(by_month):
        rows = by_month[month]
        label = datetime(2026, month, 1).strftime("%B 2026")
        listing = "; ".join(
            f"{c['id']} contact {c['designation']} ({lower_label(c['classification'])}, {c['affiliation']})"
            for c in rows
        )
        b.add(
            f"Which contact reports were filed in {label}?",
            f"{len(rows)} reports were filed in {label}: {listing}.",
            f"Contact report file, {label}", ["aggregate", "monthly"],
        )
    b.add(
        "Which contact reports involved a craft approaching within 2 nm of a merchant vessel?",
        "Three reports: SCR-2026-009 on 2 February 2026, where a skiff closed to 1.5 nm on the port "
        "bow of a bulk carrier; SCR-2026-036 on 22 April 2026, where a skiff closed to 1.8 nm on the "
        "stern of MV MALABAR EXPRESS at 0205Z; and SCR-2026-062 on 21 July 2026, where a skiff closed "
        "to 2 nm of a passing general cargo ship. All three were classified suspect and escalated.",
        "Contact report file, approach query", ["aggregate", "suspect"],
    )
    b.add(
        "Which contact reports involve a vessel that was inside a hazard or exercise area at the time?",
        "Three reports. SCR-2026-021 on 12 March 2026, when MT GULF PIONEER was inside Area Alpha "
        "while NAVAREA IX 052/2026 live firing warning was in force. SCR-2026-028 on 27 March 2026, "
        "when MV AL RAYYAN STAR was inside Area Alpha while NAVAREA IX 072/2026 was in force. "
        "SCR-2026-017 on 5 March 2026, when a survey vessel was working inside Area Bravo as "
        "declared in the survey warning.",
        "Contact report file cross-checked against the warning file",
        ["aggregate", "story"],
    )
    b.add(
        "Which exercise unit filed the most contact reports?",
        "Exercise Unit Alpha filed the most reports, covering the dhow sightings, the fast boat "
        "approaches, the live firing area transits and the AIS identity mismatch in August.",
        "Contact report file, unit query", ["aggregate"],
    )
    b.add(
        "Which contact report is the best example of a dark contact that later turned out to be a "
        "known fishing vessel?",
        "SCR-2026-045 on 24 May 2026. A wooden trawler was detected by visual and AIS means and matched "
        "to FV SAGAR SETU, MMSI 419555777, working out of Okha with gear deployed at 4.5 knots. It "
        "shows the standard pattern: a dark or intermittent contact becomes identifiable once AIS, "
        "visual appearance, home port markings and the fishing gear warning are brought together.",
        "Contact report SCR-2026-045 and vessel registry dossier", ["story", "fishing"],
    )
    return b


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def write_dataset(builder):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{builder.topic}.json"
    path.write_text(json.dumps(builder.envelope(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path, len(builder.items)


def write_jsonl(builders):
    items = []
    for b in builders:
        items.extend(b.items)
    rng.shuffle(items)
    cut = int(len(items) * 0.9)
    for name, rows in (("train_qa.jsonl", items[:cut]), ("eval_qa.jsonl", items[cut:])):
        with (OUT_DIR / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({
                    "id": row["id"],
                    "topic": row["topic"],
                    "question": row["question"],
                    "answer": row["answer"],
                }, ensure_ascii=True) + "\n")
    return cut, len(items) - cut


def main():
    builders = [
        build_domain_vessel_reports(),
        build_contact_classification(),
        build_navigational_warnings(),
        build_ais_tracks(),
        build_shipping_reports(),
        build_synthetic_contact_reports(),
    ]
    total = 0
    print(f"{'file':40s}{'examples':>10s}")
    for builder in builders:
        path, count = write_dataset(builder)
        total += count
        print(f"{path.name:40s}{count:10d}")
    train, evaluation = write_jsonl(builders)
    print(f"{'train_qa.jsonl':40s}{train:10d}")
    print(f"{'eval_qa.jsonl':40s}{evaluation:10d}")
    print(f"{'TOTAL':40s}{total:10d}")


if __name__ == "__main__":
    main()
