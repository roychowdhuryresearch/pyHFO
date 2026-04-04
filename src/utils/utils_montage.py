from __future__ import annotations

import re
from collections import defaultdict


BIPOLAR_TOKEN = "#-#"
AVERAGE_REFERENCE_SUFFIX = "AVG"
AVERAGE_REFERENCE_LABEL = "Average Ref"
ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}
CONVENTIONAL_BIPOLAR_CHAINS = [
    (
        "left_parasagittal",
        [
            ("Fp1", "F3"),
            ("F3", "C3"),
            ("C3", "P3"),
            ("P3", "O1"),
        ],
    ),
    (
        "left_temporal",
        [
            ("Fp1", "F7"),
            ("F7", "T3"),
            ("T3", "T5"),
            ("T5", "O1"),
        ],
    ),
    (
        "right_parasagittal",
        [
            ("Fp2", "F4"),
            ("F4", "C4"),
            ("C4", "P4"),
            ("P4", "O2"),
        ],
    ),
    (
        "right_temporal",
        [
            ("Fp2", "F8"),
            ("F8", "T4"),
            ("T4", "T6"),
            ("T6", "O2"),
        ],
    ),
    (
        "midline",
        [
            ("Fz", "Cz"),
            ("Cz", "Pz"),
        ],
    ),
]
CONVENTIONAL_BIPOLAR_PAIRS = [pair for _chain_name, pairs in CONVENTIONAL_BIPOLAR_CHAINS for pair in pairs]
_CONVENTIONAL_NEIGHBOR_LOOKUP = defaultdict(list)
for _left_channel, _right_channel in CONVENTIONAL_BIPOLAR_PAIRS:
    if _right_channel not in _CONVENTIONAL_NEIGHBOR_LOOKUP[_left_channel]:
        _CONVENTIONAL_NEIGHBOR_LOOKUP[_left_channel].append(_right_channel)
    if _left_channel not in _CONVENTIONAL_NEIGHBOR_LOOKUP[_right_channel]:
        _CONVENTIONAL_NEIGHBOR_LOOKUP[_right_channel].append(_left_channel)

_NUMBERED_CHANNEL_PATTERN = re.compile(
    r"^(?P<stem>.*?)(?P<number>\d+)(?P<suffix>(?:\s*[-_ ]?(?:Ref|REF|ref))?)$"
)
_KNOWN_CHANNEL_PREFIX_PATTERN = re.compile(r"^(?:(?:EEG|POL|SEEG|ECOG)\s+)+", re.IGNORECASE)
_CONVENTIONAL_CHANNELS = tuple(dict.fromkeys(channel for pair in CONVENTIONAL_BIPOLAR_PAIRS for channel in pair))
_CONVENTIONAL_LOOKUP = {channel.upper(): channel for channel in _CONVENTIONAL_CHANNELS}
_ALIAS_LOOKUP = {alias.upper(): canonical for alias, canonical in ALIASES.items()}
_EEG_TOKEN_LOOKUP = dict(_CONVENTIONAL_LOOKUP)
_EEG_TOKEN_LOOKUP.update({alias.upper(): alias for alias in ALIASES})
_MIN_CONVENTIONAL_AUTO_ENTRIES = 4
_REFERENCE_SUFFIXES = (
    "REF",
    "LE",
    "AVG",
    "AV",
    "CAR",
    "M1",
    "M2",
    "A1",
    "A2",
)


def bipolar_channel_name(channel_1, channel_2):
    return f"{channel_1}{BIPOLAR_TOKEN}{channel_2}"


def average_reference_channel_name(channel_name):
    return f"{channel_name}{BIPOLAR_TOKEN}{AVERAGE_REFERENCE_SUFFIX}"


def is_bipolar_channel(channel_name):
    return BIPOLAR_TOKEN in str(channel_name or "")


def source_channel_names(channel_names):
    if channel_names is None:
        return []
    return [str(channel) for channel in channel_names if not is_bipolar_channel(channel)]


def _normalize_channel_text(name):
    text = re.sub(r"\s+", " ", str(name or "").strip())
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*_\s*", "_", text)
    return text


def _strip_known_prefixes(label):
    return _KNOWN_CHANNEL_PREFIX_PATTERN.sub("", label)


def _strip_reference_suffix(label):
    upper = label.upper()
    for suffix in _REFERENCE_SUFFIXES:
        if upper.endswith(suffix):
            base = label[: -len(suffix)].rstrip(" -_")
            if base:
                return base
    return label


def _extract_conventional_token(name):
    label = _normalize_channel_text(name)
    if not label:
        return ""
    label = _strip_known_prefixes(label)
    candidates = [label]
    if " " in label:
        candidates.append(label.split(" ")[-1])
    for candidate in candidates:
        token = _strip_reference_suffix(candidate)
        token_upper = token.upper()
        if token_upper in _CONVENTIONAL_LOOKUP or token_upper in _ALIAS_LOOKUP:
            return token_upper
    return ""


def clean_channel_name(name):
    label = _normalize_channel_text(name)
    if not label:
        return ""
    label = _strip_known_prefixes(label)
    token_upper = _extract_conventional_token(name)
    if token_upper:
        return _EEG_TOKEN_LOOKUP.get(token_upper, token_upper)
    return _strip_reference_suffix(label)


def canonicalize_channel_name(name):
    token_upper = _extract_conventional_token(name)
    if not token_upper:
        return ""
    canonical_upper = _ALIAS_LOOKUP.get(token_upper, token_upper)
    return _CONVENTIONAL_LOOKUP.get(canonical_upper, "")


def describe_channel_name(name):
    raw_name = str(name or "")
    clean_name = clean_channel_name(raw_name)
    canonical_name = canonicalize_channel_name(raw_name)
    return {
        "raw_name": raw_name,
        "clean_name": clean_name,
        "canonical_name": canonical_name,
        "is_recognized": bool(canonical_name),
        "uses_alias": bool(canonical_name and clean_name and clean_name.upper() != canonical_name.upper()),
    }


def build_channel_name_index(available_channels):
    channels = []
    canonical_sources = {}
    priorities = {}
    for input_order, channel_name in enumerate(available_channels):
        channel_info = describe_channel_name(channel_name)
        channel_info["input_order"] = input_order
        channels.append(channel_info)
        canonical_name = channel_info["canonical_name"]
        if not canonical_name:
            continue
        priority = 1 if channel_info["clean_name"].upper() == canonical_name.upper() else 0
        if canonical_name not in canonical_sources or priority > priorities[canonical_name]:
            canonical_sources[canonical_name] = channel_info
            priorities[canonical_name] = priority
    return {
        "channels": channels,
        "canonical_sources": canonical_sources,
        "source_mapping": {
            canonical_name: channel_info["raw_name"]
            for canonical_name, channel_info in canonical_sources.items()
        },
        "unmatched_channels": [channel_info for channel_info in channels if not channel_info["canonical_name"]],
    }


def _parse_numbered_contact(name):
    raw_name = str(name or "").strip()
    if not raw_name or is_bipolar_channel(raw_name):
        return None
    normalized = _strip_known_prefixes(_normalize_channel_text(raw_name))
    match = _NUMBERED_CHANNEL_PATTERN.match(normalized)
    if match is None:
        return None
    stem = re.sub(r"\s+", " ", match.group("stem").strip())
    suffix = re.sub(r"\s+", " ", (match.group("suffix") or "").strip()).lower()
    suffix = suffix.lstrip(" -_")
    return {
        "raw_name": raw_name,
        "stem": stem,
        "stem_key": stem.casefold(),
        "number": int(match.group("number")),
        "suffix_key": suffix,
    }


def normalize_channel_dict(raw):
    normalized = {}
    source_index = build_channel_name_index(raw.keys())
    for canonical_label, channel_info in source_index["canonical_sources"].items():
        normalized[canonical_label] = raw[channel_info["raw_name"]]
    return normalized


def _resolve_conventional_channel_sources(available_channels):
    source_index = build_channel_name_index(available_channels)
    return {
        canonical_name: channel_info["raw_name"]
        for canonical_name, channel_info in source_index["canonical_sources"].items()
    }


def get_conventional_eeg_neighbor_channels(available_channels, target_channel):
    source_index = build_channel_name_index(available_channels)
    target_info = describe_channel_name(target_channel)
    canonical_target = target_info["canonical_name"]
    if not canonical_target:
        return []
    target_source = source_index["canonical_sources"].get(canonical_target)
    if target_source is None:
        return []
    neighbors = []
    for canonical_neighbor in _CONVENTIONAL_NEIGHBOR_LOOKUP.get(canonical_target, []):
        source_info = source_index["canonical_sources"].get(canonical_neighbor)
        if source_info is not None:
            neighbors.append(source_info["raw_name"])
    return neighbors


def get_adjacent_contact_neighbor_channels(available_channels, target_channel, radius=1):
    try:
        radius = max(1, int(radius))
    except (TypeError, ValueError):
        radius = 1
    target_info = _parse_numbered_contact(target_channel)
    if target_info is None:
        return []
    grouped = []
    for channel_name in source_channel_names(available_channels):
        channel_info = _parse_numbered_contact(channel_name)
        if channel_info is None:
            continue
        if channel_info["stem_key"] != target_info["stem_key"] or channel_info["suffix_key"] != target_info["suffix_key"]:
            continue
        grouped.append(channel_info)
    grouped.sort(key=lambda entry: (entry["number"], entry["raw_name"]))
    return [
        entry["raw_name"]
        for entry in grouped
        if entry["raw_name"] != target_info["raw_name"]
        and abs(entry["number"] - target_info["number"]) <= radius
    ]


def _build_pair_entry(
    chain_name,
    display_order,
    channel_1,
    channel_2,
    source_info_1,
    source_info_2,
):
    source_channel_1 = source_info_1["raw_name"] if source_info_1 else None
    source_channel_2 = source_info_2["raw_name"] if source_info_2 else None
    missing_channels = [channel for channel, source_info in ((channel_1, source_info_1), (channel_2, source_info_2)) if source_info is None]
    return {
        "display_name": f"{channel_1}-{channel_2}",
        "derived_name": bipolar_channel_name(channel_1, channel_2),
        "chain_name": chain_name,
        "display_order": display_order,
        "channel_1": channel_1,
        "channel_2": channel_2,
        "source_channel_1": source_channel_1,
        "source_channel_2": source_channel_2,
        "source_clean_1": source_info_1["clean_name"] if source_info_1 else "",
        "source_clean_2": source_info_2["clean_name"] if source_info_2 else "",
        "source_mapping": {
            channel_1: source_channel_1,
            channel_2: source_channel_2,
        },
        "missing_channels": missing_channels,
        "uses_alias": bool(
            (source_info_1 and source_info_1["uses_alias"])
            or (source_info_2 and source_info_2["uses_alias"])
        ),
        "available": bool(source_info_1 and source_info_2),
    }


def get_conventional_bipolar_metadata(available_channels):
    channel_index = build_channel_name_index(available_channels)
    canonical_sources = channel_index["canonical_sources"]
    present_pairs = []
    missing_pairs = []
    chain_summaries = []
    chain_breaks = []
    expected_display_order = []
    display_order = 0

    for chain_name, pairs in CONVENTIONAL_BIPOLAR_CHAINS:
        chain_entries = []
        for channel_1, channel_2 in pairs:
            source_info_1 = canonical_sources.get(channel_1)
            source_info_2 = canonical_sources.get(channel_2)
            entry = _build_pair_entry(
                chain_name,
                display_order,
                channel_1,
                channel_2,
                source_info_1,
                source_info_2,
            )
            expected_display_order.append(entry["display_name"])
            chain_entries.append(entry)
            if entry["available"]:
                present_pairs.append(entry)
            else:
                missing_pairs.append(entry)
            display_order += 1

        chain_present = [entry["display_name"] for entry in chain_entries if entry["available"]]
        chain_missing = [entry["display_name"] for entry in chain_entries if not entry["available"]]
        chain_summary = {
            "chain_name": chain_name,
            "display_names": [entry["display_name"] for entry in chain_entries],
            "present_pairs": chain_present,
            "missing_pairs": chain_missing,
            "is_visible": bool(chain_present),
            "is_complete": not chain_missing and bool(chain_present),
            "is_partial": bool(chain_present and chain_missing),
        }
        chain_summaries.append(chain_summary)
        if chain_summary["is_partial"]:
            chain_breaks.append(
                {
                    "chain_name": chain_name,
                    "missing_pairs": chain_missing,
                }
            )

    warnings = []
    if channel_index["unmatched_channels"]:
        warnings.append(
            {
                "type": "unrecognized_channels",
                "channels": [channel_info["raw_name"] for channel_info in channel_index["unmatched_channels"]],
            }
        )
    if missing_pairs:
        warnings.append(
            {
                "type": "missing_pairs",
                "pairs": [entry["display_name"] for entry in missing_pairs],
            }
        )

    return {
        "montage_kind": "conventional_eeg",
        "montage_name": "double_banana",
        "normalized_channels": channel_index["channels"],
        "source_mapping": channel_index["source_mapping"],
        "present_pairs": present_pairs,
        "missing_pairs": missing_pairs,
        "expected_display_order": expected_display_order,
        "present_display_order": [entry["display_name"] for entry in present_pairs],
        "chain_summaries": chain_summaries,
        "chain_breaks": chain_breaks,
        "warnings": warnings,
    }


def get_conventional_bipolar_definitions(available_channels):
    metadata = get_conventional_bipolar_metadata(available_channels)
    return [
        (entry["display_name"], entry["channel_1"], entry["channel_2"])
        for entry in metadata["present_pairs"]
    ]


def get_conventional_bipolar_source_mappings(available_channels):
    metadata = get_conventional_bipolar_metadata(available_channels)
    return [
        (entry["display_name"], entry["source_channel_1"], entry["source_channel_2"])
        for entry in metadata["present_pairs"]
    ]


def apply_conventional_bipolar_montage(raw):
    output = {}
    for display_name, source_1, source_2 in get_conventional_bipolar_source_mappings(raw.keys()):
        left_signal = raw[source_1]
        right_signal = raw[source_2]
        if getattr(left_signal, "shape", None) != getattr(right_signal, "shape", None):
            raise ValueError(f"Shape mismatch between {source_1} and {source_2}")
        output[display_name] = left_signal - right_signal
    return output


def get_conventional_bipolar_auto_entries(available_channels):
    metadata = get_conventional_bipolar_metadata(available_channels)
    return [
        (entry["derived_name"], entry["source_channel_1"], entry["source_channel_2"])
        for entry in metadata["present_pairs"]
    ]


def get_average_reference_metadata(channel_names):
    recording_channels = [str(channel) for channel in source_channel_names(channel_names)]
    present_pairs = []
    normalized_channels = []
    source_mapping = {}
    for display_order, channel_name in enumerate(recording_channels):
        channel_info = describe_channel_name(channel_name)
        display_name = channel_info["clean_name"] or channel_name
        derived_name = average_reference_channel_name(channel_name)
        normalized_channels.append(channel_info)
        source_mapping[channel_name] = channel_name
        present_pairs.append(
            {
                "display_name": display_name,
                "derived_name": derived_name,
                "chain_name": "average_reference",
                "display_order": display_order,
                "channel_1": channel_name,
                "channel_2": AVERAGE_REFERENCE_SUFFIX,
                "source_channel_1": channel_name,
                "source_channel_2": AVERAGE_REFERENCE_LABEL,
                "source_clean_1": channel_info["clean_name"],
                "source_clean_2": AVERAGE_REFERENCE_LABEL,
                "source_mapping": {
                    channel_name: channel_name,
                },
                "missing_channels": [],
                "uses_alias": False,
                "available": True,
                "reference_label": AVERAGE_REFERENCE_LABEL,
            }
        )
    return {
        "montage_kind": "average_reference",
        "montage_name": "average_reference",
        "normalized_channels": normalized_channels,
        "source_mapping": source_mapping,
        "present_pairs": present_pairs,
        "missing_pairs": [],
        "expected_display_order": [entry["display_name"] for entry in present_pairs],
        "present_display_order": [entry["display_name"] for entry in present_pairs],
        "chain_summaries": [
            {
                "chain_name": "average_reference",
                "display_names": [entry["display_name"] for entry in present_pairs],
                "present_pairs": [entry["display_name"] for entry in present_pairs],
                "missing_pairs": [],
                "is_visible": bool(present_pairs),
                "is_complete": bool(present_pairs),
                "is_partial": False,
            }
        ] if present_pairs else [],
        "chain_breaks": [],
        "warnings": [],
    }


def get_average_reference_definitions(channel_names):
    metadata = get_average_reference_metadata(channel_names)
    return [
        (entry["derived_name"], entry["source_channel_1"])
        for entry in metadata["present_pairs"]
    ]


def infer_auto_bipolar_pairs(channel_names):
    groups = defaultdict(list)
    for channel in source_channel_names(channel_names):
        label = str(channel).strip()
        match = _NUMBERED_CHANNEL_PATTERN.match(label)
        if match is None:
            continue
        stem = re.sub(r"\s+", " ", match.group("stem").strip())
        suffix = re.sub(r"\s+", " ", (match.group("suffix") or "").strip()).lower()
        number = int(match.group("number"))
        groups[(stem.casefold(), suffix)].append((number, label))

    pairs = []
    for numbered_channels in groups.values():
        numbered_channels.sort(key=lambda item: (item[0], item[1]))
        for (left_number, left_label), (right_number, right_label) in zip(numbered_channels, numbered_channels[1:]):
            if right_number == left_number + 1:
                pairs.append((left_label, right_label))
    return pairs


def get_adjacent_contact_bipolar_metadata(channel_names):
    present_pairs = []
    for display_order, (channel_1, channel_2) in enumerate(infer_auto_bipolar_pairs(channel_names)):
        source_info_1 = describe_channel_name(channel_1)
        source_info_2 = describe_channel_name(channel_2)
        present_pairs.append(
            {
                "display_name": f"{channel_1}-{channel_2}",
                "derived_name": bipolar_channel_name(channel_1, channel_2),
                "chain_name": "adjacent_contacts",
                "display_order": display_order,
                "channel_1": channel_1,
                "channel_2": channel_2,
                "source_channel_1": channel_1,
                "source_channel_2": channel_2,
                "source_clean_1": source_info_1["clean_name"],
                "source_clean_2": source_info_2["clean_name"],
                "source_mapping": {
                    channel_1: channel_1,
                    channel_2: channel_2,
                },
                "missing_channels": [],
                "uses_alias": False,
                "available": True,
            }
        )
    return {
        "montage_kind": "adjacent_contacts",
        "montage_name": "adjacent_contacts",
        "normalized_channels": [describe_channel_name(channel) for channel in source_channel_names(channel_names)],
        "source_mapping": {},
        "present_pairs": present_pairs,
        "missing_pairs": [],
        "expected_display_order": [entry["display_name"] for entry in present_pairs],
        "present_display_order": [entry["display_name"] for entry in present_pairs],
        "chain_summaries": [],
        "chain_breaks": [],
        "warnings": [],
    }


def infer_auto_bipolar_montage_metadata(channel_names):
    conventional_metadata = get_conventional_bipolar_metadata(channel_names)
    if len(conventional_metadata["present_pairs"]) >= _MIN_CONVENTIONAL_AUTO_ENTRIES:
        return conventional_metadata
    return get_adjacent_contact_bipolar_metadata(channel_names)


def infer_auto_bipolar_montage_entries(channel_names):
    metadata = infer_auto_bipolar_montage_metadata(channel_names)
    return [
        (entry["derived_name"], entry["source_channel_1"], entry["source_channel_2"])
        for entry in metadata["present_pairs"]
    ]


def infer_auto_bipolar_channel_names(channel_names):
    return [derived_name for derived_name, _channel_1, _channel_2 in infer_auto_bipolar_montage_entries(channel_names)]
