import mne
import numpy as np

from src.hfo_app import HFO_App
from src.utils.utils_montage import (
    apply_conventional_bipolar_montage,
    get_average_reference_definitions,
    get_average_reference_metadata,
    canonicalize_channel_name,
    clean_channel_name,
    describe_channel_name,
    get_adjacent_contact_neighbor_channels,
    get_conventional_bipolar_metadata,
    get_conventional_bipolar_definitions,
    get_conventional_eeg_neighbor_channels,
    get_conventional_bipolar_source_mappings,
    infer_auto_bipolar_montage_metadata,
    infer_auto_bipolar_montage_entries,
)


def _build_conventional_raw(channel_names, samples=8):
    sfreq = 200.0
    data = []
    for channel_index in range(len(channel_names)):
        data.append((np.arange(samples, dtype=float) + channel_index * 10.0) * 1e-6)
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.array(data), info, verbose=False)
    raw.info["line_freq"] = 60
    return raw


def test_conventional_bipolar_definitions_cover_the_full_double_banana_order():
    available_channels = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "O1",
        "F7",
        "T3",
        "T5",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "O2",
        "F8",
        "T4",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]

    definitions = get_conventional_bipolar_definitions(available_channels)

    assert definitions == [
        ("Fp1-F3", "Fp1", "F3"),
        ("F3-C3", "F3", "C3"),
        ("C3-P3", "C3", "P3"),
        ("P3-O1", "P3", "O1"),
        ("Fp1-F7", "Fp1", "F7"),
        ("F7-T3", "F7", "T3"),
        ("T3-T5", "T3", "T5"),
        ("T5-O1", "T5", "O1"),
        ("Fp2-F4", "Fp2", "F4"),
        ("F4-C4", "F4", "C4"),
        ("C4-P4", "C4", "P4"),
        ("P4-O2", "P4", "O2"),
        ("Fp2-F8", "Fp2", "F8"),
        ("F8-T4", "F8", "T4"),
        ("T4-T6", "T4", "T6"),
        ("T6-O2", "T6", "O2"),
        ("Fz-Cz", "Fz", "Cz"),
        ("Cz-Pz", "Cz", "Pz"),
    ]


def test_conventional_bipolar_aliases_keep_canonical_labels_but_use_source_aliases():
    available_channels = ["Fp1", "F7", "T7", "P7", "O1", "Fp2", "F8", "T8", "P8", "O2", "Fz", "Cz", "Pz"]

    assert canonicalize_channel_name("T7") == "T3"
    assert canonicalize_channel_name("P7") == "T5"
    assert canonicalize_channel_name("EEG T8-Ref") == "T4"

    mappings = get_conventional_bipolar_source_mappings(available_channels)

    assert ("F7-T3", "F7", "T7") in mappings
    assert ("T3-T5", "T7", "P7") in mappings
    assert ("F8-T4", "F8", "T8") in mappings
    assert ("T4-T6", "T8", "P8") in mappings


def test_channel_name_cleaning_handles_common_prefixes_suffixes_and_aliases():
    assert clean_channel_name(" EEG FP1-REF ") == "Fp1"
    assert clean_channel_name("POL T7-M2") == "T7"
    assert clean_channel_name("SEEG Cz_avg") == "Cz"
    assert describe_channel_name("POL T7-M2") == {
        "raw_name": "POL T7-M2",
        "clean_name": "T7",
        "canonical_name": "T3",
        "is_recognized": True,
        "uses_alias": True,
    }


def test_conventional_bipolar_skips_pairs_with_missing_channels():
    available_channels = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5"]

    definitions = get_conventional_bipolar_definitions(available_channels)
    labels = [label for label, _channel_1, _channel_2 in definitions]

    assert "P3-O1" not in labels
    assert "T5-O1" not in labels
    assert "Fp1-F3" in labels
    assert "F7-T3" in labels


def test_conventional_bipolar_metadata_tracks_source_mapping_missing_pairs_and_chain_breaks():
    available_channels = [
        "EEG Fp1-Ref",
        "F3-LE",
        "C3",
        "P3",
        "F7",
        "T7",
        "P7",
        "POL X1",
    ]

    metadata = get_conventional_bipolar_metadata(available_channels)

    assert metadata["source_mapping"]["Fp1"] == "EEG Fp1-Ref"
    assert metadata["source_mapping"]["T3"] == "T7"
    assert metadata["source_mapping"]["T5"] == "P7"
    assert metadata["present_display_order"] == [
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
    ]
    assert "P3-O1" in [entry["display_name"] for entry in metadata["missing_pairs"]]
    assert "T5-O1" in [entry["display_name"] for entry in metadata["missing_pairs"]]
    assert metadata["chain_breaks"] == [
        {"chain_name": "left_parasagittal", "missing_pairs": ["P3-O1"]},
        {"chain_name": "left_temporal", "missing_pairs": ["T5-O1"]},
    ]
    assert metadata["warnings"] == [
        {"type": "unrecognized_channels", "channels": ["POL X1"]},
        {
            "type": "missing_pairs",
            "pairs": [
                "P3-O1",
                "T5-O1",
                "Fp2-F4",
                "F4-C4",
                "C4-P4",
                "P4-O2",
                "Fp2-F8",
                "F8-T4",
                "T4-T6",
                "T6-O2",
                "Fz-Cz",
                "Cz-Pz",
            ],
        },
    ]


def test_conventional_eeg_neighbor_channels_use_canonical_adjacency_and_raw_alias_sources():
    available_channels = [
        "EEG Fp1-Ref",
        "F3-LE",
        "F7",
        "T7",
        "P7",
        "O1",
    ]

    assert get_conventional_eeg_neighbor_channels(available_channels, "EEG Fp1-Ref") == ["F3-LE", "F7"]
    assert get_conventional_eeg_neighbor_channels(available_channels, "T7") == ["F7", "P7"]


def test_adjacent_contact_neighbor_channels_find_numbered_neighbors_with_prefix_cleanup():
    available_channels = ["SEEG A1-Ref", "A2Ref", "A3Ref", "B1Ref"]

    assert get_adjacent_contact_neighbor_channels(available_channels, "A2Ref") == ["SEEG A1-Ref", "A3Ref"]
    assert get_adjacent_contact_neighbor_channels(available_channels, "B1Ref") == []


def test_apply_conventional_bipolar_montage_raises_on_shape_mismatch():
    raw = {
        "Fp1": np.array([1.0, 2.0, 3.0]),
        "F3": np.array([1.0, 2.0]),
    }

    try:
        apply_conventional_bipolar_montage(raw)
        raise AssertionError("Expected a ValueError for mismatched channel lengths")
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)


def test_apply_conventional_bipolar_montage_uses_alias_channels_and_preserves_order():
    raw = {
        "Fp1": np.array([10.0, 11.0, 12.0]),
        "F7": np.array([6.0, 6.0, 6.0]),
        "T7": np.array([1.0, 2.0, 3.0]),
        "P7": np.array([0.5, 1.0, 1.5]),
        "O1": np.array([0.0, 0.0, 0.0]),
        "Fz": np.array([3.0, 3.0, 3.0]),
        "Cz": np.array([1.0, 1.0, 1.0]),
        "Pz": np.array([0.0, 0.0, 0.0]),
    }

    mounted = apply_conventional_bipolar_montage(raw)

    assert list(mounted) == ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fz-Cz", "Cz-Pz"]
    np.testing.assert_array_equal(mounted["F7-T3"], raw["F7"] - raw["T7"])
    np.testing.assert_array_equal(mounted["T3-T5"], raw["T7"] - raw["P7"])


def test_auto_bipolar_entries_prefer_conventional_eeg_before_numbered_ieeg_pairs():
    available_channels = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T7", "P7"]

    entries = infer_auto_bipolar_montage_entries(available_channels)

    assert entries[:4] == [
        ("Fp1#-#F3", "Fp1", "F3"),
        ("F3#-#C3", "F3", "C3"),
        ("C3#-#P3", "C3", "P3"),
        ("P3#-#O1", "P3", "O1"),
    ]
    assert ("F7#-#T3", "F7", "T7") in entries
    assert ("T3#-#T5", "T7", "P7") in entries


def test_auto_bipolar_entries_keep_numbered_ieeg_fallback_when_only_sparse_scalp_labels_exist():
    available_channels = ["Fz", "Cz", "Pz", "A1Ref", "A2Ref", "A3Ref"]

    entries = infer_auto_bipolar_montage_entries(available_channels)
    metadata = infer_auto_bipolar_montage_metadata(available_channels)

    assert entries == [
        ("A1Ref#-#A2Ref", "A1Ref", "A2Ref"),
        ("A2Ref#-#A3Ref", "A2Ref", "A3Ref"),
    ]
    assert metadata["montage_kind"] == "adjacent_contacts"


def test_average_reference_metadata_uses_source_channel_order_and_clean_display_labels():
    available_channels = ["EEG Fp1-Ref", "POL T7-M2", "A1Ref"]

    metadata = get_average_reference_metadata(available_channels)
    definitions = get_average_reference_definitions(available_channels)

    assert [entry["display_name"] for entry in metadata["present_pairs"]] == ["Fp1", "T7", "A1"]
    assert definitions == [
        ("EEG Fp1-Ref#-#AVG", "EEG Fp1-Ref"),
        ("POL T7-M2#-#AVG", "POL T7-M2"),
        ("A1Ref#-#AVG", "A1Ref"),
    ]
    assert metadata["montage_kind"] == "average_reference"
    assert metadata["chain_breaks"] == []


def test_hfo_backend_auto_bipolar_uses_canonical_double_banana_names_with_alias_sources():
    raw = _build_conventional_raw(
        [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "O1",
            "F7",
            "T7",
            "P7",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "O2",
            "F8",
            "T8",
            "P8",
            "Fz",
            "Cz",
            "Pz",
        ]
    )
    app = HFO_App()
    app.load_raw(raw, file_path="<double-banana>")

    derived_channels = app.ensure_auto_bipolar_channels().tolist()

    assert derived_channels[:6] == [
        "Fp1#-#F3",
        "F3#-#C3",
        "C3#-#P3",
        "P3#-#O1",
        "Fp1#-#F7",
        "F7#-#T3",
    ]
    assert "T3#-#T5" in derived_channels
    assert "F8#-#T4" in derived_channels
    assert "T4#-#T6" in derived_channels
    assert len(derived_channels) == 18
    assert app.get_auto_bipolar_metadata()["montage_kind"] == "conventional_eeg"
    assert app.get_auto_bipolar_metadata()["chain_breaks"] == []

    derived_index = np.where(app.channel_names == "F7#-#T3")[0][0]
    f7_index = np.where(app.channel_names == "F7")[0][0]
    t7_index = np.where(app.channel_names == "T7")[0][0]
    np.testing.assert_allclose(app.eeg_data[derived_index], app.eeg_data[f7_index] - app.eeg_data[t7_index])


def test_hfo_backend_clean_channel_metadata_uses_bads_and_flat_channels():
    raw = _build_conventional_raw(["Fp1", "F3", "C3"], samples=8)
    raw.info["bads"] = ["F3"]
    raw._data[2] = 0.0

    app = HFO_App()
    app.load_raw(raw, file_path="<clean-view>")

    metadata = app.get_clean_recording_channel_metadata()
    recording_channels = app.get_recording_channel_names().tolist()

    assert metadata["recording_channels"] == recording_channels
    assert metadata["bad_channels"] == ["F3"]
    assert metadata["flat_channels"] == ["C3"]
    assert metadata["excluded_channels"] == [
        channel for channel in recording_channels if channel in {"F3", "C3"}
    ]
    assert metadata["clean_channels"] == ["Fp1"]


def test_hfo_backend_average_reference_channels_follow_source_order_and_average_signal():
    raw = _build_conventional_raw(["Fp1", "F3", "C3"], samples=8)
    app = HFO_App()
    app.load_raw(raw, file_path="<avg-ref>")

    derived_channels = app.ensure_average_reference_channels().tolist()
    recording_channels = [str(channel) for channel in app.get_recording_channel_names().tolist()]

    assert derived_channels == [f"{channel}#-#AVG" for channel in recording_channels]
    assert app.get_average_reference_metadata()["montage_kind"] == "average_reference"

    derived_index = np.where(app.channel_names == "Fp1#-#AVG")[0][0]
    fp1_index = np.where(app.channel_names == "Fp1")[0][0]
    f3_index = np.where(app.channel_names == "F3")[0][0]
    c3_index = np.where(app.channel_names == "C3")[0][0]
    average_signal = (app.eeg_data[fp1_index] + app.eeg_data[f3_index] + app.eeg_data[c3_index]) / 3.0
    np.testing.assert_allclose(app.eeg_data[derived_index], app.eeg_data[fp1_index] - average_signal)
