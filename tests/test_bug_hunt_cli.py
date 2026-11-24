"""Regression tests for agent_cli bug-hunt fixes (#1 list params, #13 bad JSON)."""

from pyhfo import agent_cli


# --- Bug #1: a JSON-encoded list param round-trips back to a real list --------

def test_parse_key_values_round_trips_list_param():
    parsed = agent_cli._parse_key_values(["freq_sp=[11, 16]", "thresh=0.5"])
    assert parsed["freq_sp"] == [11, 16]
    assert parsed["thresh"] == 0.5
    # the detector init that used to crash: tuple(float(v) for v in freq_sp)
    assert tuple(float(v) for v in parsed["freq_sp"]) == (11.0, 16.0)


def test_parse_scalar_leaves_plain_strings_and_scalars_alone():
    assert agent_cli._parse_scalar("STE") == "STE"
    assert agent_cli._parse_scalar("5") == 5
    assert agent_cli._parse_scalar("0.3") == 0.3
    assert agent_cli._parse_scalar("true") is True
    assert agent_cli._parse_scalar("none") is None


# --- Bug #13: malformed --bipolar-pairs-json -> structured invalid_argument ---

def test_prepare_malformed_bipolar_pairs_json_is_user_error(monkeypatch, capsys):
    # main() should map the parse failure to exit code 2 (invalid_argument / user input),
    # not 1 (internal error) with a raw JSONDecodeError.
    code = agent_cli.main([
        "prepare", "--input", "a.fif", "--output", "b.fif",
        "--bipolar-pairs-json", "[{bad json",
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "invalid_argument" in err
    assert "bipolar-pairs-json" in err
