from omnilingual.stt.langs import to_bcp47


def test_iso_codes_map_to_bcp47():
    assert to_bcp47("hi") == "hi-IN"
    assert to_bcp47("en") == "en-IN"
    assert to_bcp47("ta") == "ta-IN"
    assert to_bcp47("ne") == "ne-IN"


def test_full_lowercase_names_map():
    assert to_bcp47("hindi") == "hi-IN"
    assert to_bcp47("english") == "en-IN"
    assert to_bcp47("bengali") == "bn-IN"


def test_odia_iso_maps_to_sarvam_od():
    assert to_bcp47("or") == "od-IN"
    assert to_bcp47("odia") == "od-IN"


def test_unmapped_passes_through_and_empty_is_unknown():
    assert to_bcp47("es") == "es"
    assert to_bcp47("") == "unknown"
    assert to_bcp47("  ") == "unknown"
