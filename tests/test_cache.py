from omnilingual.cache import JsonCache, mt_key, stt_key


def test_miss_then_hit(tmp_path):
    cache = JsonCache(tmp_path)
    assert cache.get("stt", "abc") is None
    cache.put("stt", "abc", {"text": "hi"})
    assert cache.get("stt", "abc") == {"text": "hi"}
    assert (tmp_path / "stt" / "abc.json").exists()


def test_namespaces_are_isolated(tmp_path):
    cache = JsonCache(tmp_path)
    cache.put("stt", "k", {"a": 1})
    assert cache.get("mt", "k") is None


def test_stt_key_changes_with_model_and_mode():
    b = b"\x00\x01"
    k1 = stt_key(b, "saaras:v4", "transcribe")
    assert len(k1) == 64
    assert k1 != stt_key(b, "saaras:v3", "transcribe")
    assert k1 != stt_key(b, "saaras:v4", "translate")
    assert k1 == stt_key(b, "saaras:v4", "transcribe")


def test_mt_key_changes_with_text_and_langs():
    k = mt_key("नमस्ते", "hi-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते!", "hi-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते", "mr-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते", "hi-IN", "en-IN", "sarvam-translate:v1")


def test_put_overwrites(tmp_path):
    cache = JsonCache(tmp_path)
    cache.put("stt", "k", {"v": 1})
    cache.put("stt", "k", {"v": 2})
    assert cache.get("stt", "k") == {"v": 2}


def test_malformed_json_is_treated_as_miss(tmp_path):
    cache = JsonCache(tmp_path)
    p = tmp_path / "stt" / "broken.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert cache.get("stt", "broken") is None
