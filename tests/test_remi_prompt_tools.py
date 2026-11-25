"""Prompt tools: the tag index and its ranking, completion sources, the
history / favorites store, translation providers and the
``/api/v2/prompt-tools`` routes (remi/prompt_tools)."""

import json
import sqlite3
import threading
import time
from contextlib import closing

import pytest

from remi import prompt_tools as pt
from remi.prompt_tools import tags as T
from remi.prompt_tools.sources import ChantIndex, scan_wildcards, wildcard_values
from remi.prompt_tools.store import PromptToolsStore, StoreError
from remi.prompt_tools.translate import ProviderConfig, TranslationError, dictionary_translate, translate

TAG_CSV = """1girl,0,6000000,"1girls,sole_female"
long_hair,0,4300000,"longhair,/lh"
breasts,0,3400000,"oppai,boobs"
hair_ornament,0,900000,
very_long_hair,0,800000,
school_uniform,0,500000,"seifuku"
uniform,0,100000,
detached_collar,0,90000,
hair_between_eyes,0,850000,
masterpiece,5,1000,
hatsune_miku,4,200000,"miku"
"""


def entries(text=TAG_CSV):
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(",", 3)
        rows.append(T.TagEntry(parts[0], int(parts[1]), int(parts[2]), T._aliases(parts[3].strip('"') if len(parts) > 3 else "")))
    return rows


@pytest.fixture
def index():
    return T.TagIndex(entries(), [T.TagEntry("masterpiece", 5, 0, (), extra=True, meta="Quality tag")], {"long_hair": "长发", "1girl": "1女孩"})


def names(matches):
    return [match.entry.name for match in matches]


def search(index, query, limit=8, **kwargs):
    return T.rank(index.matches(query, limit) + index.extra_matches(query), limit=limit, **kwargs)


# region Tag index ranking


def test_exact_beats_prefix_beats_word_prefix(index):
    ranked = search(index, "hair")
    assert names(ranked)[:2] == ["hair_ornament", "hair_between_eyes"]  # prefix, by posts
    assert names(ranked)[2:4] == ["long_hair", "very_long_hair"]  # word prefixes, by posts
    assert names(search(index, "uniform"))[0] == "uniform"  # exact first, despite fewer posts


def test_spaces_and_escapes_are_normalized(index):
    assert names(search(index, "long ha"))[0] == "long_hair"
    assert T.normalize_query("  Long\\ Hair  ") == "long_hair"


def test_aliases_find_their_tag_and_report_the_alias(index):
    match = search(index, "oppai")[0]
    assert (match.entry.name, match.tier, match.alias) == ("breasts", T.Tier.ALIAS_EXACT, "oppai")
    assert "breasts" not in names(index.matches("oppai", 8, aliases=False))


def test_translations_are_searchable(index):
    assert names(search(index, "长发")) == ["long_hair"]
    assert index.translation("Long Hair") == "长发"
    assert names(index.matches("长发", 8, translations=False)) == []


def test_substring_and_abbreviation_fallbacks(index):
    assert names(search(index, "*hair")) == ["long_hair", "hair_ornament", "hair_between_eyes", "very_long_hair"]
    fuzzy = search(index, "detco")
    assert names(fuzzy) == ["detached_collar"] and fuzzy[0].tier == T.Tier.FUZZY
    assert T.abbreviation_pattern("lh").search("long_hair") and not T.abbreviation_pattern("lh").search("blonde")


def test_abbreviations_stay_within_one_tag():
    # each query's letters are spread over consecutive tags of the list, which
    # a match across line breaks of the newline-joined blob would accept
    tags = ["wooden_door", "lamp", "open_mouth", "parted_lips", "double_exposure", "tie", "comb", "detached_collar"]
    index = T.TagIndex([T.TagEntry(name, 0, 1000 - rank) for rank, name in enumerate(tags)])
    assert [m.entry.name for m in index.matches("wlop", 8) if m.tier == T.Tier.FUZZY] == []
    assert [m.entry.name for m in index.matches("detco", 8) if m.tier == T.Tier.FUZZY] == ["detached_collar"]
    assert not T.abbreviation_pattern("wl").search("wooden_door\nlamp")
    assert T.abbreviation_pattern("pl").search("parted_lips") and T.abbreviation_pattern("pl").search("parted lips")


def test_artist_prefix_is_not_searched(index):
    assert T.normalize_query("@Hatsune Mi") == "hatsune_mi"
    assert T.normalize_query("@_@") == "@_@"  # an emoticon tag, not a prefix
    assert names(search(index, "@hatsune"))[0] == "hatsune_miku"
    assert names(search(index, "@miku"))[0] == "hatsune_miku"  # by alias too


def test_extra_tags_come_first_or_last_and_replace_their_namesake(index):
    ranked = search(index, "mas")
    assert ranked[0].entry.extra and ranked[0].entry.meta == "Quality tag"
    after = T.rank(index.matches("ma", 8) + index.extra_matches("ma"), extras_first=False, limit=8)
    assert after[-1].entry.extra


def test_usage_promotes_within_limits(index):
    plain = names(search(index, "hair"))
    assert plain[0] == "hair_ornament"
    config = T.FrequencyConfig(enabled=True, function="Logarithmic (strong)", min_count=3, recommend_cap=10)
    promoted = names(search(index, "hair", uses={"hair_between_eyes": 50}, config=config))
    assert promoted[0] == "hair_between_eyes"
    # below the minimum count, or frequency sorting off: no change
    assert names(search(index, "hair", uses={"hair_between_eyes": 2}, config=config)) == plain
    assert names(search(index, "hair", uses={"hair_between_eyes": 50}, config=T.FrequencyConfig(enabled=False))) == plain
    # "Usage first" lists used tags before everything of their query
    first = T.FrequencyConfig(function="Usage first", min_count=1)
    assert names(search(index, "hair", uses={"very_long_hair": 1}, config=first))[0] == "very_long_hair"
    # the cap: only the most used candidates are promoted
    capped = T.FrequencyConfig(function="Usage first", min_count=1, recommend_cap=1)
    ranked = names(search(index, "hair", uses={"very_long_hair": 5, "long_hair": 4}, config=capped))
    assert ranked[0] == "very_long_hair" and ranked.index("long_hair") > ranked.index("hair_between_eyes")


def test_limit_and_lookup(index):
    assert len(search(index, "h", limit=2)) == 2
    assert index.lookup("Hatsune Miku").category == 4
    assert index.lookup("masterpiece") is not None and index.lookup("nope") is None


def test_files_and_formats(tmp_path):
    (tmp_path / "tags.csv").write_text(TAG_CSV, encoding="utf8")
    (tmp_path / "extra.csv").write_text("masterpiece,5,Quality tag,,杰作\nbest_quality,5,,,\n", encoding="utf8")
    (tmp_path / "old.csv").write_text("long_hair,0,长发\n", encoding="utf8")
    (tmp_path / "new.csv").write_text("long_hair,长头发\n1girl,\n", encoding="utf8")
    extras, translations = T.read_extra_file(str(tmp_path / "extra.csv"))
    assert [(e.name, e.meta) for e in extras] == [("masterpiece", "Quality tag"), ("best_quality", None)] and translations == {"masterpiece": "杰作"}
    assert T.read_translation_file(str(tmp_path / "old.csv")) == {"long_hair": "长发"}
    assert T.read_translation_file(str(tmp_path / "new.csv")) == {"long_hair": "长头发"}
    files = T.TagFiles(str(tmp_path / "tags.csv"), str(tmp_path / "extra.csv"), str(tmp_path / "new.csv"))
    cache = T.IndexCache()
    first = cache.get(files)
    assert len(first) == 11 and first.translation("masterpiece") == "杰作" and first.translation("long_hair") == "长头发"
    assert cache.get(files) is first
    (tmp_path / "tags.csv").write_text(TAG_CSV + "new_tag,0,1,\n", encoding="utf8")
    assert len(cache.get(files)) == 12


def test_a_tag_file_that_is_not_utf8_is_left_out_not_fatal(tmp_path):
    (tmp_path / "tags.csv").write_text(TAG_CSV, encoding="utf8")
    (tmp_path / "gbk.csv").write_bytes("long_hair,长发\n".encode("gbk"))
    files = T.TagFiles(str(tmp_path / "tags.csv"), None, str(tmp_path / "gbk.csv"))
    index = T.IndexCache().get(files)
    assert len(index) == 11 and index.translation("long_hair") is None


def test_edited_extra_translations_rebuild_the_index(tmp_path):
    (tmp_path / "tags.csv").write_text(TAG_CSV, encoding="utf8")
    cache = T.IndexCache()
    old = T.TagFiles(str(tmp_path / "tags.csv"), extra_translations={"long_hair": "OLD"})
    assert cache.get(old).translation("long_hair") == "OLD"
    new = T.TagFiles(str(tmp_path / "tags.csv"), extra_translations={"long_hair": "NEW"})  # same length
    assert cache.get(new).translation("long_hair") == "NEW"
    assert cache.get(new) is cache.get(new)


def test_custom_tag_groups_are_read_again_only_when_they_change(tmp_path, monkeypatch):
    from remi.prompt_tools import groups as G

    (tmp_path / "custom.yaml").write_text("- name: Mine\n  groups:\n    - name: Set\n      tags:\n        1girl: one\n", encoding="utf8")
    reads = []
    real = G._read
    monkeypatch.setattr(G, "_read", lambda path: reads.append(path) or real(path))
    tag_groups = G.TagGroups([str(tmp_path)])
    assert tag_groups.dictionary("zh_CN") == {"1girl": "one"}
    count = len(reads)
    for _ in range(3):
        tag_groups.dictionary("zh_CN")
    assert len(reads) == count


def test_a_refresh_during_a_background_rescan_waits_for_a_new_scan(tmp_path, monkeypatch):
    from remi.prompt_tools import sources

    files, block = ["old"], []
    in_scan, release = threading.Event(), threading.Event()

    def scan(roots, excluded, yaml_cache):
        names = list(files)
        if block:
            block.clear()
            in_scan.set()
            release.wait(5)
        return [sources.Wildcard(name) for name in names]

    monkeypatch.setattr(sources, "scan_wildcards", scan)
    cache = sources.WildcardCache(lambda: [str(tmp_path)], ttl=3600, background=False)
    assert [item.name for item in cache.items()] == ["old"]
    block.append(True)
    rescan = threading.Thread(target=cache._rescan, args=(cache._key,))
    rescan.start()  # a background rescan, still reading the folder as it was
    assert in_scan.wait(5)
    files.append("new")  # a wildcard file is added, then Refresh
    cache.refresh()
    result = []
    reader = threading.Thread(target=lambda: result.append(cache.items()))
    reader.start()
    time.sleep(0.05)
    release.set()
    reader.join(5)
    rescan.join(5)
    assert [item.name for item in result[0]] == ["old", "new"]


# endregion

# region Wildcards and chants


def test_concurrent_first_requests_share_one_wildcard_scan(monkeypatch):
    from remi.prompt_tools import sources

    calls = []

    def scan(roots, excluded, yaml_cache):
        calls.append(roots)
        time.sleep(0.2)
        return []

    monkeypatch.setattr(sources, "scan_wildcards", scan)
    cache = sources.WildcardCache(lambda: ["/nonexistent"], ttl=3600, background=False)
    first = threading.Thread(target=cache.items)
    first.start()
    time.sleep(0.05)
    second = threading.Thread(target=cache.items)  # arrives during the first scan
    second.start()
    first.join(5)
    second.join(5)
    assert len(calls) == 1


def test_wildcards_nested_yaml_and_values(tmp_path):
    root = tmp_path / "wildcards"
    (root / "hair" / "colors").mkdir(parents=True)
    (root / "hair" / "colors" / "light.txt").write_text("# comment\nblonde hair\n\nsilver hair\n", encoding="utf8")
    (root / "animals.txt").write_text("cat\ndog\n", encoding="utf8")
    (root / "skip").mkdir()
    (root / "skip" / "hidden.txt").write_text("x", encoding="utf8")
    (root / "collection.yaml").write_text("styles:\n  painting: [oil, watercolor]\n  bad: {nested: 1}\n", encoding="utf8")
    found = scan_wildcards([str(root)], ["skip"])
    assert [w.name for w in found] == ["animals", "hair/colors/light", "styles/painting"]
    assert wildcard_values(found[1]) == ["blonde hair", "silver hair"]
    assert wildcard_values(found[2]) == ["oil", "watercolor"]


def test_wildcard_values_are_read_once_until_the_file_changes(tmp_path, monkeypatch):
    """Each completion of a wildcard's values asked for the whole file again."""

    from remi.prompt_tools import sources

    path = tmp_path / "animals.txt"
    path.write_text("cat\ndog\n", encoding="utf8")
    wildcard = sources.Wildcard("animals", path=str(path))
    opened = []

    def counting_open(file, *args, **kwargs):
        opened.append(file)
        return open(file, *args, **kwargs)

    monkeypatch.setattr(sources, "open", counting_open, raising=False)
    first = wildcard_values(wildcard)
    first.append("changed by the caller")
    assert wildcard_values(wildcard) == ["cat", "dog"] and len(opened) == 1
    path.write_text("cat\ndog\nfox\n", encoding="utf8")
    assert wildcard_values(wildcard) == ["cat", "dog", "fox"] and len(opened) == 2


def test_wildcard_scan_survives_symlink_loops_and_reuses_yaml(tmp_path, monkeypatch):
    from remi.prompt_tools import sources

    root = tmp_path / "wildcards"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.txt").write_text("one\n", encoding="utf8")
    (root / "c.yaml").write_text("c: [1, 2]\n", encoding="utf8")
    try:
        (root / "a" / "loop").symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    parsed = []
    original = sources._yaml_wildcards
    monkeypatch.setattr(sources, "_yaml_wildcards", lambda path, source: parsed.append(path) or original(path, source))
    cache = {}
    found = scan_wildcards([str(root)], yaml_cache=cache)
    assert [w.name for w in found] == ["a/x", "c"]
    assert [w.name for w in scan_wildcards([str(root)], yaml_cache=cache)] == ["a/x", "c"] and len(parsed) == 1
    (root / "c.yaml").write_text("c: [1, 2, 3]\n", encoding="utf8")
    assert wildcard_values(scan_wildcards([str(root)], yaml_cache=cache)[1]) == ["1", "2", "3"] and len(parsed) == 2


def test_wildcard_cache_rescans_in_the_background(tmp_path):
    from remi.prompt_tools.sources import WildcardCache

    root = tmp_path / "wildcards"
    root.mkdir()
    (root / "a.txt").write_text("x\n", encoding="utf8")
    cache = WildcardCache(lambda: [str(root)], ttl=0.0, background=True)
    assert [w.name for w in cache.items()] == ["a"]
    (root / "b.txt").write_text("y\n", encoding="utf8")
    assert [w.name for w in cache.items()] == ["a"]  # stale list at once, rescan started
    deadline = time.monotonic() + 5
    while [w.name for w in cache.items()] != ["a", "b"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [w.name for w in cache.items()] == ["a", "b"]
    (root / "c.txt").write_text("z\n", encoding="utf8")
    cache.refresh()
    assert [w.name for w in cache.items()] == ["a", "b", "c"]  # a refresh scans before answering


def test_chants_search_by_name_and_terms(tmp_path):
    path = tmp_path / "chants.json"
    path.write_text(json.dumps([{"name": "Basic-Start", "terms": "Basic, Start", "content": "(masterpiece), 1girl", "color": 3}, {"name": "Negative", "terms": "low quality", "content": "(worst quality)"}]), encoding="utf8")
    chants = ChantIndex()
    assert [c.name for c in chants.search(str(path), "basic", 5)] == ["Basic-Start"]
    assert [c.name for c in chants.search(str(path), "quality", 5)] == ["Negative"]
    assert len(chants.search(str(path), "", 5)) == 2


# endregion

# region Store


@pytest.fixture
def store(tmp_path):
    store = PromptToolsStore(tmp_path / "prompt-tools.db")
    yield store
    store.close()


def test_history_deduplicates_groups_and_searches(store):
    store.record_history("a cat", "blurry", "txt2img", when=1)
    store.record_history("a dog", "blurry", "txt2img", when=2)
    again = store.record_history("a cat", "blurry", "img2img", when=3)
    assert again["uses"] == 2 and again["tab"] == "img2img"
    items, cursor, total = store.history()
    assert [item["prompt"] for item in items] == ["a cat", "a dog"] and total == 2 and cursor is None
    negatives, _, total = store.history(field="negative")
    assert total == 1 and negatives[0]["negative_prompt"] == "blurry" and negatives[0]["uses"] == 3
    assert [item["prompt"] for item in store.history(query="DOG")[0]] == ["a dog"]
    assert store.history(query="a blurry")[2] == 2 and store.history(query="cat dog")[2] == 0
    assert store.record_history("", "  ") is None


def test_searches_fold_the_case_of_non_ascii_letters(store):
    store.record_history("Ärger, Émilie", "", when=1)
    store.add_favorite("Ωmega pose", "prompt")
    for query in ("ärger", "Ärger", "ÉMILIE"):
        assert [item["prompt"] for item in store.history(query=query)[0]] == ["Ärger, Émilie"], query
    assert [item["text"] for item in store.favorites_page(query="ωMEGA", all_collections=True)[0]] == ["Ωmega pose"]


def test_a_crafted_cursor_starts_at_the_first_page(store):
    # "²".isdigit() is true but int("²") raises: a 500 before
    import base64

    store.record_history("a cat", "", when=1)
    store.add_favorite("a pose", "prompt")
    cursor = base64.urlsafe_b64encode("²".encode()).decode().rstrip("=")
    assert [item["prompt"] for item in store.history(cursor=cursor)[0]] == ["a cat"]
    assert [item["text"] for item in store.favorites_page(cursor=cursor, all_collections=True)[0]] == ["a pose"]


def test_the_history_limit_counts_unnamed_entries_only(store):
    for number in range(4):
        store.record_history(f"plain {number}", "", when=number)
    for number in range(3):
        entry = store.record_history(f"named {number}", "", when=10 + number)
        store.rename_history(entry["id"], f"n{number}")
    store.record_history("plain 4", "", when=20, limit=3)
    kept = {item["prompt"] for item in store.history()[0] if not item["name"]}
    assert kept == {"plain 4", "plain 3", "plain 2"}


def test_grouped_history_rename_and_delete_cover_the_group(store):
    first = store.record_history("a cat", "blurry", when=1)
    store.record_history("a cat", "lowres", when=2)
    store.record_history("a dog", "lowres", when=3)
    [cat, dog] = sorted(store.history(field="prompt")[0], key=lambda item: item["prompt"])
    assert cat["uses"] == 2
    store.rename_history(first["id"], "kitty", field="prompt")
    assert {item["name"] for item in store.history()[0] if item["prompt"] == "a cat"} == {"kitty"}
    assert store.delete_history(first["id"], field="prompt") == 2
    assert [item["prompt"] for item in store.history(field="prompt")[0]] == ["a dog"]
    assert store.delete_history(dog["id"]) == 1 and store.delete_history(dog["id"]) == 0


def test_history_paging_limit_names_and_removal(store):
    for number in range(5):
        store.record_history(f"prompt {number}", "", when=number)
    page, cursor, total = store.history(limit=2)
    assert [item["prompt"] for item in page] == ["prompt 4", "prompt 3"] and total == 5
    assert [item["prompt"] for item in store.history(limit=2, cursor=cursor)[0]] == ["prompt 2", "prompt 1"]
    oldest = store.history(limit=5)[0][-1]
    store.rename_history(oldest["id"], "keeper")
    store.record_history("newest", "", when=10, limit=2)
    kept = {item["prompt"] for item in store.history()[0]}
    assert kept == {"newest", "prompt 4", "prompt 0"}  # the named one survives the limit
    assert store.clear_history() == 2 and [item["name"] for item in store.history()[0]] == ["keeper"]
    with pytest.raises(StoreError):
        store.rename_history("missing", "x")


def test_moving_favorites_into_the_same_gap_many_times_keeps_the_order(store):
    """Each move in front of the favorite just moved halves the same gap; after about 50 the
    positions tied and the order fell back to the creation time (the move did nothing)."""
    x, y, z = (store.add_favorite(text) for text in ("x", "y", "z"))
    store.update_favorite(x["id"], {"before": z["id"]})  # newest first (z, y, x), then x to the front
    order = lambda: [f["id"] for f in store.favorites()]  # noqa: E731
    assert order() == [x["id"], z["id"], y["id"]]
    for _ in range(80):
        store.update_favorite(y["id"], {"before": z["id"]})
        assert order() == [x["id"], y["id"], z["id"]]
        store.update_favorite(z["id"], {"before": y["id"]})
        assert order() == [x["id"], z["id"], y["id"]]


def test_favorites_collections_and_order(store):
    collection = store.create_collection("Portraits")
    with pytest.raises(StoreError):
        store.create_collection("Portraits")
    a = store.add_favorite("1girl, smile", name="smile", collection_id=collection["id"])
    b = store.add_favorite("1girl, frown", collection_id=collection["id"])
    loose = store.add_favorite("blurry", "negative")
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [b["id"], a["id"]]  # newest first
    store.update_favorite(b["id"], {"before": None})  # to the end
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [a["id"], b["id"]]
    store.update_favorite(b["id"], {"before": a["id"]})
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [b["id"], a["id"]]
    moved = store.update_favorite(loose["id"], {"collection_id": collection["id"], "name": "neg"})
    assert moved["collection_id"] == collection["id"] and moved["name"] == "neg"
    counts = {c["name"]: c["count"] for c in store.collections()}
    assert counts == {"": 0, "Portraits": 3}
    assert [f["text"] for f in store.favorites(all_collections=True, kind="negative")] == ["blurry"]
    exported = store.export_favorites()
    assert store.delete_collection(collection["id"]) and store.favorites(all_collections=True) == []
    assert store.import_favorites(exported) == 3 and store.import_favorites(exported) == 0
    with pytest.raises(StoreError):
        store.add_favorite("x", "nope")
    with pytest.raises(StoreError):
        store.update_favorite("missing", {"name": "x"})


def test_favorites_paging_kinds_and_full_export(store):
    for number in range(7):
        store.add_favorite(f"prompt {number}")
    store.add_favorite("blurry", "negative")
    store.add_favorite("smile", "tags")
    page, cursor, total = store.favorites_page(all_collections=True, kinds=["prompt", "tags"], limit=5)
    assert total == 8 and len(page) == 5 and cursor
    rest, cursor, _ = store.favorites_page(all_collections=True, kinds=["prompt", "tags"], limit=5, cursor=cursor)
    assert len(rest) == 3 and cursor is None and "blurry" not in {f["text"] for f in page + rest}
    for number in range(600):
        store.add_favorite(f"many {number}", collection_id=None)
    exported = store.export_favorites()["favorites"]
    assert len(exported) == 609  # nothing is cut off
    assert [f["text"] for f in exported[:2]] == ["many 599", "many 598"]  # in their order


def test_favorites_import_keeps_order_and_skips_bad_items(store):
    data = {"favorites": [{"text": "b", "collection": "C"}, {"text": "a", "collection": "C", "name": "n"}, {"text": 5}, "junk", {"text": "  "}, {"text": "b", "collection": "C"}, {"text": "x" * 70000}]}
    assert store.import_favorites(data) == 2
    [collection] = [c for c in store.collections() if c["id"]]
    assert [(f["text"], f["name"]) for f in store.favorites(collection_id=collection["id"])] == [("b", ""), ("a", "n")]
    with pytest.raises(StoreError):
        store.import_favorites({"nope": []})


def test_secrets_are_kept_apart(store):
    assert store.secret("translation_key") is None
    store.set_secret("translation_key", "abc")
    assert store.secret("translation_key") == "abc"
    store.set_secret("translation_key", None)
    assert store.secret("translation_key") is None


def test_usage_counts_age_and_reset(store):
    for _ in range(3):
        store.record_use("long_hair", "tag")
    store.record_use("long_hair", "tag", negative=True)
    store.record_use("old_tag", "tag", when=time.time() - 40 * 86400)
    assert store.uses(["long_hair", "old_tag", "x"], "tag") == {"long_hair": 3, "old_tag": 1}
    assert store.uses(["long_hair"], "tag", negative=True) == {"long_hair": 1}
    assert store.uses(["long_hair", "old_tag"], "tag", max_age_days=30) == {"long_hair": 3}
    assert store.uses(["long_hair"], "lora") == {}
    assert store.reset_use("long_hair", "tag") and store.uses(["long_hair"], "tag") == {}


def test_legacy_imports(store, tmp_path):
    legacy = tmp_path / "tag_frequency.db"
    with closing(sqlite3.connect(legacy)) as db, db:  # closed after the commit (sqlite3's own `with` only commits)
        db.execute("CREATE TABLE tag_frequency (name TEXT, type INT, count_pos INT, count_neg INT, last_used TIMESTAMP)")
        db.execute(
            "INSERT INTO tag_frequency VALUES ('long_hair', 1, 7, 1, '2026-10-01 10:00:00'), ('mylora', 8, 2, 0, NULL), "
            "('masterpiece', 2, 3, 0, NULL), ('hair/colors', 5, 1, 0, NULL), ('blonde hair', 4, 2, 0, NULL), ('bad', 1, 'x', 0, NULL), ('style', 11, 1, 0, NULL)"
        )
    assert store.import_tag_frequency(legacy) == 5 and store.import_tag_frequency(legacy) == 0
    assert store.uses(["long_hair"], "tag") == {"long_hair": 7} and store.uses(["mylora"], "lora") == {"mylora": 2}
    assert store.uses(["masterpiece"], "tag") == {"masterpiece": 3}  # extra tags count as tags
    assert store.uses(["hair/colors"], "wildcard") == {"hair/colors": 1} and store.uses(["blonde hair"], "wildcard-value") == {"blonde hair": 2}
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "history.txt2img.json").write_text(json.dumps([{"prompt": "a cat", "time": 5, "name": "kept"}, {"prompt": "bad time", "time": "soon"}, {"prompt": "x" * 70000}, 7]), encoding="utf8")
    (storage / "favorite.txt2img_neg.json").write_text(json.dumps([{"prompt": "blurry", "name": "neg"}, {"prompt": "lowres"}, {"prompt": None}]), encoding="utf8")
    (storage / "history.img2img.json").write_text("{not json", encoding="utf8")
    assert store.import_prompt_all_in_one(storage) == 4 and store.import_prompt_all_in_one(storage) == 0
    history = {item["prompt"]: item for item in store.history()[0]}
    assert set(history) == {"a cat", "bad time"} and history["a cat"]["name"] == "kept" and history["a cat"]["last_used"] == 5
    assert [(f["kind"], f["text"], f["name"]) for f in store.favorites(all_collections=True)] == [("negative", "blurry", "neg"), ("negative", "lowres", "")]


def test_a_legacy_database_in_a_folder_with_uri_characters_is_read(store, tmp_path):
    """The database opens read-only through a file: URI; %, ? and # in its path are escaped."""

    folder = tmp_path / "a%25b?#c"
    folder.mkdir()
    legacy = folder / "tag_frequency.db"
    with closing(sqlite3.connect(legacy)) as db, db:
        db.execute("CREATE TABLE tag_frequency (name TEXT, type INT, count_pos INT, count_neg INT, last_used TIMESTAMP)")
        db.executemany("INSERT INTO tag_frequency VALUES (?, 1, 1, 0, NULL)", [(f"tag{i}",) for i in range(5)])
    assert store.import_tag_frequency(legacy) == 5


@pytest.fixture
def shanghai_time(monkeypatch):
    """The process's local time zone is UTC+8 for the test."""
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is POSIX only")
def test_a_tag_import_reads_its_times_as_utc(store, tmp_path, shanghai_time):
    legacy = tmp_path / "tag_frequency.db"
    with closing(sqlite3.connect(legacy)) as db, db:  # TagComplete stores CURRENT_TIMESTAMP, which is UTC
        db.execute("CREATE TABLE tag_frequency (name TEXT, type INT, count_pos INT, count_neg INT, last_used TIMESTAMP)")
        db.execute("INSERT INTO tag_frequency VALUES ('long_hair', 1, 7, 1, '2026-01-01 00:00:00')")
    assert store.import_tag_frequency(legacy) == 1
    assert store._db.execute("SELECT last_used FROM usage WHERE name = 'long_hair'").fetchone()[0] == 1767225600


def test_legacy_import_is_all_or_nothing(store, tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "favorite.txt2img.json").write_text(json.dumps([{"prompt": "a"}, {"prompt": "b"}]), encoding="utf8")
    calls = []
    original = PromptToolsStore._insert_favorite

    def failing(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise sqlite3.OperationalError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PromptToolsStore, "_insert_favorite", failing)
    with pytest.raises(sqlite3.OperationalError):
        store.import_prompt_all_in_one(storage)
    assert store.favorites(all_collections=True) == [] and store.meta("imported:prompt_all_in_one") is None
    monkeypatch.setattr(PromptToolsStore, "_insert_favorite", original)
    assert store.import_prompt_all_in_one(storage) == 2  # the next start imports once, without duplicates
    assert [f["text"] for f in store.favorites(all_collections=True)] == ["a", "b"]


# endregion

def test_old_dotted_setting_keys_are_migrated():
    from remi.prompt_tools.options import migrate_values

    data = {"tac_extra.addMode": "Insert after", "tac_alias.searchByAlias": False, "tac_alias_searchByAlias": True, "tac_maxResults": 5}
    assert migrate_values(data) == ["tac_extra_addMode"]
    assert data == {"tac_extra_addMode": "Insert after", "tac_alias_searchByAlias": True, "tac_maxResults": 5}


LEGACY_TAC_SETTINGS = [
    "tac_active", "tac_activeIn.img2img", "tac_activeIn.modelList", "tac_activeIn.modelListMode", "tac_activeIn.negativePrompts",
    "tac_activeIn.thirdParty", "tac_activeIn.txt2img", "tac_alias.onlyShowAlias", "tac_alias.searchByAlias", "tac_alwaysSpaceAtEnd",
    "tac_animaArtistPrefix", "tac_appendComma", "tac_appendSpace", "tac_chantFile", "tac_civitaiApiKey",
    "tac_colormap", "tac_delayTime", "tac_escapeParentheses", "tac_extra.addMode", "tac_extra.extraFile",
    "tac_forceRefreshEmbeddings", "tac_frequencyFunction", "tac_frequencyIncludeAlias", "tac_frequencyMaxAge", "tac_frequencyMinCount",
    "tac_frequencyRecommendCap", "tac_frequencySort", "tac_includeEmbeddingsInNormalResults", "tac_keymap", "tac_maxResults",
    "tac_modelKeywordCivitai", "tac_modelKeywordCompletion", "tac_modelKeywordLocation", "tac_modelSortOrder", "tac_refreshTempFiles",
    "tac_replaceUnderscores", "tac_resultStepLength", "tac_showAllResults", "tac_showExtraNetworkPreviews", "tac_showWikiLinks",
    "tac_skipWildcardRefresh", "tac_slidingPopup", "tac_sortWildcardResults", "tac_tagFile", "tac_translation.liveTranslation",
    "tac_translation.oldFormat", "tac_translation.searchByTranslation", "tac_translation.translationFile", "tac_undersocreReplacementExclusionList", "tac_useEmbeddings",
    "tac_useIndexedSearch", "tac_useLoraPrefixForLycos", "tac_useLoras", "tac_useLycos", "tac_useStyleVars",
    "tac_useWildcards", "tac_wildcardCompletionMode", "tac_wildcardExclusionList",
]
"""every setting the tag autocomplete extension registered before the port
(``scripts/tag_autocomplete_helper.py`` at d23761c08: 58 keys)"""


def test_every_old_tag_autocomplete_setting_is_kept_or_explained():
    import re
    from pathlib import Path

    from remi.prompt_tools import options

    assert len(LEGACY_TAC_SETTINGS) == 58
    readme = (Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd-webui-tagcomplete-neo" / "README.md").read_text(encoding="utf8")
    table = readme.split("### Not carried over", 1)[1]
    explained = dict(re.findall(r"^\| `(tac_[\w.]+)` \| (.+) \|$", table, re.MULTILINE))
    kept = {key for key in LEGACY_TAC_SETTINGS if options.RENAMED.get(key, key) in options.TAG_OPTION_KEYS}
    assert sorted(set(LEGACY_TAC_SETTINGS) - kept - set(explained)) == []
    # the table only lists what really is gone, each with a reason
    assert kept.isdisjoint(explained) and set(explained) <= set(LEGACY_TAC_SETTINGS)
    assert all(len(reason) > 30 for reason in explained.values())
    # dotted keys are renamed, never dropped
    assert all(options.RENAMED[key] in options.TAG_OPTION_KEYS for key in LEGACY_TAC_SETTINGS if "." in key and key not in explained)


# region Translation


def test_dictionary_translates_both_ways():
    table = {"long_hair": "长发", "1girl": "1女孩"}
    assert dictionary_translate(["long hair", "cat"], table, "zh_CN") == {"long hair": "长发"}
    assert dictionary_translate(["长发", "猫"], table, "en") == {"长发": "long hair"}


def test_no_provider_means_no_translation():
    with pytest.raises(TranslationError) as error:
        translate(["a"], "en", ProviderConfig("none"), {})
    assert error.value.status == 409
    assert not ProviderConfig("openai", url="http://x").enabled and not ProviderConfig("libretranslate").enabled


def test_network_providers_are_called_once_per_missing_text():
    calls, cache = [], {}

    def post(url, headers, body):
        calls.append((url, headers, body))
        return {"translatedText": [f"<{text}>" for text in body["q"]]}

    config = ProviderConfig("libretranslate", url="http://127.0.0.1:5000/", key="k")
    out = translate(["长发", "猫", "猫"], "en", config, {"long_hair": "长发"}, cached=lambda texts: {t: cache[t] for t in texts if t in cache}, store=cache.update, post=post)
    assert out == ["long hair", "<猫>", "<猫>"]
    assert calls[0][0] == "http://127.0.0.1:5000/translate" and calls[0][2]["q"] == ["猫"] and calls[0][2]["api_key"] == "k"
    translate(["猫"], "en", config, {}, cached=lambda texts: {t: cache[t] for t in texts if t in cache}, store=cache.update, post=post)
    assert len(calls) == 1  # cached


def test_openai_and_deepl_requests():
    def chat(url, headers, body):
        assert url == "http://127.0.0.1:11434/v1/chat/completions" and "Authorization" not in headers
        assert json.loads(body["messages"][1]["content"]) == ["猫"] and body["model"] == "qwen"
        return {"choices": [{"message": {"content": 'Sure: ["cat"]'}}]}

    assert translate(["猫"], "en", ProviderConfig("openai", url="http://127.0.0.1:11434/v1", model="qwen"), {}, post=chat) == ["cat"]

    def deepl(url, headers, body):
        assert url == "https://api-free.deepl.com/v2/translate" and headers["Authorization"] == "DeepL-Auth-Key abc:fx" and body["target_lang"] == "ZH-HANS"
        return {"translations": [{"text": "猫"}]}

    assert translate(["cat"], "zh_CN", ProviderConfig("deepl", key="abc:fx"), {}, post=deepl) == ["猫"]

    def broken(url, headers, body):
        return {"choices": [{"message": {"content": "no list"}}]}

    with pytest.raises(TranslationError):
        translate(["猫"], "en", ProviderConfig("openai", url="http://x/v1", model="m"), {}, post=broken)


def test_an_answer_full_of_brackets_is_read_in_linear_time():
    # The list was found with re.search(r"\[.*\]", DOTALL): an answer with many "[" and no "]"
    # took quadratic time (40,000 brackets: 0.4 s; the 4 MB the provider may answer: about an hour).
    config = ProviderConfig("openai", url="http://x/v1", model="m")

    def brackets(url, headers, body):
        return {"choices": [{"message": {"content": "[" * 400_000}}]}

    started = time.perf_counter()
    with pytest.raises(TranslationError):
        translate(["猫"], "en", config, {}, post=brackets)
    assert time.perf_counter() - started < 2

    def wrapped(url, headers, body):
        return {"choices": [{"message": {"content": 'x] note [ignored: no] ["cat"] trailing'}}]}

    # the first "[" through the last "]", as before
    with pytest.raises(TranslationError):
        translate(["猫"], "en", config, {}, post=wrapped)

    def answer(url, headers, body):
        return {"choices": [{"message": {"content": 'Here you go:\n["cat", "dog"]\nDone.'}}]}

    assert translate(["猫", "狗"], "en", config, {}, post=answer) == ["cat", "dog"]


def test_a_written_prompt_with_a_long_blank_run_is_cleaned_in_linear_time():
    # the code fence was stripped with re.sub(r"^```[\w-]*\s*|\s*```$"): a long run of blanks
    # inside the answer took quadratic time (40,000 spaces: 1 s)
    from remi.prompt_tools.translate import generate_prompt

    config = ProviderConfig("openai", url="http://x/v1", model="m")

    def blanks(url, headers, body):
        return {"choices": [{"message": {"content": "a" + " " * 400_000 + "b"}}]}

    started = time.perf_counter()
    assert generate_prompt(config, "cat", post=blanks) == "a" + " " * 400_000 + "b"
    assert time.perf_counter() - started < 2

    def answering(content):
        return lambda url, headers, body: {"choices": [{"message": {"content": content}}]}

    assert generate_prompt(config, "cat", post=answering("```\n1girl, beach\n```")) == "1girl, beach"
    assert generate_prompt(config, "cat", post=answering("```text 1girl```")) == "1girl"
    for empty in ("```", "```python\n```"):  # nothing inside the fences
        with pytest.raises(TranslationError):
            generate_prompt(config, "cat", post=answering(empty))


def test_writing_a_prompt_from_a_description():
    from remi.prompt_tools.translate import DEFAULT_INSTRUCTIONS, generate_prompt

    config = ProviderConfig("openai", url="http://127.0.0.1:11434/v1", key="k", model="qwen")
    sent = []

    def chat(url, headers, body):
        sent.append((url, headers, body))
        return {"choices": [{"message": {"content": '```\nPrompt: "1girl, (long hair:1.2),\nbeach, sunset"\n```'}}]}

    assert generate_prompt(config, "一个女孩在海边", post=chat) == "1girl, (long hair:1.2), beach, sunset"
    url, headers, body = sent[-1]
    assert url == "http://127.0.0.1:11434/v1/chat/completions" and headers["Authorization"] == "Bearer k" and body["model"] == "qwen"
    assert body["messages"] == [{"role": "system", "content": DEFAULT_INSTRUCTIONS}, {"role": "user", "content": "一个女孩在海边"}]
    generate_prompt(config, "make it night", current="1girl, beach", instructions="Only tags.", post=chat)
    assert sent[-1][2]["messages"][0]["content"] == "Only tags." and "1girl, beach" in sent[-1][2]["messages"][1]["content"]
    generate_prompt(config, "blurry things", negative=True, post=chat)
    assert "negative" in sent[-1][2]["messages"][0]["content"]
    for bad_config, description, status in ((ProviderConfig("deepl", key="x"), "cat", 409), (ProviderConfig("openai", url="http://x/v1"), "cat", 409), (config, "  ", 422), (config, "x" * 5000, 413)):
        with pytest.raises(TranslationError) as error:
            generate_prompt(bad_config, description, post=chat)
        assert error.value.status == status
    with pytest.raises(TranslationError) as refused:
        generate_prompt(config, "  ", post=chat)
    assert (refused.value.param, refused.value.message) == ("description", "describe the image first")  # listed as a refused value
    with pytest.raises(TranslationError):
        generate_prompt(config, "cat", post=lambda *args: {"choices": [{"message": {"content": "  "}}]})


# endregion

# region Routes


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def data(tmp_path, monkeypatch):
    tags = tmp_path / "tags"
    tags.mkdir()
    (tags / "danbooru.csv").write_text(TAG_CSV, encoding="utf8")
    (tags / "extra-quality-tags.csv").write_text("masterpiece,5,Quality tag,,\n", encoding="utf8")
    (tags / "demo-chants.json").write_text(json.dumps([{"name": "Basic", "terms": "start", "content": "(masterpiece), 1girl"}]), encoding="utf8")
    groups = tmp_path / "group_tags"
    groups.mkdir()
    (groups / "zh_CN.yaml").write_text("- name: 人物\n  groups:\n    - name: 对象\n      color: red\n      tags:\n        1girl: 1女孩\n        long hair: 长发\n", encoding="utf8")
    (groups / "default.yaml").write_text("- name: Person\n  groups:\n    - name: Subject\n      tags:\n        1girl: 1girl\n", encoding="utf8")
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "animals.txt").write_text("cat\ndog\n", encoding="utf8")
    monkeypatch.setattr(pt, "registry", pt.Registry([str(tags)], [str(groups)], [str(wildcards)]))
    monkeypatch.setattr(pt._groups, "directories", [str(groups)])
    pt.refresh()
    yield tmp_path
    pt.refresh()


@pytest.fixture
def server(webui, data, tmp_path):
    from remi.server.testing import server_app

    with server_app(tmp_path / "server") as server:
        yield server
    pt.forget(server.services)


@pytest.fixture
def tool_settings(webui):
    """The prompt tools' settings, registered as at the WebUI's start, and taken out again:
    the labels are the shared ``options_templates``, so every later suite's ``webui_state``
    saw them, and the core catalog test took them for core settings."""

    from remi.prompt_tools import options

    labels = webui.opts.data_labels
    added = (set(options.TAG_OPTION_KEYS) | set(options.TOOL_OPTION_KEYS)) - set(labels)
    options.register_tag_options(pt.tag_dir, pt.refresh)
    options.register_tool_options()
    yield options
    for key in added:
        labels.pop(key, None)
        webui.opts.data.pop(key, None)


def test_settings_export_cleanly(webui, data, tool_settings):
    from remi.ui.settings import settings_schema

    options = tool_settings
    schema = settings_schema(webui.opts)
    sections = {section["id"]: section for category in schema["categories"] for section in category["sections"]}
    for section_id, keys in (("tac", options.TAG_OPTION_KEYS), ("prompt_tools", options.TOOL_OPTION_KEYS)):
        items = sections[section_id]["items"]
        assert [item["type"] for item in items if item["type"] == "html"] == []
        assert {item["name"] for item in items} == set(keys)
    tag_file = next(item for item in sections["tac"]["items"] if item["name"] == "tac_tagFile")
    assert [choice["value"] for choice in tag_file["choices"]] == ["None", "danbooru.csv", "extra-quality-tags.csv"]


def test_settings_are_translated_by_their_extensions(webui, data, tool_settings):
    from pathlib import Path

    from remi.ui.settings import settings_schema

    sections = {section["id"]: section for category in settings_schema(webui.opts)["categories"] for section in category["sections"]}
    root = Path(__file__).resolve().parents[1] / "extensions-builtin"
    for section_id, extension in (("tac", "sd-webui-tagcomplete-neo"), ("prompt_tools", "sd-webui-prompt-all-in-one-neo")):
        catalog = json.loads((root / extension / "localizations" / "zh_CN.json").read_text(encoding="utf8"))
        section = sections[section_id]
        texts = {section["label"]}
        for item in section["items"]:
            texts |= {item["label"], *item.get("notes", [])}
            if item["name"] not in ("tac_tagFile", "tac_extra_extraFile", "tac_translation_translationFile", "tac_chantFile"):  # file names
                texts |= {choice["label"] for choice in item.get("choices") or []}
        missing = sorted(text for text in texts if text and not catalog.get(text))
        assert missing == [], f"{extension}/localizations/zh_CN.json lacks {missing}"


def test_config_reflects_the_extensions_and_settings(server, webui, monkeypatch):
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert config["tags"]["enabled"] and config["tags"]["chants"] and config["tags"]["source"] == "danbooru"
    assert config["tools"] == {
        "enabled": True,
        "history": True,
        "groups": True,
        "translation": None,
        "translation_provider": "none",
        "translation_key": False,
        "generate": False,
        "language": "zh_CN",
        "comments": True,
        "toolbar": "All prompt fields",
    }
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "dictionary")
    monkeypatch.setitem(webui.opts.data, "tac_animaArtistPrefix", "Always")
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert config["tools"]["translation"] == "dictionary" and config["tags"]["artist_prefix"]
    monkeypatch.setattr(pt, "registry", pt.Registry())
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert not config["tags"]["enabled"] and not config["tools"]["enabled"]
    assert server.client.get("/api/v2/prompt-tools/complete", params={"q": "hair"}).json() == {"items": []}


def test_where_completion_runs_follows_the_old_settings(server, webui, monkeypatch):
    def tags():
        return server.client.get("/api/v2/prompt-tools/config").json()["tags"]

    config = tags()
    assert (config["txt2img"], config["img2img"], config["third_party"], config["delay"], config["only_alias"]) == (True, True, True, 40, False)
    for key, value in (("tac_activeIn_txt2img", False), ("tac_activeIn_img2img", False), ("tac_activeIn_thirdParty", False), ("tac_delayTime", 5000), ("tac_alias_onlyShowAlias", True)):
        monkeypatch.setitem(webui.opts.data, key, value)
    config = tags()
    # the delay within its slider (0 to 500 ms)
    assert (config["txt2img"], config["img2img"], config["third_party"], config["delay"], config["only_alias"]) == (False, False, False, 500, True)

    # the checkpoint list: names with or without extension, or hashes
    monkeypatch.setitem(webui.opts.data, "sd_model_checkpoint", "anime/animagine.safetensors [1a2b3c4d5e]")
    monkeypatch.setitem(webui.opts.data, "sd_checkpoint_hash", "1a2b3c4d5e6f7a8b9c0d")
    for listed, mode, enabled in (
        ("", "Blacklist", True),
        ("realistic.safetensors, animagine", "Blacklist", False),
        ("animagine.safetensors", "Blacklist", False),
        ("1A2B3C4D5E", "Blacklist", False),
        ("realistic", "Blacklist", True),
        ("realistic", "Whitelist", False),
        ("1a2b3c4d5e6f7a8b9c0d", "Whitelist", True),
    ):
        monkeypatch.setitem(webui.opts.data, "tac_activeIn_modelList", listed)
        monkeypatch.setitem(webui.opts.data, "tac_activeIn_modelListMode", mode)
        assert tags()["enabled"] is enabled, (listed, mode)
    # an unknown checkpoint never turns completion off
    monkeypatch.setitem(webui.opts.data, "sd_model_checkpoint", "")
    monkeypatch.setitem(webui.opts.data, "sd_checkpoint_hash", "")
    assert tags()["enabled"] is True


def test_usage_of_tags_found_by_an_alias_counts_only_when_asked(server, webui, monkeypatch):
    def first(q):
        return server.client.get("/api/v2/prompt-tools/complete", params={"q": q, "limit": 3}).json()["items"][0]

    monkeypatch.setitem(webui.opts.data, "tac_frequencyMinCount", 1)
    monkeypatch.setitem(webui.opts.data, "tac_frequencyFunction", "Usage first")
    for _ in range(3):
        server.client.post("/api/v2/prompt-tools/usage", json={"name": "breasts", "type": "tag"})
    # "boo" finds breasts by its alias "boobs" only
    assert first("boo")["text"] == "breasts" and first("boo")["alias"] == "boobs"
    assert first("boo").get("uses") is None
    monkeypatch.setitem(webui.opts.data, "tac_frequencyIncludeAlias", True)
    assert first("boo")["uses"] == 3


def test_generate_route_uses_the_openai_compatible_provider(server, webui, monkeypatch):
    api = "/api/v2/prompt-tools"
    assert server.client.get(f"{api}/config").json()["tools"]["generate"] is False
    response = server.client.post(f"{api}/generate", json={"description": "a cat"})
    assert response.status_code == 409 and "OpenAI-compatible" in response.json()["detail"]
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "openai")
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_model", "qwen")
    monkeypatch.setitem(webui.opts.data, "prompt_tools_generate_instructions", "Tags only.")
    assert server.client.get(f"{api}/config").json()["tools"]["generate"] is True
    calls = []

    def fake(config, description, **kwargs):
        calls.append((config, description, kwargs))
        return "cat, sitting"

    monkeypatch.setattr(pt, "generate_prompt", fake)
    answer = server.client.post(f"{api}/generate", json={"description": "a cat", "prompt": "1girl", "negative": True}).json()
    assert answer == {"prompt": "cat, sitting"}
    config, description, kwargs = calls[0]
    assert config.provider == "openai" and config.model == "qwen" and description == "a cat"
    assert kwargs == {"current": "1girl", "negative": True, "instructions": "Tags only."}
    assert server.client.post(f"{api}/generate", json={"description": ""}).status_code == 422


def test_upsample_loads_a_local_model_path_only_without_hide_ui_dir_config(server, webui, monkeypatch):
    """``local_model_path`` names a model file or folder anywhere on the server, which the
    upsampler loads (llama-cpp, or transformers from the folder): --hide-ui-dir-config refuses
    it, as it refuses the other paths on the server."""

    from modules import shared
    from remi.prompt_tools import pup

    calls = []

    def fake(prompt, **kwargs):
        calls.append(kwargs["local_model_path"])
        return pup.UpsampleResult(prompt=f"{prompt}, detailed", original_prompt=prompt, model_family="anima", mode="t2i", backend="local_qwen35")

    monkeypatch.setattr(pup, "upsample_prompt", fake)
    body = {"prompt": "a cat", "local_model_path": "/srv/models/qwen.gguf"}
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    refused = server.client.post("/api/v2/prompt-tools/upsample", json=body)
    assert refused.status_code == 403 and "--hide-ui-dir-config" in refused.json()["detail"] and calls == []
    assert server.client.post("/api/v2/prompt-tools/upsample", json={"prompt": "a cat"}).json()["prompt"] == "a cat, detailed"
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    assert server.client.post("/api/v2/prompt-tools/upsample", json=body).json()["prompt"] == "a cat, detailed"
    assert calls == [None, "/srv/models/qwen.gguf"]


def test_interrupt_stops_a_local_upsample_and_the_route_says_so(server, monkeypatch):
    """A local model runs behind generation, where Interrupt (``POST /interrupt``) flips the
    queue's state: the upsampler gets that flag to stop on, cleared first (an Interrupt pressed
    while nothing ran must not stop it at the first token) and cleared again after (it must not
    reach what runs next). An interrupted upsample answers 409 with the sentence the UI translates."""

    from modules import shared
    from remi.prompt_tools import pup

    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    seen = []

    def fake(prompt, *, should_stop, **kwargs):
        seen.append(should_stop())
        server.services.queue.interrupt_all()  # the user presses Interrupt while it writes
        seen.append(should_stop())
        raise pup.GenerationInterrupted()

    monkeypatch.setattr(pup, "upsample_prompt", fake)
    server.state.interrupt()  # left over from an Interrupt while nothing ran
    answer = server.client.post("/api/v2/prompt-tools/upsample", json={"prompt": "a cat", "local_model_path": "/srv/models/qwen.gguf"})
    assert seen == [False, True]
    assert answer.status_code == 409
    assert answer.json() == {"detail": "the prompt upsampling was interrupted", "template": "the prompt upsampling was interrupted", "params": {}}
    assert server.state.interrupted is False


def test_upsample_route_lists_a_refused_value_with_its_template(server):
    refused = server.client.post("/api/v2/prompt-tools/upsample", json={"prompt": "a dog", "aspect_ratio": "wide"})
    assert refused.status_code == 422
    assert refused.json() == {
        "errors": [
            {
                "owner": None,
                "param": "aspect_ratio",
                "message": "invalid aspect_ratio 'wide'; expected 'W:H' or 'WxH'",
                "template": "invalid aspect_ratio {{value}}; expected 'W:H' or 'WxH'",
                "params": {"value": "'wide'"},
            }
        ]
    }
    data_only = server.client.post("/api/v2/prompt-tools/upsample", json={"prompt": "a dog", "reference_images": ["https://example.com/x.png"]})
    assert data_only.status_code == 422 and data_only.json()["errors"][0]["param"] == "reference_images"


def test_unloading_an_extension_turns_its_feature_off(data):
    tags, groups = pt.registry.tag_dirs[0], pt.registry.group_dirs[0]
    pt.unregister_tag_data(tags)
    assert not pt.registry.tags_enabled and pt.registry.tools_enabled
    pt.unregister_tool_data(groups)
    assert not pt.registry.tools_enabled and pt._groups.directories == []
    pt.register_tag_data(tags)
    pt.register_tool_data(groups)
    assert pt.registry.tags_enabled and pt.registry.tools_enabled


def test_translation_key_is_write_only(server, webui, monkeypatch):
    api = "/api/v2/prompt-tools"
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "deepl")
    config = server.client.get(f"{api}/config").json()["tools"]
    assert config["translation"] is None and config["translation_provider"] == "deepl" and not config["translation_key"]
    assert server.client.put(f"{api}/translation-key", json={"key": " secret:fx "}).json() == {"configured": True}
    config = server.client.get(f"{api}/config").json()["tools"]
    assert config["translation"] == "deepl" and config["translation_key"] is True
    assert "secret" not in json.dumps(server.client.get(f"{api}/config").json())
    tools = pt.tools_for(server.services)
    assert tools.translation_config().key == "secret:fx"
    assert server.client.put(f"{api}/translation-key", json={"key": ""}).json() == {"configured": False}
    assert tools.translation_config().key == ""


@pytest.mark.parametrize(
    ("flag", "value", "detail"),
    [
        ("freeze_settings", True, "changing settings is disabled (--freeze-settings)"),
        ("freeze_settings_in_sections", "saving-images, prompt_tools", "the settings in this section are frozen"),
    ],
)
def test_frozen_settings_keep_the_translation_key(server, webui, monkeypatch, flag, value, detail):
    from modules import shared

    api = "/api/v2/prompt-tools"
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "deepl")
    assert server.client.put(f"{api}/translation-key", json={"key": "kept"}).json() == {"configured": True}
    monkeypatch.setattr(shared.cmd_opts, flag, value, raising=False)
    # the reason the settings give for a frozen setting (frozen_reason), as a sentence the UI translates:
    # the status alone read "It changed meanwhile"
    for key in ("other", ""):
        response = server.client.put(f"{api}/translation-key", json={"key": key})
        assert response.status_code == 409 and response.json() == {"detail": detail, "template": detail, "params": {}}
    assert pt.tools_for(server.services).translation_config().key == "kept"


def test_a_refused_collection_name_names_its_field_and_reason(server):
    """A collection name that is too long, or blank, answered FastAPI's or the store's English
    detail, which the UI could only say as "The server rejected the request.": it is a refused
    value on `name`, with a sentence the UI translates."""

    api = "/api/v2/prompt-tools"
    too_long = {"owner": None, "param": "name", "message": "a collection name must be at most 200 characters", "template": "a collection name must be at most {{maximum}} characters", "params": {"maximum": "200"}}
    blank = {"owner": None, "param": "name", "message": "a collection needs a name"}
    for name, error in (("x" * 201, too_long), ("  ", blank), ("", blank)):
        response = server.client.post(f"{api}/collections", json={"name": name})
        assert response.status_code == 422 and response.json() == {"errors": [error]}, name
    collection = server.client.post(f"{api}/collections", json={"name": "y" * 200}).json()
    for name, error in (("x" * 201, too_long), (" ", blank)):
        response = server.client.patch(f"{api}/collections/{collection['id']}", json={"name": name})
        assert response.status_code == 422 and response.json() == {"errors": [error]}, name


def test_the_translation_key_is_only_sent_where_it_was_entered_for(server, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "deepl")
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_url", "https://api.deepl.com")
    tools = pt.tools_for(server.services)
    assert tools.set_translation_key("secret:fx") and tools.translation_config().key == "secret:fx"
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_url", "https://elsewhere.example")
    assert tools.translation_config().key == ""  # a changed URL gets no key until it is entered again
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_url", "https://api.deepl.com")
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "openai")
    assert tools.translation_config().key == ""
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "deepl")
    assert tools.translation_config().key == "secret:fx"
    tools.set_translation_key("")


def test_a_key_in_the_settings_moves_to_the_store(server, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_key", "old-key")
    monkeypatch.setattr(webui.opts, "save", lambda filename: None, raising=False)
    tools = pt.tools_for(server.services)
    assert tools.migrate_translation_key()
    assert "prompt_tools_translation_key" not in webui.opts.data and tools.translation_config().key == "old-key"
    assert not tools.migrate_translation_key()


def test_complete_tags_and_record_usage(server, webui, monkeypatch):
    def complete(**params):
        response = server.client.get("/api/v2/prompt-tools/complete", params=params)
        assert response.status_code == 200, response.text
        return response.json()["items"]

    items = complete(q="hair", limit=4)
    assert [item["text"] for item in items] == ["hair_ornament", "hair_between_eyes", "long_hair", "very_long_hair"]
    assert items[2]["count"] == 4300000 and items[0]["type"] == "tag"
    assert complete(q="mas")[0] == {**complete(q="mas")[0], "type": "extra", "text": "masterpiece", "meta": "Quality tag"}
    monkeypatch.setitem(webui.opts.data, "tac_frequencyMinCount", 1)
    monkeypatch.setitem(webui.opts.data, "tac_frequencyFunction", "Usage first")
    for _ in range(2):
        assert server.client.post("/api/v2/prompt-tools/usage", json={"name": "very_long_hair", "type": "tag"}).json()["ok"]
    items = complete(q="hair", limit=4)
    assert items[0]["text"] == "very_long_hair" and items[0]["uses"] == 2
    # counted per prompt: the negative prompt is unaffected
    assert complete(q="hair", limit=4, negative=True)[0]["text"] == "hair_ornament"
    assert server.client.delete("/api/v2/prompt-tools/usage", params={"name": "very_long_hair"}).json()["ok"]
    assert complete(q="hair", limit=1)[0]["text"] == "hair_ornament"
    assert complete(q="@hatsune")[0]["text"] == "hatsune_miku"  # Anima's artist prefix is not searched
    assert complete(mode="chant", q="bas")[0] == {**complete(mode="chant", q="bas")[0], "type": "chant", "text": "(masterpiece), 1girl", "label": "Basic"}
    assert [item["text"] for item in complete(mode="wildcard", q="ani")] == ["animals"]
    assert [item["text"] for item in complete(mode="wildcard-value", wildcard="animals", q="d")] == ["dog"]
    assert server.client.get("/api/v2/prompt-tools/complete", params={"mode": "nope"}).status_code == 422


def test_lookup_groups_and_dictionary_translation(server, webui, monkeypatch):
    found = server.client.post("/api/v2/prompt-tools/lookup", json={"tags": ["long hair", "1girl", "unknown thing"]}).json()["tags"]
    assert found["long hair"] == {"category": 0, "count": 4300000, "translation": "长发"} and "unknown thing" not in found
    categories = server.client.get("/api/v2/prompt-tools/groups", params={"locale": "zh-CN"}).json()["categories"]
    assert categories[0]["name"] == "人物" and categories[0]["groups"][0]["tags"][0] == {"tag": "1girl", "label": "1女孩"}
    assert server.client.get("/api/v2/prompt-tools/groups", params={"locale": "en"}).json()["categories"][0]["name"] == "Person"
    assert server.client.post("/api/v2/prompt-tools/translate", json={"texts": ["长发"]}).status_code == 409
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "dictionary")
    answer = server.client.post("/api/v2/prompt-tools/translate", json={"texts": ["长发", "猫"], "target": "en"}).json()
    assert answer == {"provider": "dictionary", "translations": ["long hair", None]}


def test_network_completion_reads_the_network_index(server, tmp_path):
    from PIL import Image

    from remi.networks import NetworkEntry, NetworkIndex, lora_prompt

    class Static:
        def __init__(self, entries):
            self.entries = entries

        def refresh(self):
            pass

        def list(self):
            return list(self.entries)

    root = tmp_path / "Lora"
    (root / "style").mkdir(parents=True)
    lora = root / "style" / "ink.safetensors"
    lora.write_bytes(b"\0" * 16)
    (root / "style" / "ink.json").write_text(json.dumps({"preferred weight": 0.7, "activation text": "ink style", "sd version": "SDXL", "description": "Ink lines"}), encoding="utf8")
    Image.new("RGB", (8, 8)).save(root / "style" / "ink.png")
    plain = root / "plain.safetensors"
    plain.write_bytes(b"\0" * 16)
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "bad_hands.pt").write_bytes(b"\0" * 16)

    def failing(user):
        raise RuntimeError("broken provider")

    entries = {
        "lora": [
            NetworkEntry("lora", "ink", str(lora), title="ink", root=str(root), alias="inkAlias", prompt=lambda user: lora_prompt("inkAlias", user)),
            NetworkEntry("lora", "plain", str(plain), title="plain", root=str(root)),
        ],
        "embeddings": [
            NetworkEntry("embeddings", "bad_hands", str(embeddings / "bad_hands.pt"), root=str(embeddings), prompt=lambda user: "bad_hands_v2"),
            NetworkEntry("embeddings", "bad_feet", str(embeddings / "bad_feet.pt"), root=str(embeddings), prompt=failing),
        ],
    }
    index = NetworkIndex(server.services.files, lambda: {kind: Static(items) for kind, items in entries.items()}, option=lambda name, default: default)
    server.services.networks = index
    pt.tools_for(server.services).networks = index

    def complete(**params):
        response = server.client.get("/api/v2/prompt-tools/complete", params=params)
        assert response.status_code == 200, response.text
        return response.json()["items"]

    [ink] = complete(mode="lora", q="ink")
    preview = ink.pop("preview")
    assert isinstance(preview, str) and preview
    assert {key: value for key, value in ink.items() if value is not None} == {"type": "lora", "text": "inkAlias", "label": "ink", "weight": 0.7, "keywords": "ink style", "meta": "SDXL", "description": "Ink lines"}
    [plain_item] = complete(mode="lora", q="plain")
    assert plain_item["weight"] == 1.0 and plain_item["keywords"] is None and plain_item["meta"] is None and plain_item["preview"] is None
    found = {item["label"]: item["text"] for item in complete(mode="embedding", q="bad")}
    assert found == {"bad_hands": "bad_hands_v2", "bad_feet": "bad_feet"}  # a failing prompt falls back to the name
    assert {item["type"] for item in complete(mode="network", q="")} == {"lora", "embedding", "chant"}


def test_history_api_and_recording_from_jobs(server):
    api = "/api/v2/prompt-tools/history"
    created = server.client.post(api, json={"prompt": "a cat", "negative_prompt": "blurry", "tab": "txt2img"})
    assert created.status_code == 201 and created.json()["uses"] == 1
    empty = server.client.post(api, json={})
    assert empty.status_code == 422 and empty.json() == {"errors": [{"owner": None, "param": "prompt", "message": "both prompts are empty"}]}
    entry = created.json()
    assert server.client.patch(f"{api}/{entry['id']}", json={"name": "cat"}).json()["name"] == "cat"
    assert server.client.patch(f"{api}/missing", json={"name": "x"}).status_code == 404

    tools = pt.attach(server.services)
    # paused: the job stays queued (and is removed below, leaving no task behind)
    assert server.client.post("/api/v2/queue/pause").status_code == 200
    try:
        job = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "a dog", "negative_prompt": "lowres"}, "scripts": {}})
        assert job.status_code == 201, job.text
        assert tools._recorder.flush()
    finally:
        if "job" in locals() and job.status_code == 201:
            assert server.client.delete(f"/api/v2/jobs/{job.json()['job_id']}").status_code == 204
        server.client.post("/api/v2/queue/resume")
    page = server.client.get(api, params={"q": "dog"}).json()
    assert page["total"] == 1 and page["items"][0]["negative_prompt"] == "lowres" and page["items"][0]["tab"] == "txt2img"
    assert server.client.get(api, params={"field": "negative"}).json()["total"] == 2
    assert server.client.delete(f"{api}/{entry['id']}").json() == {"ok": True, "removed": 1}
    assert server.client.delete(f"{api}/{entry['id']}").status_code == 404
    assert server.client.delete(api).json() == {"removed": 1}
    assert server.client.get(api).json()["total"] == 0


def test_grouped_history_api(server):
    api = "/api/v2/prompt-tools/history"
    first = server.client.post(api, json={"prompt": "a cat", "negative_prompt": "blurry"}).json()
    server.client.post(api, json={"prompt": "a cat", "negative_prompt": "lowres"})
    renamed = server.client.patch(f"{api}/{first['id']}", json={"name": "cat", "field": "prompt"})
    assert renamed.status_code == 200
    assert {item["name"] for item in server.client.get(api).json()["items"]} == {"cat"}
    assert server.client.delete(f"{api}/{first['id']}", params={"field": "prompt"}).json() == {"ok": True, "removed": 2}
    assert server.client.get(api, params={"field": "prompt"}).json()["total"] == 0


def test_favorites_api(server):
    api = "/api/v2/prompt-tools"
    collection = server.client.post(f"{api}/collections", json={"name": "Faces"}).json()
    assert server.client.post(f"{api}/collections", json={"name": "Faces"}).status_code == 409
    favorite = server.client.post(f"{api}/favorites", json={"text": "smile", "kind": "tags", "collection_id": collection["id"]})
    assert favorite.status_code == 201
    loose = server.client.post(f"{api}/favorites", json={"text": "1girl, smile"}).json()
    server.client.post(f"{api}/favorites", json={"text": "blurry", "kind": "negative"})
    assert [f["text"] for f in server.client.get(f"{api}/favorites", params={"collection": ""}).json()["items"]] == ["blurry", "1girl, smile"]
    assert server.client.get(f"{api}/favorites").json()["total"] == 3
    # the prompt editor asks for its own kind and the tags
    wanted = server.client.get(f"{api}/favorites", params=[("kind", "prompt"), ("kind", "tags")]).json()
    assert sorted(f["text"] for f in wanted["items"]) == ["1girl, smile", "smile"] and wanted["total"] == 2
    paged = server.client.get(f"{api}/favorites", params={"limit": 2}).json()
    assert len(paged["items"]) == 2 and paged["next_cursor"]
    assert len(server.client.get(f"{api}/favorites", params={"limit": 2, "cursor": paged["next_cursor"]}).json()["items"]) == 1
    server.client.delete(f"{api}/favorites/{server.client.get(f'{api}/favorites', params={'kind': 'negative'}).json()['items'][0]['id']}")
    moved = server.client.patch(f"{api}/favorites/{loose['id']}", json={"collection_id": collection["id"]}).json()
    assert moved["collection_id"] == collection["id"]
    # name only: the collection stays
    assert server.client.patch(f"{api}/favorites/{loose['id']}", json={"name": "both"}).json()["collection_id"] == collection["id"]
    counts = {c["name"]: c["count"] for c in server.client.get(f"{api}/collections").json()}
    assert counts == {"": 0, "Faces": 2}
    exported = server.client.get(f"{api}/favorites-export").json()
    assert server.client.delete(f"{api}/collections/{collection['id']}").json()["ok"]
    assert server.client.get(f"{api}/favorites").json()["items"] == []
    assert server.client.post(f"{api}/favorites-import", json=exported).json() == {"added": 2}
    assert server.client.post(f"{api}/favorites", json={"text": ""}).status_code == 422


# endregion
