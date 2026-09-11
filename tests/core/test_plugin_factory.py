import warnings

from ramappy.core import plugin_factory


class _DummyEP:
    def __init__(self, name, should_fail=False, load_order=None):
        self.name = name
        self._should_fail = should_fail
        self._load_order = load_order
        self.loaded = 0

    def load(self):
        self.loaded += 1
        if self._load_order is not None:
            self._load_order.append(self.name)
        if self._should_fail:
            raise RuntimeError("boom")


def test_discover_plugins_is_cached_and_sorted(monkeypatch):
    load_order: list[str] = []
    ep_b = _DummyEP("b", load_order=load_order)
    ep_a = _DummyEP("a", load_order=load_order)

    calls = []

    def fake_entry_points(*, group):
        calls.append(group)
        return [ep_b, ep_a]

    monkeypatch.setattr(plugin_factory, "entry_points", fake_entry_points)
    plugin_factory._loaded_groups.clear()

    plugin_factory.discover_and_import_plugins("io")
    plugin_factory.discover_and_import_plugins("io")

    assert calls == ["ramappy.plugins.io"]
    assert ep_a.loaded == 1
    assert ep_b.loaded == 1
    assert load_order == ["a", "b"]


def test_discover_plugins_warns_on_broken_plugin(monkeypatch):
    good = _DummyEP("good")
    bad = _DummyEP("bad", should_fail=True)

    def fake_entry_points(*, group):
        return [good, bad]

    monkeypatch.setattr(plugin_factory, "entry_points", fake_entry_points)
    plugin_factory._loaded_groups.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugin_factory.discover_and_import_plugins("steps")

    assert good.loaded == 1
    assert bad.loaded == 1
    assert any("Failed loading ramappy plugin" in str(w.message) for w in caught)
