from __future__ import annotations

import ramappy


def test_auto_enable_accelerations_is_enabled_by_default(monkeypatch):
    called: list[bool] = []

    def fake_enable(*, verbose: bool = False) -> bool:
        called.append(verbose)
        return True

    monkeypatch.delenv("RAMAPPY_AUTO_SKLEARNEX", raising=False)
    monkeypatch.setattr(ramappy, "_accel", type("FakeAccel", (), {"enable_sklearnex": staticmethod(fake_enable)}))

    assert ramappy._auto_enable_accelerations() is True
    assert called == [False]


def test_auto_enable_accelerations_respects_explicit_opt_in(monkeypatch):
    called: list[bool] = []

    def fake_enable(*, verbose: bool = False) -> bool:
        called.append(verbose)
        return True

    monkeypatch.setenv("RAMAPPY_AUTO_SKLEARNEX", "1")
    monkeypatch.setattr(ramappy, "_accel", type("FakeAccel", (), {"enable_sklearnex": staticmethod(fake_enable)}))

    assert ramappy._auto_enable_accelerations() is True
    assert called == [False]


def test_auto_enable_accelerations_respects_explicit_disable(monkeypatch):
    called: list[bool] = []

    def fake_enable(*, verbose: bool = False) -> bool:
        called.append(verbose)
        return True

    monkeypatch.setenv("RAMAPPY_AUTO_SKLEARNEX", "false")
    monkeypatch.setattr(ramappy, "_accel", type("FakeAccel", (), {"enable_sklearnex": staticmethod(fake_enable)}))

    assert ramappy._auto_enable_accelerations() is False
    assert called == []
