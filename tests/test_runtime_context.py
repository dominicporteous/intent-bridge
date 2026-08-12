from intent_bridge.runtime import context as runtime_context


def test_informational_grounding_bootstraps_from_home_assistant(monkeypatch):
    monkeypatch.setattr(
        runtime_context,
        "runtime_context",
        lambda timezone: f"Trusted runtime timezone={timezone}",
    )
    context = runtime_context.informational_runtime_context(
        "UTC",
        "en-GB",
        "",
        {"area_name": "Office"},
        home_assistant_config={
            "time_zone": "Europe/London",
            "language": "en",
            "country": "GB",
            "currency": "GBP",
            "unit_system": {"temperature": "°C", "length": "km"},
            "latitude": 51.5,
            "longitude": -0.1,
        },
        timezone_explicit=False,
        locale_explicit=False,
        location_explicit=False,
    )

    assert "Europe/London" in context
    assert "locale=en-GB" in context
    assert "default geographic location=country code GB" in context
    assert "temperature=°C" in context
    assert "currency=GBP" in context
    assert "Voice-origin room: Office" in context
    assert "51.5" not in context
    assert "-0.1" not in context


def test_explicit_grounding_overrides_home_assistant():
    context = runtime_context.informational_runtime_context(
        "America/New_York",
        "en-US",
        "New York, United States",
        home_assistant_config={
            "time_zone": "Europe/London",
            "language": "en",
            "country": "GB",
        },
        timezone_explicit=True,
        locale_explicit=True,
        location_explicit=True,
    )

    assert "America/New_York" in context or " in UTC" in context
    assert "locale=en-US" in context
    assert "default geographic location=New York, United States" in context
    assert "country code GB" not in context
