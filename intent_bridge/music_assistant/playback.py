"""Music Assistant playback dispatch and radio policy."""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any

try:
    from music_assistant_models.enums import QueueOption
except Exception:  # pragma: no cover - deployment dependency guard
    QueueOption = None

from intent_bridge.config import log, settings
from intent_bridge.indicators.controller import (
    SatelliteIndicatorHandle,
    voice_activity_indicators,
)
from intent_bridge.music_assistant.client import (
    MusicPlayDispatchResult,
    NativeMusicAssistant,
    _ma_enum_value,
    _ma_media_summary,
    _ma_name,
    _ma_uri,
)


async def _ma_dispatch_play_media(
    manager: NativeMusicAssistant,
    client: Any,
    *,
    queue_id: str,
    media: str | list[str],
    option: Any,
    radio_mode: bool,
    label: str,
    origin_context: dict[str, Any] | None = None,
) -> MusicPlayDispatchResult:
    """Optimistically dispatch long-running play_media for voice interactions.

    Music Assistant 2.8.x may resolve an artist/playlist, generate radio tracks,
    build the queue, resolve stream details and prepare audio before the command
    returns. A short ACK window catches immediate success/errors. If the command
    is still running, the voice request succeeds optimistically while the MA
    future stays alive. Identical in-flight writes for the same queue/media are
    suppressed to avoid impatient voice retries duplicating playback requests.
    """
    media_payload = media if isinstance(media, list) else [media]
    fingerprint_payload = {
        "queue_id": queue_id,
        "media": [str(item) for item in media_payload],
        "option": str(getattr(option, "value", option)),
        "radio_mode": bool(radio_mode),
    }
    fingerprint = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))

    existing = manager.get_inflight_playback(queue_id, fingerprint)
    if existing is not None:
        log.info(
            "MA NATIVE PLAY_MEDIA DUPLICATE SUPPRESSED queue_id=%r label=%s age=%.1fs",
            queue_id,
            existing.label,
            max(0.0, time.time() - existing.started_at),
        )
        return MusicPlayDispatchResult(
            command_acknowledged=False,
            command_dispatched=False,
            still_processing=True,
            duplicate_suppressed=True,
        )

    manager.begin_queue_playback_generation(queue_id)

    log.info(
        "MA NATIVE PLAY_MEDIA DISPATCH queue_id=%r media_payload=%r option=%s "
        "radio_mode=%s ack_wait=%.1fs",
        queue_id,
        media_payload,
        getattr(option, "value", option),
        radio_mode,
        settings.music_assistant.play_ack_timeout_seconds,
    )

    async def _complete() -> None:
        await asyncio.wait_for(
            client.player_queues.play_media(
                queue_id=queue_id,
                media=media_payload,
                option=option,
                radio_mode=bool(radio_mode),
            ),
            timeout=settings.music_assistant.play_completion_timeout_seconds,
        )

    task = asyncio.create_task(_complete(), name=f"ma-play-media-{queue_id}")
    manager.register_inflight_playback(queue_id, fingerprint, task, label)
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=settings.music_assistant.play_ack_timeout_seconds,
        )
        manager.clear_inflight_playback(queue_id, fingerprint, task)
        log.info("MA NATIVE PLAY_MEDIA ACKNOWLEDGED queue_id=%r", queue_id)
        return MusicPlayDispatchResult(
            command_acknowledged=True,
            command_dispatched=True,
            still_processing=False,
            duplicate_suppressed=False,
        )
    except TimeoutError:

        async def _background_lifecycle() -> None:
            indicator_handle: SatelliteIndicatorHandle | None = None
            try:
                # LED resolution/control happens only after the voice ACK window
                # and never delays the optimistic spoken response.
                indicator_handle = await voice_activity_indicators.begin(origin_context)
                await task
            finally:
                manager.clear_inflight_playback(queue_id, fingerprint, task)
                await voice_activity_indicators.end(indicator_handle)

        lifecycle = asyncio.create_task(
            _background_lifecycle(),
            name=f"ma-play-lifecycle-{queue_id}",
        )
        manager.track_background_task(lifecycle, label=label)
        log.info(
            "MA NATIVE PLAY_MEDIA STILL PROCESSING queue_id=%r; optimistic voice ACK; inflight=%d",
            queue_id,
            manager.inflight_playback_count,
        )
        return MusicPlayDispatchResult(
            command_acknowledged=False,
            command_dispatched=True,
            still_processing=True,
            duplicate_suppressed=False,
        )
    except Exception:
        manager.clear_inflight_playback(queue_id, fingerprint, task)
        raise


def _ma_media_provider_and_item_id(item: Any) -> tuple[str, str]:
    """Return provider + item ID needed by MA item-specific controller calls."""
    provider = str(_ma_enum_value(getattr(item, "provider", "")) or "").strip()
    item_id = str(getattr(item, "item_id", "") or "").strip()
    uri = _ma_uri(item)
    if uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        if not provider:
            provider = scheme.strip()
        if not item_id and "/" in rest:
            # MA URI shape: provider://media_type/item_id. Preserve any
            # additional slashes in provider-specific item IDs.
            item_id = rest.split("/", 1)[1].strip()
    return provider, item_id


def _ma_queue_playback_marker(client: Any, queue_id: str) -> tuple[str, str, str, str]:
    """Stable-enough cached marker used to distinguish new audio from old audio."""
    queue = client.player_queues.get(queue_id)
    if queue is None:
        return ("", "", "", "")
    current = getattr(queue, "current_item", None)
    if current is None:
        return (
            str(_ma_enum_value(getattr(queue, "state", "")) or "").casefold(),
            "",
            "",
            "",
        )
    media_item = getattr(current, "media_item", None)
    uri = _ma_uri(media_item) if media_item is not None else ""
    if not uri:
        uri = str(getattr(current, "uri", "") or "").strip()
    queue_item_id = str(
        getattr(current, "queue_item_id", None) or getattr(current, "item_id", None) or ""
    )
    name = str(getattr(current, "name", "") or _ma_name(media_item) or "").strip()
    state = str(_ma_enum_value(getattr(queue, "state", "")) or "").casefold()
    return (state, queue_item_id, uri, name.casefold())


def _ma_marker_is_playing(marker: tuple[str, str, str, str]) -> bool:
    return marker[0] == "playing"


def _ma_first_audio_matches(
    marker: tuple[str, str, str, str],
    *,
    baseline: tuple[str, str, str, str],
    expected_uri: str | None = None,
    expected_name: str | None = None,
) -> bool:
    if not _ma_marker_is_playing(marker):
        return False
    # If the player was not already playing, any new PLAYING state is enough.
    if not _ma_marker_is_playing(baseline):
        return True
    # When replacing already-playing audio, require evidence that the current
    # item changed so an old PLAYING state cannot satisfy the watcher instantly.
    if marker[1:] != baseline[1:]:
        if expected_uri and marker[2] and marker[2] == expected_uri:
            return True
        if expected_name and marker[3] and marker[3] == expected_name.casefold():
            return True
        # Queue item IDs are regenerated by REPLACE and are the most reliable
        # generic signal when MA normalizes a library URI to a provider URI.
        if marker[1] and marker[1] != baseline[1]:
            return True
        if marker[2] and marker[2] != baseline[2]:
            return True
        if marker[3] and marker[3] != baseline[3]:
            return True
    return False


async def _ma_wait_for_first_audio(
    client: Any,
    *,
    queue_id: str,
    baseline: tuple[str, str, str, str],
    expected_uri: str | None,
    expected_name: str | None,
    timeout: float,
    manager: NativeMusicAssistant | None = None,
    generation: int | None = None,
) -> tuple[bool, float | None, tuple[str, str, str, str]]:
    """Watch MA's event-updated queue cache for the requested audio to start."""
    started = time.monotonic()
    deadline = started + timeout
    last_marker = baseline
    while True:
        if (
            manager is not None
            and generation is not None
            and not manager.is_current_queue_playback_generation(queue_id, generation)
        ):
            return False, None, last_marker
        marker = _ma_queue_playback_marker(client, queue_id)
        last_marker = marker
        if _ma_first_audio_matches(
            marker,
            baseline=baseline,
            expected_uri=expected_uri,
            expected_name=expected_name,
        ):
            elapsed = max(0.0, time.monotonic() - started)
            return True, elapsed, marker
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, None, last_marker
        await asyncio.sleep(min(settings.music_assistant.first_audio_poll_seconds, remaining))


def _ma_radio_seed_candidates(tracks: list[Any]) -> list[Any]:
    candidates: list[Any] = []
    seen: set[str] = set()
    for track in tracks:
        uri = _ma_uri(track)
        if not uri or uri in seen:
            continue
        if getattr(track, "available", True) is False:
            continue
        seen.add(uri)
        candidates.append(track)
        if len(candidates) >= settings.music_assistant.radio_seed_top_n:
            break
    return candidates


def _ma_choose_radio_seed(
    manager: NativeMusicAssistant,
    *,
    artist_uri: str,
    tracks: list[Any],
) -> tuple[Any, int, list[dict[str, Any]]]:
    candidates = _ma_radio_seed_candidates(tracks)
    if not candidates:
        raise ValueError("Music Assistant returned no available top tracks for the artist")

    previous_uri = manager.last_radio_seed(artist_uri)
    pool = candidates
    if previous_uri and len(candidates) > 1:
        without_previous = [item for item in candidates if _ma_uri(item) != previous_uri]
        if without_previous:
            pool = without_previous

    strategy = settings.music_assistant.radio_seed_strategy
    if strategy == "first":
        selected = pool[0]
    elif strategy == "random":
        selected = random.choice(pool)
    else:
        # Weighted randomness preserves MA's top-track ordering without always
        # starting track #1. Candidate rank is taken from MA, not invented here.
        original_rank = {_ma_uri(item): idx for idx, item in enumerate(candidates)}
        weights = [
            max(1, settings.music_assistant.radio_seed_top_n - original_rank[_ma_uri(item)])
            for item in pool
        ]
        selected = random.choices(pool, weights=weights, k=1)[0]

    selected_uri = _ma_uri(selected)
    manager.remember_radio_seed(artist_uri, selected_uri)
    rank = next(
        (idx + 1 for idx, item in enumerate(candidates) if _ma_uri(item) == selected_uri), 1
    )
    return selected, rank, [_ma_media_summary(item, "track") for item in candidates]


@dataclass(frozen=True)
class FastRadioDispatchResult:
    command_acknowledged: bool
    command_dispatched: bool
    still_processing: bool
    duplicate_suppressed: bool
    first_audio_observed: bool
    first_audio_seconds: float | None
    seed: dict[str, Any] | None
    seed_rank: int | None
    seed_pool: list[dict[str, Any]]


async def _ma_dispatch_fast_artist_radio(
    manager: NativeMusicAssistant,
    client: Any,
    *,
    queue_id: str,
    artist: Any,
    origin_context: dict[str, Any] | None,
    label: str,
) -> FastRadioDispatchResult:
    """Start one MA-provided artist seed ASAP, then let MA own radio continuation."""
    artist_uri = _ma_uri(artist)
    provider, artist_item_id = _ma_media_provider_and_item_id(artist)
    if not artist_uri or not provider or not artist_item_id:
        raise ValueError("Selected Music Assistant artist is missing URI/provider/item_id")

    fingerprint = json.dumps(
        {"queue_id": queue_id, "mode": "fast_artist_radio", "artist": artist_uri},
        sort_keys=True,
        separators=(",", ":"),
    )
    existing = manager.get_inflight_playback(queue_id, fingerprint)
    if existing is not None:
        log.info(
            "MA FAST RADIO DUPLICATE SUPPRESSED queue_id=%r artist=%r age=%.1fs",
            queue_id,
            artist_uri,
            max(0.0, time.time() - existing.started_at),
        )
        return FastRadioDispatchResult(
            command_acknowledged=False,
            command_dispatched=False,
            still_processing=True,
            duplicate_suppressed=True,
            first_audio_observed=False,
            first_audio_seconds=None,
            seed=None,
            seed_rank=None,
            seed_pool=[],
        )

    bootstrap_started = time.monotonic()
    top_tracks_started = bootstrap_started
    tracks = await client.music.get_artist_tracks(
        item_id=artist_item_id,
        provider_instance_id_or_domain=provider,
        in_library_only=False,
    )
    top_tracks_ms = (time.monotonic() - top_tracks_started) * 1000.0
    seed, seed_rank, seed_pool = _ma_choose_radio_seed(
        manager,
        artist_uri=artist_uri,
        tracks=list(tracks),
    )
    seed_uri = _ma_uri(seed)
    seed_name = _ma_name(seed)
    baseline = _ma_queue_playback_marker(client, queue_id)
    generation = manager.begin_queue_playback_generation(queue_id)

    log.info(
        "MA FAST RADIO SEED queue_id=%r artist=%r provider=%r top_tracks_ms=%.1f "
        "seed=%r seed_name=%r seed_rank=%s pool=%s strategy=%s generation=%s",
        queue_id,
        artist_uri,
        provider,
        top_tracks_ms,
        seed_uri,
        seed_name,
        seed_rank,
        len(seed_pool),
        settings.music_assistant.radio_seed_strategy,
        generation,
    )

    assert QueueOption is not None

    async def _play_seed() -> None:
        await asyncio.wait_for(
            client.player_queues.play_media(
                queue_id=queue_id,
                media=[seed_uri],
                option=QueueOption.REPLACE,
                radio_mode=False,
            ),
            timeout=settings.music_assistant.background_timeout_seconds,
        )

    command_task = asyncio.create_task(_play_seed(), name=f"ma-fast-radio-play-{queue_id}")

    dedupe_release = asyncio.Event()

    async def _hold_dedupe() -> None:
        await dedupe_release.wait()

    dedupe_task = asyncio.create_task(_hold_dedupe(), name=f"ma-fast-radio-dedupe-{queue_id}")
    manager.register_inflight_playback(queue_id, fingerprint, dedupe_task, label)
    manager.track_background_task(dedupe_task, label=f"{label}:dedupe")

    audio_task = asyncio.create_task(
        _ma_wait_for_first_audio(
            client,
            queue_id=queue_id,
            baseline=baseline,
            expected_uri=seed_uri,
            expected_name=seed_name,
            timeout=settings.music_assistant.background_timeout_seconds,
            manager=manager,
            generation=generation,
        ),
        name=f"ma-fast-radio-first-audio-{queue_id}",
    )

    # Give immediate command errors or first-audio success a short opportunity
    # to surface, but never make recommendation/radio generation part of the
    # synchronous voice response path.
    done, _ = await asyncio.wait(
        {command_task, audio_task},
        timeout=settings.music_assistant.play_ack_timeout_seconds,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if command_task in done:
        try:
            command_task.result()
        except Exception:
            manager.clear_inflight_playback(queue_id, fingerprint, dedupe_task)
            dedupe_release.set()
            audio_task.cancel()
            await asyncio.gather(audio_task, return_exceptions=True)
            raise

    immediate_audio = False
    immediate_audio_seconds: float | None = None
    if audio_task in done:
        immediate_audio, immediate_audio_seconds, marker = audio_task.result()
        if immediate_audio:
            log.info(
                "MA TIME_TO_FIRST_AUDIO queue_id=%r artist=%r seed=%r seconds=%.3f marker=%r",
                queue_id,
                artist_uri,
                seed_uri,
                immediate_audio_seconds or 0.0,
                marker,
            )

    async def _command_cleanup() -> None:
        await command_task

    if not command_task.done():
        cleanup = asyncio.create_task(
            _command_cleanup(), name=f"ma-fast-radio-command-cleanup-{queue_id}"
        )
        manager.track_background_task(cleanup, label=f"{label}:seed_play")

    async def _radio_lifecycle() -> None:
        indicator_handle: SatelliteIndicatorHandle | None = None
        first_audio = immediate_audio
        first_audio_seconds = immediate_audio_seconds
        try:
            if not first_audio:
                # Indicator represents waiting for audible playback, not waiting
                # for MA's recommendation engine or play_media response to end.
                indicator_handle = await voice_activity_indicators.begin(origin_context)
                try:
                    ux_remaining = max(
                        0.05,
                        settings.music_assistant.first_audio_timeout_seconds
                        - (time.monotonic() - bootstrap_started),
                    )
                    first_audio, first_audio_seconds, marker = await asyncio.wait_for(
                        asyncio.shield(audio_task),
                        timeout=ux_remaining,
                    )
                except TimeoutError:
                    log.warning(
                        "MA FIRST AUDIO UX TIMEOUT queue_id=%r artist=%r seed=%r timeout=%.1fs; "
                        "indicator will restore while background watcher continues",
                        queue_id,
                        artist_uri,
                        seed_uri,
                        settings.music_assistant.first_audio_timeout_seconds,
                    )
                    await voice_activity_indicators.end(indicator_handle)
                    indicator_handle = None
                    # Continue waiting without holding the LED indefinitely.
                    first_audio, first_audio_seconds, marker = await audio_task
                if first_audio:
                    log.info(
                        "MA TIME_TO_FIRST_AUDIO queue_id=%r artist=%r seed=%r seconds=%.3f marker=%r",
                        queue_id,
                        artist_uri,
                        seed_uri,
                        first_audio_seconds or 0.0,
                        marker,
                    )

            if not first_audio:
                return
            if not manager.is_current_queue_playback_generation(queue_id, generation):
                log.info(
                    "MA FAST RADIO continuation skipped: request superseded queue_id=%r generation=%s",
                    queue_id,
                    generation,
                )
                return

            # MA owns radio continuation. On 2.8.7 this is a quick setting change
            # that schedules the expensive similar-track fill asynchronously.
            await asyncio.wait_for(
                client.player_queues.dont_stop_the_music(
                    queue_id=queue_id,
                    dont_stop_the_music_enabled=True,
                ),
                timeout=settings.music_assistant.command_timeout_seconds,
            )
            log.info(
                "MA FAST RADIO DSTM ENABLED queue_id=%r artist=%r seed=%r generation=%s",
                queue_id,
                artist_uri,
                seed_uri,
                generation,
            )
        finally:
            await voice_activity_indicators.end(indicator_handle)
            manager.clear_inflight_playback(queue_id, fingerprint, dedupe_task)
            dedupe_release.set()

    lifecycle = asyncio.create_task(_radio_lifecycle(), name=f"ma-fast-radio-lifecycle-{queue_id}")
    manager.track_background_task(lifecycle, label=f"{label}:radio_lifecycle")

    return FastRadioDispatchResult(
        command_acknowledged=command_task.done()
        and not command_task.cancelled()
        and command_task.exception() is None,
        command_dispatched=True,
        still_processing=not command_task.done(),
        duplicate_suppressed=False,
        first_audio_observed=immediate_audio,
        first_audio_seconds=immediate_audio_seconds,
        seed=_ma_media_summary(seed, "track"),
        seed_rank=seed_rank,
        seed_pool=seed_pool,
    )
