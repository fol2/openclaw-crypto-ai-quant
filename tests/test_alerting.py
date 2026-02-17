from __future__ import annotations

import threading

from engine import alerting


def test_note_alert_drop_thread_safe(monkeypatch):
    monkeypatch.setattr(alerting, "_ALERT_DROPPED_COUNT", 0)
    monkeypatch.setattr(alerting._alert_logger, "warning", lambda *args, **kwargs: None)

    workers = 12
    increments_per_worker = 4000
    start_barrier = threading.Barrier(workers)

    def _worker() -> None:
        start_barrier.wait()
        for _ in range(increments_per_worker):
            alerting._note_alert_drop()

    threads = [threading.Thread(target=_worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert alerting._ALERT_DROPPED_COUNT == workers * increments_per_worker
