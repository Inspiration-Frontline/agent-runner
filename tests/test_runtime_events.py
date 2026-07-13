import logging

from agent_runner.runtime.events import RuntimeEvent, RuntimeEventBus, RuntimeEventType


async def test_failing_handler_is_logged_and_does_not_stop_other_handlers(caplog):
    event_bus = RuntimeEventBus()
    called: list[str] = []

    def failing_handler(event: RuntimeEvent) -> None:
        raise RuntimeError("handler failed")

    async def succeeding_handler(event: RuntimeEvent) -> None:
        called.append(event.event_id)

    event = RuntimeEvent.create(RuntimeEventType.REQUEST_START, {}, "request-1")
    event_bus.subscribe(RuntimeEventType.REQUEST_START, failing_handler)
    event_bus.subscribe(RuntimeEventType.REQUEST_START, succeeding_handler)

    with caplog.at_level(logging.ERROR):
        await event_bus.publish(event)

    assert called == [event.event_id]
    assert "Runtime event handler failed" in caplog.text
