import pytest
from pydantic import ValidationError

from agent_runner.config import ChatRequest, ConversationReferenceRequest


def reference(source_id: str = "conv_source", boundary: int = 1) -> dict[str, object]:
    return {
        "source_conversation_id": source_id,
        "source_end_round_number": boundary,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"source_conversation_id": "conv_source", "source_end_round_number": 1, "extra": True},
        reference(boundary=0),
        reference(source_id="invalid"),
    ],
)
def test_reference_request_rejects_invalid_contract(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConversationReferenceRequest.model_validate(payload)


def test_chat_request_rejects_duplicate_and_self_references() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            conversation_id="conv_destination",
            message="Use sources",
            references=[reference(), reference()],
        )

    with pytest.raises(ValidationError):
        ChatRequest(
            conversation_id="conv_destination",
            message="Use sources",
            references=[reference(source_id="conv_destination")],
        )


def test_chat_request_rejects_more_than_ten_references() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            conversation_id="conv_destination",
            message="Use sources",
            references=[reference(source_id=f"conv_source_{index}") for index in range(11)],
        )
