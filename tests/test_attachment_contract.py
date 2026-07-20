import pytest
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    ConversationFileKind,
    ConversationFileStatus,
    PreparedConversationFile,
)

from agent_runner.config import ChatRequest
from agent_runner.context.builder import Message
from agent_runner.runtime.openai_agents_runtime import OpenAIAgentsRuntime
from agent_runner.runtime.orchestrator import RuntimeOrchestrator


def test_chat_request_accepts_attachment_only_and_rejects_duplicate_files() -> None:
    request = ChatRequest(
        conversation_id="conv_attachments",
        file_ids=["file_one"],
        ui_locale="en-US",
    )

    assert request.message == ""
    assert request.file_ids == ["file_one"]

    with pytest.raises(ValueError):
        ChatRequest(
            conversation_id="conv_attachments",
            file_ids=["file_one", "file_one"],
        )


def test_attachment_input_separates_signed_sdk_urls_from_stable_capture() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ChatRequest(
        conversation_id="conv_attachments",
        message="Compare the files",
        file_ids=["file_image", "file_document"],
        ui_locale="en-US",
    )
    prepared_files = [
        PreparedConversationFile(
            file_id="file_image",
            original_filename="diagram.png",
            mime_type="image/png",
            kind=ConversationFileKind.IMAGE,
            status=ConversationFileStatus.READY,
            download_url="https://signed.example/image",
        ),
        PreparedConversationFile(
            file_id="file_document",
            original_filename="notes.txt",
            mime_type="text/plain",
            kind=ConversationFileKind.TEXT,
            status=ConversationFileStatus.READY,
            extracted_text="Exact extracted evidence.",
        ),
    ]

    current_message, metadata, instruction = orchestrator._build_attachment_input(request, prepared_files)

    sdk_content = metadata["sdk_content"]
    capture_content = metadata["capture_content"]
    assert instruction == ""
    assert "Exact extracted evidence." in current_message
    assert sdk_content[1]["image_url"] == "https://signed.example/image"
    assert capture_content[1]["file_url"]["url"] == "agentbreaker-file://file_image"
    assert "signed.example" not in str(capture_content)

    user_request = orchestrator._build_user_request(request)
    assert [part.file_url.url for part in user_request.content_parts if part.file_url] == [
        "agentbreaker-file://file_image",
        "agentbreaker-file://file_document",
    ]


def test_attachment_only_instruction_uses_the_ui_locale_without_visible_fake_text() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ChatRequest(
        conversation_id="conv_attachments",
        file_ids=["file_document"],
        ui_locale="zh-CN",
    )
    prepared_file = PreparedConversationFile(
        file_id="file_document",
        original_filename="notes.txt",
        mime_type="text/plain",
        kind=ConversationFileKind.TEXT,
        status=ConversationFileStatus.READY,
        extracted_text="Evidence",
    )

    _, _, instruction = orchestrator._build_attachment_input(request, [prepared_file])
    user_request = orchestrator._build_user_request(request)

    assert "Simplified Chinese" in instruction
    assert user_request.content == ""
    assert len(user_request.content_parts) == 1
    assert user_request.content_parts[0].type == "file_url"


def test_plain_text_capture_does_not_persist_an_empty_content_parts_list() -> None:
    """The attachment metadata envelope must not turn an ordinary text message into invalid RPC content."""
    runtime = OpenAIAgentsRuntime()
    captured = runtime._to_capture_message(Message(
        role="user",
        content="Question",
        metadata={"capture_content": []},
    ))

    assert captured == {"role": "user", "content": "Question"}


def test_plain_text_attachment_input_omits_empty_model_metadata() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ChatRequest(
        conversation_id="conv_plain_text",
        message="Continue from the previous answer",
    )

    attachment_input = orchestrator._build_attachment_input(request, [])

    assert attachment_input.current_message == "Continue from the previous answer"
    assert attachment_input.as_metadata() == {}
