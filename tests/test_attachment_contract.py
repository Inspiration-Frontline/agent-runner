import pytest
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    ConversationFileKind,
    ConversationFileStatus,
    PreparedConversationFile,
)

from agent_runner.config import ConversationRequest
from agent_runner.context.builder import CaptureFilePart, Message, ModelImagePart
from agent_runner.runtime.openai_agents_sdk_adapter import OpenAIAgentsSdkAdapter
from agent_runner.runtime.orchestrator import RuntimeOrchestrator


def test_conversation_request_accepts_attachment_only_and_rejects_duplicate_files() -> None:
    request = ConversationRequest(
        conversation_id="conv_attachments",
        file_ids=["file_one"],
        ui_locale="en-US",
    )

    assert request.message == ""
    assert request.file_ids == ["file_one"]

    with pytest.raises(ValueError):
        ConversationRequest(
            conversation_id="conv_attachments",
            file_ids=["file_one", "file_one"],
        )


def test_attachment_input_separates_signed_sdk_urls_from_stable_capture() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ConversationRequest(
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

    attachment_input = orchestrator._build_attachment_input(request, prepared_files)

    assert attachment_input.additional_instruction == ""
    assert "Exact extracted evidence." in attachment_input.current_message
    assert isinstance(attachment_input.model_content[1], ModelImagePart)
    assert attachment_input.model_content[1].url == "https://signed.example/image"
    assert isinstance(attachment_input.capture_content[1], CaptureFilePart)
    assert attachment_input.capture_content[1].file_id == "file_image"
    assert "signed.example" not in str(attachment_input.capture_content)

    user_request = orchestrator._build_user_request(request)
    assert [part.file_url.url for part in user_request.content_parts if part.file_url] == [
        "agentbreaker-file://file_image",
        "agentbreaker-file://file_document",
    ]


def test_attachment_only_instruction_uses_the_ui_locale_without_visible_fake_text() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ConversationRequest(
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

    attachment_input = orchestrator._build_attachment_input(request, [prepared_file])
    user_request = orchestrator._build_user_request(request)

    assert "Simplified Chinese" in attachment_input.additional_instruction
    assert user_request.content == ""
    assert len(user_request.content_parts) == 1
    assert user_request.content_parts[0].type == "file_url"


def test_plain_text_capture_uses_scalar_content() -> None:
    """A text-only typed message must remain scalar at the persistence boundary."""
    runtime = OpenAIAgentsSdkAdapter()
    captured = runtime._to_capture_message(Message(role="user", content="Question"))

    assert captured.role == "user"
    assert captured.content == "Question"
    assert captured.capture_content == ()


def test_plain_text_attachment_input_uses_empty_typed_parts() -> None:
    orchestrator = object.__new__(RuntimeOrchestrator)
    request = ConversationRequest(
        conversation_id="conv_plain_text",
        message="Continue from the previous answer",
    )

    attachment_input = orchestrator._build_attachment_input(request, [])

    assert attachment_input.current_message == "Continue from the previous answer"
    assert attachment_input.model_content == ()
    assert attachment_input.capture_content == ()
