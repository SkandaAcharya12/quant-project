"""Offline tests for the educational OpenAI-backed assistant."""
from __future__ import annotations

import unittest
from unittest import mock

import requests

from research_assistant import (
    AssistantConfig,
    AssistantError,
    AssistantMessage,
    MarketResearchAssistant,
)
from research_assistant.agent import OPENAI_RESPONSES_URL


class _FakeResponse:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        return self._body


class ResearchAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = MarketResearchAssistant(
            AssistantConfig(api_key="test-key", model="test-model")
        )

    @mock.patch("research_assistant.agent.requests.post")
    def test_answer_uses_bounded_responses_api_and_extracts_message_text(
        self, post: mock.MagicMock
    ) -> None:
        post.return_value = _FakeResponse(
            200,
            {
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "VaR is a loss threshold."},
                            {"type": "output_text", "text": "Losses can be larger."},
                        ],
                    },
                ]
            },
        )
        messages = [AssistantMessage(role="user", content="Explain VaR")]

        answer = self.assistant.answer(messages, "Market: Japan; date: 2026-11-10.")

        self.assertEqual(answer, "VaR is a loss threshold.\n\nLosses can be larger.")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], OPENAI_RESPONSES_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["store"])
        self.assertIn("Market: Japan", payload["instructions"])
        self.assertEqual(payload["input"], [{"role": "user", "content": "Explain VaR"}])
        self.assertEqual(kwargs["timeout"], 35.0)

    @mock.patch("research_assistant.agent.requests.post")
    def test_answer_limits_history_and_rejects_non_user_last_turn(
        self, post: mock.MagicMock
    ) -> None:
        assistant = MarketResearchAssistant(
            AssistantConfig(api_key="test-key", max_history_messages=2)
        )
        post.return_value = _FakeResponse(
            200,
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]},
        )
        conversation = [
            AssistantMessage(role="user", content="Old question"),
            AssistantMessage(role="assistant", content="Old answer"),
            AssistantMessage(role="user", content="Current question"),
        ]

        assistant.answer(conversation, "")
        self.assertEqual(
            post.call_args.kwargs["json"]["input"],
            [
                {"role": "assistant", "content": "Old answer"},
                {"role": "user", "content": "Current question"},
            ],
        )
        with self.assertRaises(AssistantError):
            assistant.answer([AssistantMessage(role="assistant", content="Not a question")], "")

    @mock.patch("research_assistant.agent.requests.post")
    def test_answer_handles_rate_limit_and_timeouts(self, post: mock.MagicMock) -> None:
        message = AssistantMessage(role="user", content="How does this work?")
        post.return_value = _FakeResponse(429, {})
        with self.assertRaisesRegex(AssistantError, "rate-limited"):
            self.assistant.answer([message], "")

        post.side_effect = requests.Timeout
        with self.assertRaisesRegex(AssistantError, "timed out"):
            self.assistant.answer([message], "")

    @mock.patch("research_assistant.agent.requests.post")
    def test_answer_handles_rejected_key_and_empty_output(self, post: mock.MagicMock) -> None:
        message = AssistantMessage(role="user", content="How does this work?")
        post.return_value = _FakeResponse(401, {})
        with self.assertRaisesRegex(AssistantError, "API key was rejected"):
            self.assistant.answer([message], "")

        post.return_value = _FakeResponse(200, {"output": []})
        with self.assertRaisesRegex(AssistantError, "did not return a text answer"):
            self.assistant.answer([message], "")


if __name__ == "__main__":
    unittest.main()

