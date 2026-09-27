"""Code Review Agent.

Analyzes a pull request diff against patterns that humans have flagged before,
then returns inline comment suggestions (file/line/severity/category/message)
that can be posted back to the review using the GitHub adapter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.agents.base import AgentContext, AgentResponse, AgentType, BaseAgent
from app.core.config import Settings
from app.core.utils import sha256_hex, truncate
from app.llm.gateway import BaseLLMGateway
from app.llm.prompts import CODE_REVIEW_PROMPT
from app.memory.service import MemoryService, RetrievedContext

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_SEVERITIES = {"error", "warning", "info"}


class CodeReviewAgent(BaseAgent):
    #: How much of the raw diff is sent to the model.
    MAX_DIFF_CHARS = 8000

    def __init__(
        self,
        memory_service: MemoryService,
        llm_gateway: BaseLLMGateway,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            agent_type=AgentType.CODE_REVIEW,
            memory_service=memory_service,
            llm_gateway=llm_gateway,
            system_prompt=CODE_REVIEW_PROMPT,
        )
        self.settings = settings or memory_service.settings

    async def retrieve_context(self, context: AgentContext) -> list[RetrievedContext]:
        """Find historical review patterns that resemble this diff."""
        payload = context.payload
        query = payload.get("diff_summary") or context.query
        return await self.memory.find_similar_review_patterns(
            diff_summary=query,
            repo=payload.get("repo"),
            top_k=8,
            threshold=self.settings.VECTOR_SIMILARITY_THRESHOLD,
        )

    async def generate_response(self, context: AgentContext) -> AgentResponse:
        historical = context.historical_context
        messages = self._build_prompt(self._with_analysis_query(context), historical)

        response_text = await self.llm.chat_completion(
            messages=messages,
            temperature=0.1,  # very low: code analysis must be reproducible
            max_tokens=2000,
        )

        comments = self._parse_inline_comments(response_text, historical)
        avg_similarity = self._average_similarity(historical)
        confidence = round(min(0.9, 0.55 + 0.35 * avg_similarity), 3)

        follow_up: list[dict[str, Any]] = []
        if comments:
            follow_up.append(
                {
                    "action": "human_review_required",
                    "reason": "Agent findings must be confirmed by a human reviewer",
                    "priority": "high"
                    if any(c.get("severity") == "error" for c in comments)
                    else "medium",
                }
            )
        else:
            follow_up.append(
                {
                    "action": "no_action",
                    "reason": "No issues detected against historical patterns",
                    "priority": "low",
                }
            )

        return AgentResponse(
            response=response_text,
            confidence=confidence,
            suggestions=comments,
            historical_matches=self._historical_match_ids(historical),
            follow_up_actions=follow_up,
            metadata={
                "agent_type": self.agent_type.value,
                "patterns_matched": len(historical),
                "inline_comments_count": len(comments),
                "avg_similarity": round(avg_similarity, 3),
                "diff_hash": context.payload.get("diff_hash")
                or sha256_hex(context.payload.get("diff", "") or context.query),
            },
        )

    def _with_analysis_query(self, context: AgentContext) -> AgentContext:
        """Return a shallow copy whose query is the rich PR analysis prompt."""
        payload = context.payload
        diff = payload.get("diff") or ""
        parts = [
            f"PR #{payload.get('pr_number', payload.get('pr_id', 'unknown'))} "
            f"in {payload.get('repo', 'unknown')}: {payload.get('title', '')}",
            f"Author: {payload.get('author', 'unknown')}",
            f"Description: {truncate(payload.get('description') or '', 1000)}",
            f"Diff summary: {payload.get('diff_summary') or 'not provided'}",
            "Full diff:",
            truncate(diff, self.MAX_DIFF_CHARS) or "not provided",
        ]
        return AgentContext(
            query="\n".join(parts),
            payload=payload,
            historical_context=context.historical_context,
            user_id=context.user_id,
            team_id=context.team_id,
            conversation_history=context.conversation_history,
        )

    # ------------------------------------------------------------- parsing
    @staticmethod
    def extract_json(text: str) -> Optional[dict[str, Any]]:
        """Extract the first JSON object from a fenced block or the whole text."""
        if not text:
            return None
        for candidate in list(_JSON_BLOCK_RE.findall(text)) + [text.strip()]:
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _parse_inline_comments(
        self, response: str, historical: list[RetrievedContext]
    ) -> list[dict[str, Any]]:
        """Normalise the model output into review comment records.

        Primary path is the JSON block requested by the system prompt; if the
        model answered in prose we fall back to bullet parsing so the review is
        never silently empty.
        """
        payload = self.extract_json(response)
        raw_comments = payload.get("inline_comments") if payload else None

        comments: list[dict[str, Any]] = []
        if isinstance(raw_comments, list):
            for index, item in enumerate(raw_comments):
                if not isinstance(item, dict):
                    continue
                reference = item.get("historical_reference_id")
                if not reference and index < len(historical) and historical[index].source_id:
                    reference = str(historical[index].source_id)
                comments.append(self._normalize_comment(item, reference))
        else:
            comments = self._bullet_fallback(response, historical)

        return [comment for comment in comments if comment]

    @staticmethod
    def _normalize_comment(item: dict[str, Any], reference: Optional[str]) -> dict[str, Any]:
        severity = str(item.get("severity", "info")).lower()
        if severity not in _SEVERITIES:
            severity = "info"
        try:
            line_number = int(item.get("line_number", 1) or 1)
        except (TypeError, ValueError):
            line_number = 1
        try:
            confidence = float(item.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "file_path": str(item.get("file_path") or item.get("file") or "unknown"),
            "line_number": max(1, line_number),
            "severity": severity,
            "category": str(item.get("category", "style")).lower(),
            "message": str(item.get("message") or item.get("comment") or "").strip(),
            "confidence": round(min(max(confidence, 0.0), 1.0), 3),
            "historical_reference_id": reference,
        }

    @staticmethod
    def _bullet_fallback(
        response: str, historical: list[RetrievedContext]
    ) -> list[dict[str, Any]]:
        """Fallback parser: ``- **file:line** severity - message`` bullets."""
        comments: list[dict[str, Any]] = []
        pattern = re.compile(
            r"^[-*]\s+\**(?P<file>[^\s:*]+)(?::(?P<line>\d+))?\**\s*(?P<rest>.*)$"
        )
        for line in response.splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            rest = match.group("rest").strip(" -*")
            comments.append(
                {
                    "file_path": match.group("file"),
                    "line_number": int(match.group("line") or 1),
                    "severity": "info",
                    "category": "style",
                    "message": rest or "Review this change",
                    "confidence": 0.5,
                    "historical_reference_id": str(historical[0].source_id)
                    if historical and historical[0].source_id
                    else None,
                }
            )
        return comments

    async def record_outcome(
        self,
        context: AgentContext,
        response: AgentResponse,
        feedback: Optional[str] = None,
    ) -> None:
        suggestion = await self.memory.record_suggestion(
            agent_type=self.agent_type.value,
            query_context=truncate(context.query, 2000),
            suggestion=response.response,
            confidence=response.confidence,
            historical_matches=response.historical_matches,
            review_id=context.payload.get("review_id"),
        )
        if feedback:
            await self.memory.record_feedback(suggestion.id, feedback)
