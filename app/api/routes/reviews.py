"""Code review routes: browse reviews, re-run analysis, give feedback."""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    UserContext,
    get_adapters,
    get_current_user,
    get_orchestrator,
    get_session,
)
from app.api.schemas import (
    AgentQueryResponse,
    CodeReviewResponse,
    FeedbackRequest,
    FeedbackResponse,
    ReviewListResponse,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.utils import utcnow
from app.memory.models import AgentSuggestion, CodeReview, ReviewComment, ReviewPattern
from app.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["code-reviews"])


def _ensure_team_access(user: UserContext, team_id: Optional[UUID]) -> None:
    if not user.is_admin and user.team_id and team_id and team_id != user.team_id:
        raise PermissionDeniedError("Access to another team's review is not permitted")


async def _hydrate(
    session: AsyncSession, reviews: list[CodeReview]
) -> list[CodeReviewResponse]:
    """Attach comments and patterns to review rows without N+1 surprises."""
    responses: list[CodeReviewResponse] = []
    for review in reviews:
        comments = list(
            (
                await session.execute(
                    select(ReviewComment)
                    .where(ReviewComment.review_id == review.id)
                    .order_by(ReviewComment.file_path, ReviewComment.line_number)
                )
            ).scalars().all()
        )
        patterns = list(
            (
                await session.execute(
                    select(ReviewPattern)
                    .where(ReviewPattern.review_id == review.id)
                    .order_by(ReviewPattern.pattern_type)
                )
            ).scalars().all()
        )
        response = CodeReviewResponse.model_validate(review)
        response.comments = comments  # type: ignore[assignment]
        response.patterns = [
            {
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "frequency": pattern.frequency,
            }
            for pattern in patterns
        ]
        responses.append(response)
    return responses


@router.get("", response_model=ReviewListResponse)
async def list_reviews(
    repo: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None, alias="status"),
    flagged_only: bool = Query(False, description="Only reviews with agent findings"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> ReviewListResponse:
    """List pull request reviews, scoped to the caller's team unless admin."""
    query = select(CodeReview)
    if not user.is_admin and user.team_id:
        query = query.where(CodeReview.team_id == user.team_id)
    if repo:
        query = query.where(CodeReview.repo == repo)
    if author:
        query = query.where(CodeReview.author == author)
    if review_status:
        query = query.where(CodeReview.status == review_status)
    if flagged_only:
        query = query.where(CodeReview.reviewer_agent_notes.is_not(None))

    total = int(await db.scalar(select(func.count()).select_from(query.subquery())) or 0)
    result = await db.execute(
        query.order_by(desc(CodeReview.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    reviews = list(result.scalars().all())
    return ReviewListResponse(
        reviews=await _hydrate(db, reviews),
        total=total,
    )


@router.get("/{review_id}", response_model=CodeReviewResponse)
async def get_review(
    review_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> CodeReviewResponse:
    """Review detail with inline comments and matched patterns."""
    review = await db.get(CodeReview, review_id)
    if review is None:
        raise NotFoundError("Review not found")
    _ensure_team_access(user, review.team_id)
    hydrated = await _hydrate(db, [review])
    return hydrated[0]


@router.post("/{review_id}/analyze", response_model=AgentQueryResponse)
async def analyze_review(
    review_id: UUID,
    diff: Optional[str] = Query(
        default=None, description="Optional raw diff (fetched from the provider when omitted)"
    ),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    adapters: dict[str, Any] = Depends(get_adapters),
) -> AgentQueryResponse:
    """(Re)run the Code Review Agent for a stored pull request."""
    review = await db.get(CodeReview, review_id)
    if review is None:
        raise NotFoundError("Review not found")
    _ensure_team_access(user, review.team_id)

    diff_text = await _fetch_diff(adapters, review, diff)
    if not diff_text:
        raise NotFoundError(
            "No diff available: pass ?diff=... or configure GITHUB/GITLAB credentials"
        )

    response = await orchestrator.process_pr_event(
        {
            "number": review.pr_id,
            "repo": review.repo,
            "title": review.title,
            "description": review.description,
            "author": review.author,
            "diff": diff_text,
            "review_id": str(review.id),
        },
        review.team_id,
    )
    review.reviewer_agent_notes = response.response
    await db.flush()

    payload = response.as_dict()
    payload["agent_type"] = str(payload.get("metadata", {}).get("agent_type", "code_review"))
    return AgentQueryResponse(**payload)


@router.post("/{review_id}/feedback", response_model=FeedbackResponse)
async def review_feedback(
    review_id: UUID,
    payload: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> FeedbackResponse:
    """Attach human reviewer verdict to the review and its latest agent suggestion."""
    review = await db.get(CodeReview, review_id)
    if review is None:
        raise NotFoundError("Review not found")
    _ensure_team_access(user, review.team_id)

    review.human_feedback = payload.feedback

    suggestion = (
        await db.execute(
            select(AgentSuggestion)
            .where(AgentSuggestion.review_id == review.id)
            .order_by(desc(AgentSuggestion.created_at))
        )
    ).scalars().first()
    if suggestion is None:
        raise NotFoundError("No agent suggestion recorded for this review")

    suggestion.user_feedback = payload.feedback
    suggestion.feedback_at = utcnow()
    return FeedbackResponse(status="recorded", suggestion_id=suggestion.id)


async def _fetch_diff(
    adapters: dict[str, Any], review: CodeReview, diff: Optional[str] = None
) -> str:
    """Use the supplied diff, else fetch it from a configured git provider."""
    if diff:
        return diff
    github = adapters.get("github")
    if github is not None and github.enabled:
        try:
            return await github.fetch_pr_diff(review.repo, review.pr_id)
        except Exception as exc:  # noqa: BLE001 - provider access is best effort
            logger.warning("could not fetch diff from github: %s", exc)
    gitlab = adapters.get("gitlab")
    if gitlab is not None and gitlab.enabled:
        try:
            return await gitlab.fetch_mr_changes(review.repo, review.pr_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not fetch diff from gitlab: %s", exc)
    return ""
