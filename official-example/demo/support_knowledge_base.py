"""A tiny local knowledge base for the customer support email agent."""

from __future__ import annotations

from typing import Any


SUPPORT_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc-password-reset",
        "topic": "account",
        "title": "Password Reset Guide",
        "content": (
            "Customers can reset their password from the login page by clicking "
            "'Forgot Password'. A reset email is sent within 5 minutes. If the "
            "email does not arrive, ask the customer to check spam and verify "
            "their account email address."
        ),
        "tags": ["password", "reset", "login", "forgot password", "account"],
    },
    {
        "id": "doc-export-pdf-bug",
        "topic": "bug_report",
        "title": "Known Issue: PDF Export Crash",
        "content": (
            "Version 3.2.1 has a known bug where PDF export may crash for large "
            "projects with embedded images. Support should collect the app "
            "version, operating system, sample file size, and crash timestamp. "
            "Offer CSV export as a temporary workaround."
        ),
        "tags": ["export", "pdf", "crash", "bug", "workaround"],
    },
    {
        "id": "doc-billing-double-charge",
        "topic": "billing",
        "title": "Billing Policy for Duplicate Charges",
        "content": (
            "Duplicate subscription charges should be treated as urgent. Support "
            "must apologize, confirm the billing email, collect the last four "
            "digits of the payment method, and escalate the case to billing "
            "operations within the same business day."
        ),
        "tags": ["billing", "charged twice", "duplicate charge", "refund"],
    },
    {
        "id": "doc-feature-requests",
        "topic": "feature_request",
        "title": "Feature Request Intake Process",
        "content": (
            "Thank the customer for the idea, capture the use case, affected "
            "platform, and business impact, and log the request for product "
            "review. Avoid promising delivery dates."
        ),
        "tags": ["feature", "request", "dark mode", "roadmap", "product"],
    },
    {
        "id": "doc-api-504",
        "topic": "technical_issue",
        "title": "Troubleshooting Intermittent API 504 Errors",
        "content": (
            "For intermittent 504 errors, gather request IDs, timestamps, retry "
            "frequency, region, and endpoint names. Recommend exponential backoff "
            "with retries and verify whether the customer is hitting rate or "
            "timeout limits. Escalate to technical support if production traffic "
            "is affected."
        ),
        "tags": ["api", "504", "gateway timeout", "integration", "retry"],
    },
    {
        "id": "doc-follow-up-policy",
        "topic": "operations",
        "title": "Follow-up Scheduling Policy",
        "content": (
            "Schedule a follow-up when a case depends on another team, customer "
            "confirmation, or monitoring results. Standard follow-up windows are "
            "1 day for urgent issues, 2 days for technical investigations, and "
            "3 days for product feedback."
        ),
        "tags": ["follow-up", "sla", "schedule", "operations"],
    },
]

