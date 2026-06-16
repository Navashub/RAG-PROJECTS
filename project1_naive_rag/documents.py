# documents.py
# ─────────────────────────────────────────────────────────
# The knowledge base our RAG system will answer questions about.
#
# In a real system this would be:
#   - PDFs loaded from disk
#   - Pages scraped from a website
#   - Rows pulled from a database
#   - Files from an S3 bucket
#
# For now, we use plain dicts so the structure is obvious.
# Each document has:
#   "source" → where it came from (shown in answers so users can verify)
#   "text"   → the actual content to index
# ─────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "source": "remote_work_policy.txt",
        "text": """
Remote Work Policy - TechCorp Inc.

Effective January 2024, TechCorp allows all employees to work remotely up to 3 days per week.
Employees must be available between 10am and 3pm in their local timezone — these are core hours.
A stable internet connection of at least 50 Mbps is required for remote work.
Employees working remotely must attend all scheduled team meetings via video call.
Remote work is not permitted during the first 90 days of employment.
Requests for fully remote work must be approved by both the direct manager and HR.
Equipment such as laptops and monitors may be requested through the IT portal.
Any data security incidents must be reported to the security team within 1 hour.
        """,
    },
    {
        "source": "employee_benefits.txt",
        "text": """
Employee Benefits Guide - TechCorp Inc.

Health Insurance: TechCorp covers 90% of medical premiums for employees and 70% for dependents.
All employees receive 20 days of paid vacation annually. After 5 years, this increases to 25 days.
TechCorp provides a $1,000 annual learning budget per employee for courses, books, or conferences.
Parental leave: 16 weeks fully paid for primary caregivers, 6 weeks for secondary caregivers.
The company matches 401k contributions up to 4% of salary.
Employees can access mental health support through the Employee Assistance Program (EAP).
A $500 annual wellness stipend can be used for gym memberships, fitness equipment, or apps.
Stock options vest over a 4-year period with a 1-year cliff.
        """,
    },
    {
        "source": "engineering_standards.txt",
        "text": """
Engineering Standards and Practices - TechCorp

All code must pass automated tests before being merged into the main branch.
Pull requests require at least 2 approvals from senior engineers.
Every service must have a runbook documenting how to debug and recover from incidents.
Deployments to production happen every Tuesday and Thursday unless it is an emergency patch.
All APIs must follow the company REST guidelines and include OpenAPI documentation.
Database migrations must be backward-compatible and reversible.
Secrets must never be committed to source control — use the company secrets manager.
Performance budgets: API endpoints must respond within 200ms at the 95th percentile.
        """,
    },
]