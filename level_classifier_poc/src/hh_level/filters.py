from __future__ import annotations

import re

DEV_INCLUDE = [
    "developer",
    "software",
    "programmer",
    "backend",
    "frontend",
    "fullstack",
    "java",
    "python",
    "c#",
    "c++",
    "golang",
    "go",
    "php",
    "javascript",
    "js",
    "typescript",
    "kotlin",
    "swift",
    "ios",
    "android",
    "unity",
    ".net",
    "node",
    "react",
    "vue",
    "angular",
    "django",
    "flask",
    "spring",
]

DEV_EXCLUDE = [
    "qa",
    "tester",
    "test",
    "аналитик",
    "support",
    "devops",
    "sre",
    "sysadmin",
    "администратор",
    "pm",
    "project",
    "product",
    "manager",
    "designer",
    "ux",
    "ui",
    "data analyst",
    "data scientist",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def is_it_developer(title: str | None, skills: str | None) -> bool:
    t = _norm(title or "")
    sk = _norm(skills or "")
    text = f"{t} {sk}"

    if any(x in text for x in DEV_EXCLUDE):
        return False
    return any(x in text for x in DEV_INCLUDE)
