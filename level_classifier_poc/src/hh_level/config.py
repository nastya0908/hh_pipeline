from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Columns:
    title: Optional[str] = None
    skills: Optional[str] = None
    description: Optional[str] = None
    salary: Optional[str] = None
    city: Optional[str] = None
    age: Optional[str] = None
    experience: Optional[str] = None


DEFAULT_COL_CANDIDATES = {
    "title": [
        "title",
        "position",
        "job_title",
        "vacancy",
        "profession",
        "name",
        "ищет работу на должность:",
        "ищет работу на должность",
    ],
    "skills": [
        "skills",
        "key_skills",
        "skill_set",
        "stack",
        "ключевые навыки",
        "навыки",
    ],
    "description": [
        "description",
        "about",
        "summary",
        "text",
        "experience_text",
        "обо мне",
        "описание",
        "резюме",
    ],
    "salary": [
        "salary",
        "expected_salary",
        "compensation",
        "pay",
        "зп",
        "зарплата",
    ],
    "city": [
        "city",
        "area",
        "location",
        "town",
        "город",
    ],
    "age": [
        "age",
        "years_old",
        "пол, возраст",
    ],
    "experience": [
        "experience",
        "exp",
        "work_experience",
        "total_experience",
        "experience_years",
        "опыт (двойное нажатие для полной версии)",
        "опыт",
    ],
}
