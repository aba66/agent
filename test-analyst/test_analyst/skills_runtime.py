from __future__ import annotations

from pathlib import Path

from .models import SkillDefinition


class SkillRegistry:
    """Load lightweight project-local skills used by the agents."""

    def __init__(self, skill_root: str | Path) -> None:
        self.skill_root = Path(skill_root).resolve()

    def load(self, name: str) -> SkillDefinition:
        skill_dir = self.skill_root / name
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return SkillDefinition(name=name, instruction="")

        references: list[str] = []
        reference_dir = skill_dir / "references"
        if reference_dir.exists():
            for ref_file in sorted(reference_dir.glob("*.md"))[:3]:
                references.append(ref_file.read_text(encoding="utf-8").strip())

        return SkillDefinition(
            name=name,
            instruction=skill_file.read_text(encoding="utf-8").strip(),
            references=references,
            path=skill_dir,
        )

    def render_bundle(self, names: list[str]) -> str:
        parts: list[str] = []
        for name in names:
            skill = self.load(name)
            if not skill.instruction:
                continue
            parts.append(f"[Skill:{name}]\n{skill.instruction}")
            for index, reference in enumerate(skill.references, start=1):
                parts.append(f"[Skill:{name}:reference:{index}]\n{reference}")
        return "\n\n".join(parts).strip()
