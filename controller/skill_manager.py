"""
Handles everything to do with skills as files on disk:
listing them, loading their code, saving new ones, and running them.

A "skill" is just a .py file in skills/ or generated_skills/ that
defines a top-level `run(**kwargs)` function.
"""

import os
import importlib.util

SKILLS_DIRS = ["skills", "generated_skills"]


def list_skills():
    skills = []
    for d in SKILLS_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".py") and not fname.startswith("_"):
                skills.append(fname[:-3])
    return sorted(set(skills))


def _find_skill_path(name: str):
    for d in SKILLS_DIRS:
        path = os.path.join(d, f"{name}.py")
        if os.path.isfile(path):
            return path
    return None


def get_skill_code(name: str):
    path = _find_skill_path(name)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_skill(name: str, code: str) -> str:
    os.makedirs("generated_skills", exist_ok=True)
    path = os.path.join("generated_skills", f"{name}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def run_skill(name: str, arguments: dict | None = None) -> dict:
    arguments = arguments or {}
    path = _find_skill_path(name)
    if not path:
        return {"ok": False, "error": f"Skill '{name}' not found."}

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load skill: {e}"}

    if not hasattr(module, "run"):
        return {"ok": False, "error": "Skill has no run() function."}

    try:
        result = module.run(**arguments)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
