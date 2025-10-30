import re
from pathlib import Path

# Patterns we consider invalid for src-layout package imports
BARE_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+par\b"),
    re.compile(r"^\s*from\s+(field|mesh|makedustopac|beam|polar)\s+import\b"),
    re.compile(r"^\s*from\s+(dust_density|dust_temperature|dust_opacities)\s+import\b"),
    re.compile(r"^\s*from\s+(gas_density|gas_temperature|gas_velocity)\s+import\b"),
]

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "fargo2radmc3d"


def test_no_bare_intrapackage_imports():
    assert SRC_ROOT.is_dir()
    offenders = []
    for py in SRC_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in BARE_IMPORT_PATTERNS:
                if pat.search(line):
                    offenders.append((py.relative_to(SRC_ROOT), i, line.strip()))
    assert offenders == [], f"Found legacy bare imports that should use package-relative/absolute paths: {offenders}"
