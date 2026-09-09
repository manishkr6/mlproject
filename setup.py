from pathlib import Path

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> list[str]:
    """Return the list of requirements from a file, ignoring editable flags and comments."""
    with open(file_path, encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith(("-e", "#"))
        ]


BASE_DIR = Path(__file__).resolve().parent


setup(
  name="Mlproject",
  version="0.0.1",
  author="Manish",
  author_email="manishkr8927@gmail.com",
  packages=find_packages(),
  install_requires=get_requirements(BASE_DIR / "requirements.txt")
)