from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements():
    requirements = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="carla-risk-aware-rl",
    version="0.1.0",
    description="Risk-aware reinforcement learning scaffold for CARLA motion planning.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["rarl", "rarl.*", "tests", "tests.*", "a", "a.*"]),
    python_requires=">=3.8,<3.9",
    install_requires=read_requirements(),
)
