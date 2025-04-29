from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="speech-to-text-cli",
    version="0.1.0",
    author="Oxidane",
    author_email="example@example.com",
    description="A CLI tool for transcribing audio and video using Whisper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oxidane/speech-to-text-cli",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tqdm>=4.64.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "full": [
            "torch>=1.13.1",
            "openai-whisper>=20230314",
        ],
    },
    entry_points={
        "console_scripts": [
            "sttcli=speechtotextcli.cli:main",
        ],
    },
) 