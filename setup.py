from setuptools import setup, find_packages

setup(
    name="yuanmoxiao-cerebellum",
    version="0.1.0",
    description="Zero-model code understanding engine — pure Python BPE tokenizer, intent parser, 46 templates × 6 languages",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="YuanAn / YuanMoXiao",
    author_email="",
    url="https://github.com/YOUR_USERNAME/yuanmoxiao-cerebellum",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "deepseek": ["httpx"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="tokenizer bpe code-generation code-understanding zero-model offline",
)
