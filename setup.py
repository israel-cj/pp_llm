from setuptools import setup, find_packages

setup(
    name="llmpipeline",
    version="0.0.1",
    packages=find_packages(),
    description="Create a pipeline based on LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Israel Campero Jurado and Joaquin Vanschoren",
    author_email="learsi1911@gmail.com",
    url="https://github.com/israel-cj/pp_llm.git",
    python_requires=">=3.10",
    install_requires=[        "pandas",
        "numpy",

        "openai",
        "openml",
        "scikit-learn",
        "stopit",
        "ipython",
        "mlxtend",
    ],
)

