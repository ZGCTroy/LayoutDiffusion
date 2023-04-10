from setuptools import setup

setup(
    name="dpm_solver",
    py_modules=["dpm_solver"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
