from setuptools import setup

setup(
    name="layout_diffusion",
    py_modules=["layout_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
