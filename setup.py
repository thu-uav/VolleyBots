from setuptools import find_packages, setup

setup(
    name="volley_bots",
    version="0.1.0",
    author="volleybots",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "plotly",
        "einops",
        "av",
        "pandas",
        "h5py",
        "filterpy",
        "usd-core==23.2",
        "numpy==1.23.5",
        "urllib3==1.26.18",
        "seaborn"
    ],
)
