## This repo aims to create a pipeline to evaluate, compare and <del> try on complex and compressed </del> evaluate neural network methods.

### NOTE: Consider updating checks to allow submodules within the `models` folder. Organizing models by task can help reduce the number of `.py` files in a single directory.

The pipeline should support modularity, enabling independent modules that can be swapped or updated without affecting others.

<b>Directory structure:</b>
```
└── denizberkin-torch_pipeline/
    ├── README.md
    ├── clivis.py
    ├── complexnn.py
    ├── cosnn.py
    ├── main.py
    ├── requirements.txt
    ├── .pre-commit-config.yaml
    ├── configs/
    │   ├── classification.yaml
    │   ├── detection.yaml
    │   └── segmentation.yaml
    │       ...
    ├── data/
    │   ├── __init__.py
    │   ├── base.py
    │   └── build.py
    │       ...
    ├── losses/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── build.py
    │   └── focal.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── build.py
    │   └── sklearn_wrapper.py
    ├── models/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── build.py
    │   ├── classification/
    │   │   └── ...
    │   └── segmentation/
    │       └── ...
    ├── optimizers/
    │   ├── __init__.py
    │   └── build.py
    │       ...
    ├── schedulers/
    │   ├── __init__.py
    │   ├── base.py
    │   └── build.py
    │       ...
    ├── trackers/
    │   ├── __init__.py
    │   └── base.py
    │       ...
    ├── trainer/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── build.py
    │   └── trainer.py
    │       ...
    └── utils/
        ├── config.py
        ├── constants.py
        ├── logger.py
        ├── perf.py
        ├── schema.py
        ├── seed.py
        └── utils.py
```
