## This repo aims to create a pipeline to evaluate, compare and try on complex and compressed neural network methods.

Said pipeline should not be limited to evaluating a set of spesific networks instead it needs to aim for independant modules,
which could get changed/replaced without breaking other available modules.

```
my_nn_pipeline/
│
├── configs/
│   └── ...
├── data/
│   ├── __init__.py
│   └── build.py
│   └── ...
├── evaluator/
│   ├── __init__.py
│   ├── build.py
│   └── ...
├── experiments/
│   ├── __init__.py
│   └── mlflow.py
├── losses/
│   ├── __init__.py
│   ├── build.py
│   └── ...
├── metrics/
│   ├── __init__.py
│   ├── build.py
│   └── ...
├── models/
│   ├── __init__.py
│   ├── build.py
│   └── ...
├── optimizers/
│   ├── __init__.py
│   ├── build.py
│   └── ...
├── scripts/                # maybe to launch jobs (future work)
│   └── ...
├── trainer/
│   ├── __init__.py
│   ├── trainer.py          # base trainer class
│   └── ...
├── utils/                  # logger, seed etc. utility
│   ├── logger.py
│   ├── seed.py
│   ├── checkpoint.py
│   └── ...
│
├── requirements.txt
├── main.py (setup.py)
└── README.md
```