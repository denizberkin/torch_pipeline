# Some notes
This repo aims to create a pipeline to evaluate, compare and <del> try on complex and compressed </del> evaluate neural network methods. <br>
I tried to make it as modular as possible, allowing for experimentation with components and comparison of multiple methods. \<still needs work>

# Cloning the repository
```bash
git clone https://github.com/denizberkin/torch_pipeline.git
cd torch_pipeline
```

# Installing dependencies
> [!NOTE] 
You might want to comment out some of the dependencies but be aware that it might still break the pipeline.
### with just pip
```bash
pip install -r requirements.txt
```

### with venv
```bash
pip install virtualenv
python -m venv $your_env_name
source $your_env_name/bin/activate
pip install -r requirements.txt
```

### with conda
```bash
conda create -n torch_pipeline python=$your_version
conda activate torch_pipeline
pip install -r requirements.txt
```

# Running the pipeline
```bash
python main.py --config configs/$your_config_filename.yaml
# or
python main.py --config classification.yaml  # it still checks configs/ folder 
```
> [!NOTE] 
There are already some pre-defined configs in the `configs/` folder, you can play around with them as well. 
<br>

# Adding a new module
To create a new module, you can check `/base.py` file in the respective folder. It will contain the base class for the module and what methods to be implemented. <br>
> [!NOTE] 
Without exception, all modules will require a `get_alias` method which helps to identify the module from the config file. 
<br>

### Example of defining a new model;
1. check [base.py](models/base.py) to see what methods need to be implemented
2. create module file `models/<task_type>/<model_file>.py`
3. inherit from the base class and implement your logic
```py
from models.base import BaseModel
class CustomModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
    def forward(self, x):
        return x  # your forward logic
    def get_alias(cls):
        return "alias-to-be-used-in-config"
```
4. Add it in config file
```yaml
model:
  - name: alias-to-be-used-in-config
    pretrained: false
    pretrained_path: null
    kwargs:
      arg1: value1
      arg2: value2
```
> [!NOTE] 
You can pass any arguments to the kwargs section but your module must define the `**kwargs` parameter in the `__init__` method. Check [config schema](utils/schema.py) or [example config](configs/classification.yaml) for details.
<br>
<br>

# **Directory structure:**

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


# Necessary additions
- [ ] Add functions to check modules currently available (models, optims, losses, metrics etc.)
- [ ] Add mixed precision training support
- [ ] Add augmentation module, custom augmentation support
- [ ] Add experiment tracking support (initially mlflow)
- [ ] Add huggingface datasets support
- [ ] Add huggingface models support
- [ ] Update current experiment artifacts & save logic, it is messy code
- [ ] Fix [pascalvoc](data/pascalvoc.py) dataloader
