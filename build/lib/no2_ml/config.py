from pathlib import Path
import yaml
from no2_ml.features import feature_spec
from no2_ml.models.registry import model_names
from no2_ml.preprocessing.quality import QCConfig


def read_yaml(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        value=yaml.safe_load(stream)
    if not isinstance(value,dict):raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def load_experiment(path):
    path=Path(path).resolve()
    cfg=read_yaml(path)
    allowed={'input_manifest','output_dir','preprocessing_config','features_config','models_config','validation_config','feature_set','models','seed','purpose'}
    if set(cfg)-allowed:raise ValueError(f"Unknown experiment keys: {sorted(set(cfg)-allowed)}")
    for key in ['input_manifest','output_dir','preprocessing_config','features_config','models_config','validation_config']:
        if not cfg.get(key):raise ValueError(f"Missing experiment config field {key}")
    def resolved(key):return (path.parent/cfg[key]).resolve()
    qc=read_yaml(resolved('preprocessing_config'));QCConfig(**qc)
    fc=read_yaml(resolved('features_config'))
    version=feature_spec(cfg.get('feature_set',fc['default'])).version
    if version not in fc['available']:raise ValueError("Feature version not enabled by features config")
    mc=read_yaml(resolved('models_config'))
    names=cfg.get('models',list(mc))
    if not isinstance(names,list) or not names or len(set(names))!=len(names) or not set(names).issubset(model_names()):
        raise ValueError("Invalid registered model selection")
    if not set(names).issubset(mc):raise ValueError("Selected model missing from model configuration")
    seed=cfg.get('seed',42)
    if not isinstance(seed,int) or isinstance(seed,bool):raise ValueError("seed must be an integer")
    purpose=cfg.get('purpose','research')
    if purpose not in {'research','software_test'}:raise ValueError("purpose must be research or software_test")
    return dict(input_manifest=str(resolved('input_manifest')),output_dir=str(resolved('output_dir')),
                feature_version=version,qc_config=qc,model_configs=mc,models=names,seed=seed,
                validation=read_yaml(resolved('validation_config')),purpose=purpose)
