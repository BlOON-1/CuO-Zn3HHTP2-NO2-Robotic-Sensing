from pathlib import Path
import json
from no2_ml.config import load_experiment
from no2_ml.io.manifest import features_from_manifest
from no2_ml.evaluation.export import export_csv
from no2_ml.prediction.artifacts import save_artifact, jsonable
from .nested import nested_validate, refit_final


def train_from_config(path):
    cfg=load_experiment(path)
    output=Path(cfg['output_dir'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    try:
        table=features_from_manifest(cfg['input_manifest'],feature_version=cfg['feature_version'],qc_config=cfg['qc_config'])
        args=dict(feature_version=cfg['feature_version'],models=cfg['models'],model_configs=cfg['model_configs'],validation=cfg['validation'],seed=cfg['seed'])
        nested=nested_validate(table,**args)
        name,pipeline,refit_logs=refit_final(table,**args)
        export_csv(table,output/'features.csv')
        export_csv(nested.predictions,output/'out_of_fold_predictions.csv')
        export_csv(nested.metrics,output/'metrics.csv')
        export_csv(nested.splits,output/'validation_splits.csv')
        export_csv(nested.inner_splits,output/'inner_validation_splits.csv')
        (output/'hyperparameter_search.json').write_text(json.dumps(jsonable(nested.searches+refit_logs),indent=2,allow_nan=False),encoding='utf8')
        save_artifact(pipeline,output/'artifact',model_name=name,feature_version=cfg['feature_version'],
                      channel=int(table.channel.iloc[0]),training_config=cfg,qc_config=cfg['qc_config'])
    except Exception as exc:
        (output/'run_status.json').write_text(json.dumps({'status':'failed','error':str(exc),'purpose':cfg['purpose']},indent=2),encoding='utf8')
        raise
    (output/'run_status.json').write_text(json.dumps({'status':'complete','purpose':cfg['purpose'],
        'prediction_mode':'offline_reviewed_cycle','refit_model':name,'evaluation':'grouped nested OOF, not external test'},indent=2),encoding='utf8')
    return output
