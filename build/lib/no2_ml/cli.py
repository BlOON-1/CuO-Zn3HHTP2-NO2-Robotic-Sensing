"""Explicit commands; imports do not execute actions."""
import argparse
from pathlib import Path


def build_parser():
    parser=argparse.ArgumentParser(prog='no2-ml',description='Versioned offline NO2 regression; no automatic research-data discovery')
    sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--config',type=Path,required=True)
    extract=sub.add_parser('extract');extract.add_argument('--config',type=Path,required=True);extract.add_argument('--output',type=Path,required=True)
    pred=sub.add_parser('predict');pred.add_argument('--manifest',type=Path,required=True);pred.add_argument('--artifact',type=Path,required=True)
    pred.add_argument('--output',type=Path,required=True);pred.add_argument('--trusted-model',action='store_true',help='Only for artifacts you trust; joblib can execute code')
    pred.add_argument('--feature-version',default=None)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--input-csv',type=Path,required=True);evaluate.add_argument('--output',type=Path,required=True)
    render=sub.add_parser('render')
    sources=render.add_mutually_exclusive_group(required=True)
    sources.add_argument('--input-csv',type=Path)
    sources.add_argument('--results-dir',type=Path,help='Compatibility alias: resolve saved OOF/combined prediction CSV only')
    render.add_argument('--output-dir',type=Path,required=True);render.add_argument('--dpi',type=int,default=300)
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    try:
        if args.command=='train':
            from .training.run import train_from_config
            target=train_from_config(args.config)
        elif args.command=='extract':
            from .config import load_experiment
            from .io.manifest import features_from_manifest
            from .evaluation.export import export_csv
            cfg=load_experiment(args.config)
            target=export_csv(features_from_manifest(cfg['input_manifest'],feature_version=cfg['feature_version'],qc_config=cfg['qc_config']),args.output)
        elif args.command=='predict':
            from .prediction import predict_manifest
            from .evaluation.export import export_csv
            target=export_csv(predict_manifest(args.manifest,args.artifact,trusted=args.trusted_model,expected_feature_version=args.feature_version),args.output)
        elif args.command=='evaluate':
            from .visualization.render import read_saved_predictions
            from .evaluation.export import export_csv
            _,metrics=read_saved_predictions(args.input_csv)
            target=export_csv(metrics,args.output)
        else:
            from .visualization.render import render_predictions
            source=args.input_csv
            if args.results_dir is not None:
                paths=[args.results_dir/name for name in ['out_of_fold_predictions.csv','Figure5b_plot_data.csv','predictions_final_test.csv']]
                source=next((p for p in paths if p.is_file()),None)
                if source is None:raise ValueError("No saved prediction CSV in results directory; no fallback to demo files")
            target=render_predictions(source,args.output_dir,dpi=args.dpi)
    except (ValueError,FileNotFoundError,FileExistsError,ImportError,KeyError,TypeError) as exc:
        parser.exit(2,f'error: {exc}\n')
    print(target)
    return 0


if __name__=='__main__':
    main()
