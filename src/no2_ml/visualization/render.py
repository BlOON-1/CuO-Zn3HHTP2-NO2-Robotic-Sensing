"""Saved predictions -> parity, residuals, and model comparison; no model fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from no2_ml.evaluation import evaluate_predictions, compare_models
from no2_ml.evaluation.export import export_csv


def technical_summary(d):
    """Within-file pooled technical SD; never an independent-test interval."""
    rows=[]
    for (name,c),part in d.groupby(['model_name','y_true'],sort=False):
        files=part.groupby('source_id').y_pred.agg(['count','mean','var'])
        df=int((files['count']-1).sum())
        within=float(np.sqrt(((files['count']-1)*files['var'].fillna(0)).sum()/df)) if df>0 else np.nan
        rows.append(dict(model_name=name,reference_ppm=c,n_cycles=len(part),n_source_files=len(files),
            mean_predicted_ppm=part.y_pred.mean(),sd_within_file_ppm=within,
            sd_between_file_means_ppm=files['mean'].std(ddof=1),within_file_degrees_of_freedom=df))
    return pd.DataFrame(rows)


def read_saved_predictions(path, *, legacy_feature_version='historical_unverified@0'):
    d=pd.read_csv(path,comment='#',dtype={'sample_id':str,'group_id':str,'fold_id':str},float_precision='round_trip')
    saved_metrics=None
    if 'record_type' in d:
        if not d.record_type.isin(['prediction','summary','metrics']).all():raise ValueError("Invalid legacy record_type")
        saved_metrics=d[d.record_type.eq('metrics')].copy()
        summary=d[d.record_type.eq('summary')].copy()
        d=d[d.record_type.eq('prediction')].copy()
    else:summary=None
    if 'dataset_split' not in d or not d.dataset_split.isin(['outer_test','group_held_out','test']).all():
        raise ValueError("Only saved held-out predictions may be rendered; training/demo records rejected")
    if d.dataset_split.nunique()!=1:
        raise ValueError("Mixed prediction split semantics")
    if 'data_type' in d and d.data_type.astype(str).str.contains('synthetic|illustrat',case=False).any():
        raise ValueError("Illustrative predictions are not accepted by the formal renderer")
    if 'y_true' not in d:
        required={'reference_concentration_ppm','predicted_concentration_ppm','sample_id','group_id','file_name','validation_type'}
        if not required.issubset(d):raise ValueError("Legacy saved prediction fields missing")
        if d.validation_type.nunique()!=1 or d.channel.nunique()!=1:raise ValueError("Mixed legacy analysis strata")
        d=d.rename(columns={'reference_concentration_ppm':'y_true','predicted_concentration_ppm':'y_pred','file_name':'source_id'})
        d['model_name']=d.model_name.fillna('legacy_model') if 'model_name' in d else 'legacy_model'
        d['feature_version']=legacy_feature_version
        d['fold_id']=d.get('outer_fold',d.group_id).astype(str)
        d['evaluation_scope']='single_test_set' if d.dataset_split.eq('test').all() else 'pooled_oof'
    if d.channel.nunique()!=1:raise ValueError("Renderer requires one channel stratum")
    compare_models(d)
    if 'residual_ppm' in d and not np.allclose(d.residual_ppm,d.y_pred-d.y_true):raise ValueError("Saved residual mismatch")
    metrics=evaluate_predictions(d)
    if saved_metrics is not None and not saved_metrics.empty:
        if len(saved_metrics)!=1 or d.model_name.nunique()!=1:raise ValueError("Legacy metrics must describe one evaluation")
        overall=metrics[metrics.evaluation_scope.isin(['pooled_oof','single_test_set'])].set_index('name')
        for old,key in [('MAE_ppm','mae'),('RMSE_ppm','rmse'),('Bias_ppm','mean_bias_error'),('R2','r2')]:
            if old in saved_metrics and not np.isclose(saved_metrics.iloc[0][old],overall.loc[key,'value'],rtol=1e-6,atol=1e-8,equal_nan=True):
                raise ValueError(f"Saved metric mismatch: {old}")
    if summary is not None and not summary.empty:
        expected=technical_summary(d)
        if set(summary.reference_concentration_ppm)!=set(expected.reference_ppm):raise ValueError("Saved summary concentration mismatch")
        for _,row in expected.iterrows():
            saved=summary[summary.reference_concentration_ppm.eq(row.reference_ppm)]
            if len(saved)!=1:raise ValueError("Missing/duplicate summary")
            for key in ['mean_predicted_ppm','sd_within_file_ppm','sd_between_file_means_ppm','n_cycles','n_source_files','within_file_degrees_of_freedom']:
                if key in summary and not np.isclose(saved.iloc[0][key],row[key],equal_nan=True):raise ValueError(f"Saved technical summary mismatch: {key}")
    return d,metrics


def render_predictions(input_csv,output_dir,*,dpi=300):
    source=Path(input_csv)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    d,metrics=read_saved_predictions(source)
    if not isinstance(dpi,int) or dpi<72:raise ValueError("dpi must be an integer >=72")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font='Arial' if 'Arial' in fonts else 'DejaVu Sans'
    summary=technical_summary(d)
    overall=metrics[metrics.evaluation_scope.isin(['pooled_oof','single_test_set'])]
    colors=['#596A7A','#D49143','#1764A0','#3B8C71','#8064A2','#BD6656']
    with plt.rc_context({'font.family':font,'pdf.fonttype':42,'svg.fonttype':'none','font.size':9}):
        for i,(name,part) in enumerate(d.groupby('model_name',sort=False)):
            fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
            ax=axes[0];color=colors[i%len(colors)]
            ax.scatter(part.y_true,part.y_pred,s=14,facecolors='none',edgecolors=color,label='Held-out cycles')
            group=summary[summary.model_name.eq(name)]
            good=group.sd_within_file_ppm.notna()
            ax.errorbar(group.loc[good,'reference_ppm'],group.loc[good,'mean_predicted_ppm'],
                        yerr=group.loc[good,'sd_within_file_ppm'],fmt='o',color=color,ms=3,capsize=2,label='Mean ± technical SD')
            lo=min(part.y_true.min(),part.y_pred.min());hi=max(part.y_true.max(),part.y_pred.max());pad=max((hi-lo)*.06,.1)
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',lw=.8,color='.5')
            ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Reference NO$_2$ (ppm)',ylabel='Predicted NO$_2$ (ppm)')
            ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=6)
            m=overall[overall.model_name.eq(name)].set_index('name')
            r2=m.loc['r2','value'];r2text='undefined' if pd.isna(r2) else f'{r2:.3f}'
            ax.text(.97,.04,f"R² = {r2text}\nMAE = {m.loc['mae','value']:.2f} ppm\nRMSE = {m.loc['rmse','value']:.2f} ppm",transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
            axes[1].scatter(part.y_true,part.y_pred-part.y_true,s=14,color=color)
            axes[1].axhline(0,color='.5',ls='--',lw=.8)
            axes[1].set(xlabel='Reference NO$_2$ (ppm)',ylabel='Predicted − reference (ppm)')
            fig.suptitle(f'{name}: {part.evaluation_scope.iloc[0]}')
            # Names derive only from registered/legacy model labels, never paths.
            stem=''.join(c for c in str(name) if c.isalnum() or c in '_-') or 'model'
            for extension in ['png','pdf','svg']:fig.savefig(output/f'{stem}_parity_residuals.{extension}',dpi=dpi)
            plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
        for ax,metric,label in zip(axes,['mae','rmse'],['MAE (ppm)','RMSE (ppm)']):
            values=overall[overall.name.eq(metric)]
            ax.barh(values.model_name,values.value,color=colors[:len(values)] if len(values)<=len(colors) else colors[0])
            ax.invert_yaxis();ax.set_xlabel(label);ax.set_xlim(left=0)
        for extension in ['png','pdf','svg']:fig.savefig(output/f'model_comparison.{extension}',dpi=dpi)
        plt.close(fig)
    export_csv(metrics,output/'rendered_metrics.csv')
    export_csv(summary,output/'technical_summary.csv')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:raise ValueError("Saved predictions changed during rendering")
    (output/'render_manifest.json').write_text(json.dumps({'source_sha256':before,'source_unchanged':True,'font':font,
        'dpi':dpi,'metrics_source':'unmodified saved predictions','no_training':True},indent=2),encoding='utf8')
    return output
