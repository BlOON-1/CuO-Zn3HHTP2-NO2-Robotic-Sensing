"""Explicit reviewed cycle manifest; no labels are inferred from filenames."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .instrument import read_instrument
from .schema import CycleBoundary
from no2_ml.features import extract_features, feature_spec
from no2_ml.preprocessing.quality import QCConfig


def features_from_manifest(path, *, feature_version, qc_config=None, require_reference=True):
    path=Path(path)
    d=pd.read_csv(path,dtype={"sample_id":str,"source_id":str,"group_id":str,"source_file":str},keep_default_na=False)
    required={"sample_id","source_id","source_file","group_id","channel","baseline_start_s","exposure_start_s","exposure_end_s","recovery_end_s","approved","boundary_version"}
    if require_reference:required.add('reference_ppm')
    if not required.issubset(d) or d.empty or d.sample_id.duplicated().any():
        raise ValueError("Manifest missing fields, empty or duplicate sample IDs")
    for key in ['sample_id','source_id','group_id','source_file']:
        if d[key].astype(str).str.strip().eq('').any():raise ValueError(f"Empty manifest {key}")
    if not d.groupby('source_id').group_id.nunique().eq(1).all():
        raise ValueError("One source ID crosses groups")
    records={}; hashgroups={};sequencegroups={};sources={}; rows=[]
    qc=QCConfig(**(qc_config or {}))
    spec=feature_spec(feature_version)
    for row in d.to_dict('records'):
        source=(path.parent/row['source_file']).resolve()
        group=str(row['group_id'])
        if source not in records:
            raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            records[source]=(read_instrument(source,source_id=row['source_id'],invalid_resistance='flag'),digest)
        record,digest=records[source]
        if sources.setdefault(row['source_id'],str(source))!=str(source):
            raise ValueError("One source ID maps to different files")
        if hashgroups.setdefault(digest,group)!=group:
            raise ValueError("Identical source files assigned to different groups")
        ch=float(row['channel'])
        if not np.isfinite(ch) or ch<1 or not ch.is_integer():raise ValueError("Invalid channel ID")
        ch=int(ch)
        if ch not in record.resistance_ohm:raise ValueError("Manifest channel missing in signal")
        signature=hashlib.sha256(np.c_[record.time_s,record.resistance_ohm[ch]].astype('<f8').tobytes()).hexdigest()
        if sequencegroups.setdefault(signature,group)!=group:
            raise ValueError("Identical measured time series assigned to different groups")
        approved=str(row['approved']).lower()=='true'
        boundary=CycleBoundary(*(float(row[k]) for k in ['baseline_start_s','exposure_start_s','exposure_end_s','recovery_end_s']),
                               approved=approved,boundary_version=str(row['boundary_version']))
        anomaly=None
        if qc.version=='legacy08_mask_v1':
            if not row.get('anomaly_mask_file'):raise ValueError("Historical QC requires anomaly_mask_file")
            mask=pd.read_csv(path.parent/row['anomaly_mask_file'])
            if not {'time_s','candidate_anomaly'}.issubset(mask) or not np.array_equal(mask.time_s.to_numpy(float),record.time_s):
                raise ValueError("Archived QC mask timestamps must match raw signal")
            tokens=mask.candidate_anomaly.astype(str).str.lower()
            if not tokens.isin(['true','false']).all():raise ValueError("QC mask must contain true/false")
            anomaly=tokens.eq('true').to_numpy()
        result=extract_features(record,ch,boundary,spec.version,qc_config=qc,candidate_anomaly=anomaly)
        item=dict(sample_id=row['sample_id'],source_id=row['source_id'],group_id=group,channel=ch,
            source_sha256=digest,feature_version=result.feature_version,feature_fingerprint=result.feature_fingerprint,
            qc_version=qc.version,qc_fingerprint=qc.fingerprint,baseline_ohm=result.baseline_ohm,quality_issues=json.dumps(result.issues),
            prediction_mode='offline_reviewed_cycle',**result.values)
        if require_reference:
            reference=float(row['reference_ppm'])
            if not np.isfinite(reference) or reference<0:raise ValueError("Explicit nonnegative reference_ppm required")
            item['reference_ppm']=reference
        rows.append(item)
    # Detect in-place source modifications during extraction, without rewriting inputs.
    for source,(_,digest) in records.items():
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError("Source changed during extraction")
    return pd.DataFrame(rows)
