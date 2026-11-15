from pathlib import Path


def export_csv(table, path):
    """Exclusive creation: existing results are never overwritten."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        table.to_csv(stream,index=False,float_format='%.17g')
    return path
