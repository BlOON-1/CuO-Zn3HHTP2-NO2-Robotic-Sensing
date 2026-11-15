"""Strict, read-only parser for the existing multi-channel instrument TXT dialect."""
from pathlib import Path
import re
import csv
import numpy as np
from .schema import SignalRecord

TIME_UNITS = {"毫秒": .001, "ms": .001, "秒": 1., "s": 1.}
RESISTANCE_UNITS = {"欧姆": 1., "欧": 1., "Ω": 1., "ohm": 1.,
                    "kΩ": 1e3, "MΩ": 1e6, "kohm": 1e3, "Mohm": 1e6}


def _unit(header):
    match = re.search(r"[（(]([^()（）]+)[）)]", header)
    return match.group(1).strip() if match else ""


def read_instrument(path, *, source_id=None, invalid_resistance="raise", encoding=None):
    """No concentration is inferred. 'flag' retains invalid resistance as NaN/zero."""
    path = Path(path)
    if invalid_resistance not in {"raise", "flag"}:
        raise ValueError("invalid_resistance must be raise or flag")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("Empty instrument file")
    text = None
    for candidate in ([encoding] if encoding else ["utf-8-sig", "gb18030", "utf-16"]):
        try:
            decoded = raw.decode(candidate, errors="strict")
        except UnicodeError:
            continue
        if "时间" in decoded and "通道" in decoded:
            text = decoded
            break
    if text is None:
        raise ValueError("Cannot decode a recognized instrument header; specify encoding")
    lines = text.splitlines()
    h = next((i for i, line in enumerate(lines) if "时间" in line and "电阻" in line and "\t" in line), None)
    if h is None:
        raise ValueError("Missing tab-separated instrument header")
    headers = [s.strip() for s in lines[h].split("\t")]
    times = [i for i, s in enumerate(headers) if s.startswith("时间")]
    if len(times) != 1 or _unit(headers[times[0]]) not in TIME_UNITS:
        raise ValueError("Missing/ambiguous time column or unsupported time unit")
    channels = {}
    for i, header in enumerate(headers):
        match = re.match(r"通道(\d+)电阻", header)
        if match:
            ch = int(match.group(1))
            unit = _unit(header)
            if unit not in RESISTANCE_UNITS or ch in channels:
                raise ValueError("Unsupported resistance unit or duplicate channel")
            channels[ch] = (i, RESISTANCE_UNITS[unit])
    if not channels:
        raise ValueError("No resistance columns")
    rows = [row for row in csv.reader(lines[h+1:], delimiter="\t") if any(s.strip() for s in row)]
    if not rows or any(len(row) != len(headers) for row in rows):
        raise ValueError("Empty data or inconsistent column count")
    def numeric(index):
        values = []
        for row in rows:
            try:
                values.append(float(row[index]))
            except ValueError:
                values.append(np.nan)
        return np.asarray(values)
    t = numeric(times[0])*TIME_UNITS[_unit(headers[times[0]])]
    signals, issues = {}, []
    for ch, (i, factor) in channels.items():
        r = numeric(i)*factor
        count = int((~np.isfinite(r) | (r <= 0)).sum())
        if count:
            if invalid_resistance == "raise":
                raise ValueError(f"Channel {ch}: {count} invalid resistance fields")
            issues.append(f"channel={ch}: invalid_resistance_count={count}")
        signals[ch] = r
    return SignalRecord(t, signals, str(source_id or path.stem), str(path.resolve()), tuple(issues))
