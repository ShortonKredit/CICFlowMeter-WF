# cicflowmeter_wf/trace_reader.py

import pandas as pd
import io

def detect_separator_from_first_line(line: str) -> str:
    """
    Detects if the separator is a comma or semicolon based on a text line.
    """
    if ";" in line:
        return ";"
    return ","

def read_trace_csv(filepath_or_buffer) -> pd.DataFrame:
    """
    Reads a trace CSV file or buffer, handling both comma and semicolon delimiters.
    Preserves original packet order.
    """
    if isinstance(filepath_or_buffer, str):
        with open(filepath_or_buffer, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = detect_separator_from_first_line(first_line)
        df = pd.read_csv(filepath_or_buffer, sep=sep)
    else:
        # It's a file-like object or buffer (e.g. from tarfile stream)
        # Read the first line to detect separator
        first_line = filepath_or_buffer.readline()
        if isinstance(first_line, bytes):
            first_line_str = first_line.decode("utf-8", errors="ignore")
        else:
            first_line_str = first_line
        
        sep = detect_separator_from_first_line(first_line_str)
        
        # Reconstruct buffer by prepending the first line
        if isinstance(first_line, bytes):
            content = first_line + filepath_or_buffer.read()
            df = pd.read_csv(io.BytesIO(content), sep=sep)
        else:
            content = first_line + filepath_or_buffer.read()
            df = pd.read_csv(io.StringIO(content), sep=sep)
            
    return df
