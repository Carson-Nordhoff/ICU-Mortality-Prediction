import pandas as pd
import io

def format_dataframe_info(df: pd.DataFrame) -> str:

    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    return df_info_str