def show_stats_table_text(lol_data):
    na = lol_data.isna().sum()
    print(f"Valori NaN:\n{na}\n")

    dup = lol_data.duplicated().sum()
    print(f"Valori duplicati: {dup}\n")