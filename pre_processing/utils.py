def convert_target_to_numeric(data, target_column='win'):
    data[target_column] = data[target_column].astype(int)
    return data

