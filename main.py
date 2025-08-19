import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

from displayers.plotters import *
from displayers.textual_tables import show_stats_table_text
from pre_processing import utils
from pre_processing.outliers_management import remove_outliers_iqr
from supervised_training.supervised_learning import train_models_with_cv


def preprocess_data(lol_data, verbose=False):
    # Seleziono le colonne numeriche da preprocessare
    selected_columns = lol_data[['kills', 'deaths', 'assists', 'killParticipation',
                                 'goldPerMinute', 'totalMinionsKilled', 'totalDamageDealt',
                                 'visionScorePerMinute', 'skillshotsDodged', 'skillshotsHit']]

    # Mostro le statistiche iniziali se verbose=True
    if verbose:
        plot_column_statistics(selected_columns, selected_columns.columns)
        plot_donut_win(lol_data)

    # Rimuovo gli outlier con IQR per ogni colonna numerica
    for col in list(selected_columns.columns):
        selected_columns = remove_outliers_iqr(selected_columns, col)

    # Aggiungo la colonna target 'win' senza modifiche
    selected_columns['win'] = lol_data['win']

    if verbose:
        # Mostro grafico a ciambella della distribuzione delle vittorie/sconfitte
        plot_donut_win(selected_columns)
        # Mostro le statistiche dopo rimozione degli outlier
        plot_column_statistics(selected_columns, selected_columns.columns)
        # Mostro tabella testuale con valori NaN e duplicati
        show_stats_table_text(selected_columns)

    # Conversione della colonna target in numerico (True → 1, False → 0)
    selected_columns=utils.convert_target_to_numeric(selected_columns)

    # Salvataggio del dataset preprocessato in CSV
    selected_columns.to_csv('./datasets/euw_lol_games_outliners_removed.csv', index=False)

    return selected_columns


def train_and_predict(processed_dataset, apply_smote=False):
    # x = tutte le feature tranne il target
    x = processed_dataset[['kills', 'deaths', 'assists',
                           'killParticipation', 'goldPerMinute',
                           'totalMinionsKilled', 'totalDamageDealt',
                           'visionScorePerMinute', 'skillshotsDodged',
                           'skillshotsHit']]

    # y = variabile target (win)
    y = processed_dataset['win']

    # Applica SMOTE solo se richiesto
    if apply_smote:
        smote = SMOTE()
        x, y = smote.fit_resample(x, y)

    print(f"SMOTE attivo: {apply_smote} → Distribuzione classi:\n{y.value_counts()}")

    # Addestra i modelli con cross-validation
    best_models = train_models_with_cv(x, y)

    # Crea una copia del dataset e aggiungi le predizioni
    dataset_with_predictions = processed_dataset.copy()

    for model in best_models:
        print(f"\nMiglior modello per {model}: {best_models[model]['best_estimator']}")
        print(f"Parametri ottimali: {best_models[model]['best_params']}")

        dataset_with_predictions[model] = best_models[model]["best_estimator"].predict(
            processed_dataset[['kills', 'deaths', 'assists',
                               'killParticipation', 'goldPerMinute',
                               'totalMinionsKilled', 'totalDamageDealt',
                               'visionScorePerMinute', 'skillshotsDodged',
                               'skillshotsHit']])

    # Salva il dataset con le predizioni
    dataset_with_predictions.to_csv('./datasets/euw_lol_games_predictions.csv', index=False)

    return dataset_with_predictions

def main():
    # Carica il dataset grezzo
    raw_dataset = pd.read_csv('./datasets/euw_lol_games.csv')

    # Preprocessamento
    processed_dataset = preprocess_data(raw_dataset)

    # Normalizzazione delle colonne numeriche
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_columns = processed_dataset.select_dtypes(include=['float64', 'int64']).columns.drop('win')
    processed_dataset[numeric_columns] = scaler.fit_transform(processed_dataset[numeric_columns])

    # Versione senza SMOTE
    dataset_no_smote = train_and_predict(processed_dataset, apply_smote=False)

    # Versione con SMOTE
    dataset_smote = train_and_predict(processed_dataset, apply_smote=True)

    # Salvataggio risultati
    dataset_no_smote.to_csv('./datasets/euw_lol_games_predictions_NO_SMOTE.csv', index=False)
    dataset_smote.to_csv('./datasets/euw_lol_games_predictions_SMOTE.csv', index=False)

if __name__ == "__main__":
    main()


