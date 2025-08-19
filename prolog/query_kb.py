from pyswip import Prolog
import pandas as pd
from tabulate import tabulate

# Carica il dataset CSV
def carica_dataset(percorso):
    return pd.read_csv(percorso)

# Stampa i primi 3 risultati di una query
def stampa_primi_3_result(prolog, query, df):
    results = list(prolog.query(query))
    rows = []
    count = 0
    for result in results:
        if count < 3:
            game_id = result["Game"]
            game_record = df[df['Unnamed: 0'] == int(game_id[1:]) - 1]
            selected_columns = ['Unnamed: 0', 'kills', 'deaths', 'assists', 'killParticipation',
                                'goldPerMinute', 'totalMinionsKilled', 'totalDamageDealt',
                                'visionScorePerMinute', 'skillshotsDodged', 'skillshotsHit', 'win']
            for _, row in game_record[selected_columns].iterrows():
                rows.append(row.tolist())
            count += 1

    headers = ['Game ID', 'Kills', 'Deaths', 'Assists', 'Kill Participation',
               'GPM', 'CS', 'Damage', 'Vision', 'Skillshots Dodged', 'Skillshots Hit', 'Win']
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    data_path = "../datasets/euw_lol_games.csv"
    df = carica_dataset(data_path)

    prolog = Prolog()
    prolog.consult("../prolog/lol_kb.pl")

    queries = {
        "Astonishing games": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), astonishing(Game)",
        "Normal games": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), normal(Game)",
        "Bad games": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), bad(Game)",
        "Games with high KDA": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), high_kda(Game)",
        "Games with high kill participation": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), high_kill_participation(Game)",
        "Games with high GPM": "partita(Game, _, _, _, _, _, _, _, _, _, _, _), high_gpm(Game)"
    }

    for description, query in queries.items():
        print(f"\n{description}:")
        stampa_primi_3_result(prolog, query, df)

if __name__ == "__main__":
    main()
