import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, K2, AIC, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
import logging

# Disabilita log pgmpy
logging.getLogger('pgmpy').setLevel(logging.WARNING)

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data[['kills', 'deaths', 'assists', 'killParticipation',
                 'goldPerMinute', 'totalMinionsKilled', 'totalDamageDealt',
                 'visionScorePerMinute', 'skillshotsDodged', 'skillshotsHit', 'win']]

def discretize_data(data):
    features_to_discretize = [
        'kills', 'deaths', 'assists', 'killParticipation',
        'goldPerMinute', 'totalMinionsKilled', 'totalDamageDealt',
        'visionScorePerMinute', 'skillshotsDodged', 'skillshotsHit'
    ]
    for col in features_to_discretize:
        data[col] = pd.qcut(data[col], q=3, labels=False)
    data['win'] = data['win'].astype(int)
    return data

def learn_structure(data, scoring_method, max_iter=5000):
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=scoring_method, max_iter=max_iter)
    return model

def fit_parameters(model, data):
    bayesian_model = DiscreteBayesianNetwork(model.edges())
    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)
    return bayesian_model


def plot_network(model, title="Bayesian Network"):
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(model.edges())

    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(nx_graph, seed=42, k=1.5)

    nx.draw(
        nx_graph, pos,
        with_labels=True,
        node_size=2500,
        node_color='skyblue',
        font_size=10,
        font_weight='bold',
        arrows=True
    )

    plt.title(title)
    plt.show()


def generate_random_examples(model, num_samples=5):
    samples = model.simulate(num_samples)
    # Seleziona solo le colonne principali più win
    cols_to_show = ['kills', 'deaths', 'assists', 'killParticipation', 'goldPerMinute', 'win']
    print(f"\nEsempi casuali generati dal modello :")
    print(samples[cols_to_show])


def main():
    dataset_path = "../datasets/euw_lol_games_outliners_removed.csv"
    data = load_data(dataset_path)
    data = discretize_data(data)

    # --- K2 ---
    print("\n--- Modello K2 ---")
    k2_model = learn_structure(data, K2(data))
    k2_score = K2(data).score(k2_model)
    print(f"K2 Score: {k2_score:.4f}")
    k2_model_fitted = fit_parameters(k2_model, data)
    plot_network(k2_model_fitted, title="K2 Learned Bayesian Network")

    # --- AIC ---
    print("\n--- Modello AIC ---")
    aic_model = learn_structure(data, AIC(data))
    aic_score = AIC(data).score(aic_model)
    print(f"AIC Score: {aic_score:.4f}")
    aic_model_fitted = fit_parameters(aic_model, data)
    plot_network(aic_model_fitted, title="AIC Learned Bayesian Network")

    # Esempi casuali solo per il modello migliore (AIC)
    generate_random_examples(aic_model_fitted, num_samples=5)

if __name__ == "__main__":
    main()
