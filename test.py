import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from descriptives import group_animal_production
from lexical import Similarity, Clusters, consecutive_similarity_graph

def saved_function_outputs(file, data):
    with open(f'results/{file}.txt', 'w') as f:
        f.write(str(data))

def test_group_animal_production():
    print('Test of group animal production function:')
    grouped_data = group_animal_production()
    plot.savefig('results/group_animal_production.png')
    print("Group animal production saved as group_animal_production.png")
    saved_function_outputs('descriptives_funct_test', grouped_data)



def test_similarity():
    print('Test of cosine similarity function:')
    similarity = Similarity('word2vec.txt')
    word1 = 'jellyfish'
    word2 = 'penguin'
    similarity = similarity.cosine_similarity(word1, word2)
    print(f"The cosine similarity between {word1} and {word2} is {similarity}")
    saved_function_outputs('similarity_funct_test', f"The cosine similarity between {word1} and {word2} is {similarity}")

def test_pairwise_similarity():
    similarity = Similarity('word2vec.txt')
    dataframe = similarity.pairwise_similarity('data-cochlear.txt', similarity)
    dataframe.to_csv('results/pairwise_similarity.csv', index = False)
    print("Pairwise similarity function saved as pairwise_similarity.csv")


def test_consecutive_similarity_graph():
    similarity = Similarity('word2vec.txt')
    dataframe = similarity.pairwise_similarity('data-cochlear.txt', similarity)
    consecutive_similarity_graph(dataframe)
    plot.savefig('results/consecutive_similarity_graph.png')
    print("Consective similarity graph saved as consecutive_similarity_graph.png")

def test_clusters():
    clusters = Clusters('word2vec.txt')
    clusters.visualize_clusters('data-cochlear.txt')
    plot.savefig('results/visualize_clusters.png')
    print("Clusters graph saved as visualize_clusters.png")
    cochlear_data = pd.read_csv('data-cochlear.txt', header = None, names = ['ID', 'item'], delimiter = '\t')
    participant_data = cochlear_data.groupby('ID')
    cluster_count = []
    switch_count = []
    for group_id, group_data in participant_data:
        items = group_data['item'].tolist()
        cluster_number, switch_number = clusters.compute_clusters(items)
        cluster_count.append(cluster_number)
        switch_count.append(switch_number)
    clusters_output = pd.DataFrame({
        'Participant ID': list(participant_data.groups.keys()),
        'Clusters': cluster_count, 
        'Switches': switch_count})
    clusters_output.to_csv('results/cluster_and_switch_count.csv', index = False)
    print("Cluster and switch count saved as cluster_and_switch_count.csv")

test_group_animal_production()
test_similarity()
test_pairwise_similarity()
test_consecutive_similarity_graph()
test_clusters()




    