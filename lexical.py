import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from switch import switch_simdrop



class Similarity:


    def __init__(self, file):
        self.model = self.load_word2vec(file)


    def load_word2vec(self, file):
        """Loads word2vec file into code"""
        word_vectors = {} #creating dictionary

        with open(file, 'r', encoding = 'utf8') as file:
            for line in file:
                data = line.strip().split()
                word = data[0].lower()
                vector = np.array([float(number) for number in data[1:]]) 
                word_vectors[word] = vector
        return word_vectors
    

    def cosine_similarity(self, word1, word2):
        """computes the cosine similarity between words based on their numeric representation in word2vec"""
        if word1.lower() == word2.lower():
            return 1.0
        
        if word1 not in self.model or word2 not in self.model:
            return 0
        word1vector = self.model[word1.lower()]
        word2vector = self.model[word2.lower()]

        dot_product = np.dot(word1vector, word2vector)
        word1vector_length = np.linalg.norm(word1vector)
        word2vector_length = np.linalg.norm(word2vector)

        similarity_value = dot_product / (word1vector_length * word2vector_length)

        return similarity_value 

    @staticmethod
    def pairwise_similarity(file, model):
        """computes pairwise cosine similarity between each consecutive word produced by 
        participants based on file provided"""
        cochlear_data = pd.read_csv(file, header = None, names = ['ID', 'item'], delimiter = '\t')
        participant_data = cochlear_data.groupby('ID')
        output = []
        for group_id, group in participant_data: 
            item1 = group.iloc[0]['item']
            output.append({'ID': group_id, 'item': item1, 'similarity': 2.0})
            
            for item in range(1, len(group)):
                word1 = group.iloc[item - 1]['item']
                word2 = group.iloc[item]['item']
                similarity_value = model.cosine_similarity(word1, word2)
                output.append({'ID': group_id, 'item': word2, 'similarity': similarity_value})

        return pd.DataFrame(output)
        

def consecutive_similarity_graph(dataframe):
    """Displays a graph showing the mean semantic similarity of consecutive responses per group"""
    participant_groupings = pd.read_csv('participant_groupings.csv')
    mean_participant_response_similarity = dataframe.groupby('ID')['similarity'].mean().reset_index()
    similarity_group_data = mean_participant_response_similarity.merge(participant_groupings, left_on = 'ID', right_on = 'Subject')

    group_mean = similarity_group_data.groupby('Group')['similarity'].mean()

    plot.figure(figsize = (8,6))
    plot.bar(group_mean.index, group_mean.values, color = ['blue', 'purple'], alpha = 0.7)

    plot.title('Mean Semantic Similarity of Consecutive Responses Per Group')
    plot.xlabel('Group')
    plot.ylabel('Mean Semantic Similarity')
    plot.savefig('results/consecutive_similarity_graph.png')
    plot.show()


class Clusters:
    def __init__(self, word2vec_file):
        self.similarity = Similarity(word2vec_file)

    def compute_clusters(self, participant_data):
        """Computes clusters and switches based on similarity values in word2vec and simdrop method"""
        similarity_data = self.similarity.pairwise_similarity('data-cochlear.txt', self.similarity)
        similarities = similarity_data['similarity'].tolist()
        switches = switch_simdrop(participant_data, similarities)

        clusters = []
        current_cluster = []

        for item, switch in enumerate(switches): 
            if item >= len(participant_data):
                break 
            if switch == 1:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [participant_data[item]]
            else: 
                current_cluster.append(participant_data[item])
        
        if current_cluster:
            clusters.append(current_cluster)

        cluster_number = len(clusters) 
        switch_number =  switches.count(1)
        return cluster_number, switch_number

    def visualize_clusters(self, data_file):
        """Displays the mean number of clusters and switches per group"""
        cochlear_data = pd.read_csv(data_file, header = None, names = ['ID', 'item'], delimiter = '\t')        
        participant_groupings = pd.read_csv('participant_groupings.csv')
        merged_data = pd.merge(cochlear_data, participant_groupings, left_on = 'ID', right_on = 'Subject', how = 'left')

        cluster_count = []
        switch_count = []

        for group_id, group_data in merged_data.groupby('Subject'):
            items = group_data['item'].tolist()
            cluster_number, switch_number = self.compute_clusters(items)

            cluster_count.append(cluster_number)
            switch_count.append(switch_number)

        participant_groupings['cluster_count'] = cluster_count
        participant_groupings['switch_count'] = switch_count

        merged_data = pd.merge(cochlear_data, participant_groupings, left_on = 'ID', right_on = 'Subject', how = 'left')
        group_means = participant_groupings.groupby('Group').agg({'cluster_count': 'mean', 'switch_count': 'mean'}).reset_index()
        
        fig, ax = plot.subplots(figsize = (8, 6))
        index = np.arange(len(group_means))
        
        ax.bar(index - 0.2, group_means['cluster_count'], 0.4, label = 'Clusters', color = 'blue', alpha = 0.7) 
        ax.bar(index + 0.2 , group_means['switch_count'], 0.4,  label = 'Switches', color = 'purple', alpha = 0.7)
        group_names = group_means['Group'].tolist()
        ax.set_xticks(index)
        ax.set_xticklabels(group_names)
    
        ax.set_title('Mean Number of Clusters and Switches Per Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Number of Clusters and Switches')
        ax.legend()
        plot.savefig('results/visualize_clusters.png')
        plot.show()

           
similarity = Similarity('word2vec.txt')
dataframe = Similarity.pairwise_similarity('data-cochlear.txt', similarity)

consecutive_similarity_graph(dataframe)

clusters = Clusters('word2vec.txt')
clusters.visualize_clusters('data-cochlear.txt')








        

