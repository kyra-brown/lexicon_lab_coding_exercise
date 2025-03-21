import pandas as pd
import matplotlib.pyplot as plot

def group_animal_production():
    """Caculates and displays the number of animals named by the NH and CI groups"""
    cochlear_data = pd.read_csv('data-cochlear.txt', header = None, names = ['Subject', 'animals_produced'], delimiter = '\t')
    participant_groupings = pd.read_csv('participant_groupings.csv')
    animals_named = cochlear_data.groupby('Subject').size().reset_index(name = 'animals_produced')
    combined_data = pd.merge(animals_named, participant_groupings, on = 'Subject')
    grouped_data = combined_data.groupby('Group')['animals_produced'].sum().reset_index()

    print(grouped_data)

    mean = combined_data.groupby('Group')['animals_produced'].mean().reset_index()
    standard_deviation = combined_data.groupby('Group')['animals_produced'].std().reset_index()

    plot.figure(figsize = (8,6))
    plot.bar(mean['Group'], mean['animals_produced'], yerr = standard_deviation['animals_produced'],
    capsize = 6, color = ['blue', 'purple'], alpha = 0.7)

    plot.title('Mean Number of Animals Produced by Group')
    plot.xlabel('Group')
    plot.ylabel('Mean Number of Animals Produced')
    plot.savefig('results/group_animal_production.png')
    plot.show()
    
    return grouped_data
group_animal_production()
    
