"""
    This module filters the folder and gets the top 30 spices of butterfly species
    with the most images.
"""

import os
import shutil
from collections import Counter

class TopButterflySpecies:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.top_species = []

    def get_top_species(self, top_n=30):
        species_counts = {}
        for species in os.listdir(self.dataset_path):
            species_dir = os.path.join(self.dataset_path, species)
            if os.path.isdir(species_dir):
                num_files = len([
                    f for f in os.listdir(species_dir)
                    if os.path.isfile(os.path.join(species_dir, f))
                ])
                species_counts[species] = num_files
        # Get top N species by sample count
        top_species = Counter(species_counts).most_common(top_n)
        self.top_species = [species for species, count in top_species]
        return self.top_species

    def delete_non_top_species(self, dataset_path="", top_n=30):
     
        for species in os.listdir(dataset_path):
            species_dir = os.path.join(dataset_path, species)
            if os.path.isdir(species_dir) and species not in self.top_species:
                shutil.rmtree(species_dir)
                print(f"Deleted: {species_dir}")
# Example usage:
top_species_finder = TopButterflySpecies('dataset/train')
print(top_species_finder.get_top_species())

# I am going to delete the non-top species from train, test and valid datasets
# top_species_finder.delete_non_top_species('dataset/train')
# top_species_finder.delete_non_top_species('dataset/test')
# top_species_finder.delete_non_top_species('dataset/valid')

"""
Add to the docuement: 
    Agregar cuales fueron las especies escogidas y fueron escogidas 
    del dataset de train como las 30 especies con mas imagenes.
"""