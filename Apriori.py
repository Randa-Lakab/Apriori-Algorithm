# apriori.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example dataset
data = pd.DataFrame({
    'Milk': [1, 0, 1, 1],
    'Bread': [1, 1, 1, 0],
    'Butter': [0, 1, 1, 1]
})

frequent_itemsets = apriori(data, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(rules)
