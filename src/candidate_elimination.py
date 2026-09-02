"""
Candidate-Elimination & Version Space Learning Module from First Principles (CO1/CO2)
Implements Candidate-Elimination algorithm on discretized agro-climatic yield risk instances,
maintaining Specific (S) and General (G) boundary hypotheses and analyzing Inductive Bias.
"""

import numpy as np
import pandas as pd

class CandidateElimination:
    def __init__(self, attribute_names, domain_values):
        """
        Parameters:
        - attribute_names: List of column names representing features.
        - domain_values: Dict mapping each attribute to its list of possible values.
        """
        self.attribute_names = attribute_names
        self.domain_values = domain_values
        self.num_attrs = len(attribute_names)
        
        # Initialize Specific Boundary S0: ['∅', '∅', ...]
        self.S = [['∅'] * self.num_attrs]
        # Initialize General Boundary G0: [['?', '?', ...]]
        self.G = [['?'] * self.num_attrs]

    def _is_consistent(self, hypothesis, instance):
        """Checks if a hypothesis matches an instance."""
        for h_val, inst_val in zip(hypothesis, instance):
            if h_val != '?' and h_val != inst_val:
                return False
        return True

    def _is_more_general(self, h1, h2):
        """Returns True if hypothesis h1 is more general than or equal to h2."""
        for val1, val2 in zip(h1, h2):
            if val1 != '?' and val1 != val2:
                return False
        return True

    def fit(self, X_discrete, y_binary):
        """
        Executes Candidate-Elimination over binary discrete dataset.
        y_binary: True/1 for Positive Instances (e.g. High Yield Risk), False/0 for Negative Instances.
        """
        for idx, (instance, label) in enumerate(zip(X_discrete, y_binary)):
            instance = list(instance)
            if label == 1:  # Positive Instance
                # 1. Remove from G any hypothesis inconsistent with instance
                self.G = [g for g in self.G if self._is_consistent(g, instance)]
                
                # 2. Update S boundary
                new_S = []
                for s in self.S:
                    if self._is_consistent(s, instance):
                        new_S.append(s)
                    else:
                        # Minimal generalization of s consistent with instance
                        generalized_s = list(s)
                        for i in range(self.num_attrs):
                            if generalized_s[i] == '∅':
                                generalized_s[i] = instance[i]
                            elif generalized_s[i] != instance[i]:
                                generalized_s[i] = '?'
                                
                        # Keep only if consistent with some g in G
                        if any(self._is_more_general(g, generalized_s) for g in self.G):
                            new_S.append(generalized_s)
                            
                # Remove hypotheses in S that are more general than another hypothesis in S
                self.S = []
                for s in new_S:
                    if not any(self._is_more_general(s, other) and s != other for other in new_S):
                        if s not in self.S:
                            self.S.append(s)

            else:  # Negative Instance (label == 0)
                # 1. Remove from S any hypothesis consistent with instance
                self.S = [s for s in self.S if not self._is_consistent(s, instance)]
                
                # 2. Update G boundary
                new_G = []
                for g in self.G:
                    if not self._is_consistent(g, instance):
                        new_G.append(g)
                    else:
                        # Minimal specializations of g inconsistent with instance
                        for i in range(self.num_attrs):
                            if g[i] == '?':
                                for val in self.domain_values[self.attribute_names[i]]:
                                    if val != instance[i]:
                                        specialized_g = list(g)
                                        specialized_g[i] = val
                                        # Keep if more general than some s in S
                                        if any(self._is_more_general(specialized_g, s) for s in self.S):
                                            new_G.append(specialized_g)
                                            
                # Remove hypotheses in G that are more specific than another hypothesis in G
                self.G = []
                for g in new_G:
                    if not any(self._is_more_general(other, g) and g != other for other in new_G):
                        if g not in self.G:
                            self.G.append(g)
                            
        return self.S, self.G

def discretize_agro_climatic_data(df):
    """Discretizes continuous features into categorical risk bands for Version Space analysis."""
    df_disc = pd.DataFrame()
    
    # 1. Rainfall Category
    df_disc['Rainfall_Band'] = pd.cut(df['Rainfall_mm'], bins=[-np.inf, 500, 1000, np.inf], labels=['Low_Rain', 'Normal_Rain', 'High_Rain'])
    
    # 2. Heat Stress
    df_disc['Heat_Stress'] = pd.cut(df['Temp_Max_C'], bins=[-np.inf, 33, 37, np.inf], labels=['Moderate_Temp', 'High_Temp', 'Extreme_Temp'])
    
    # 3. GDD Category
    df_disc['GDD_Band'] = pd.cut(df['GDD_Degree_Days'], bins=[-np.inf, 1800, 2400, np.inf], labels=['Low_GDD', 'Optimal_GDD', 'High_GDD'])
    
    # 4. Soil Quality
    df_disc['Soil_Quality'] = pd.cut(df['Organic_Carbon_Pct'], bins=[-np.inf, 0.45, 0.75, np.inf], labels=['Poor_Soil', 'Average_Soil', 'Fertile_Soil'])
    
    # 5. Season
    df_disc['Season'] = df['Season'].astype(str)
    
    # Target: High Yield Risk (Yield < 2600 kg/ha)
    y_risk = (df['Crop_Yield_kg_ha'] < 2600).astype(int)
    
    domain_values = {
        'Rainfall_Band': ['Low_Rain', 'Normal_Rain', 'High_Rain'],
        'Heat_Stress': ['Moderate_Temp', 'High_Temp', 'Extreme_Temp'],
        'GDD_Band': ['Low_GDD', 'Optimal_GDD', 'High_GDD'],
        'Soil_Quality': ['Poor_Soil', 'Average_Soil', 'Fertile_Soil'],
        'Season': ['Kharif', 'Rabi', 'Zaid']
    }
    
    return df_disc, y_risk, domain_values

if __name__ == '__main__':
    # Test on a small representative set of discretized data
    attr_names = ['Rainfall', 'Heat', 'Soil']
    domains = {'Rainfall': ['Low', 'Normal', 'High'], 'Heat': ['Cool', 'Hot'], 'Soil': ['Poor', 'Fertile']}
    
    ce = CandidateElimination(attr_names, domains)
    X_test = [
        ['Low', 'Hot', 'Poor'],
        ['Normal', 'Cool', 'Fertile'],
        ['Low', 'Hot', 'Fertile']
    ]
    y_test = [1, 0, 1]  # 1 = High Yield Risk
    
    S_final, G_final = ce.fit(X_test, y_test)
    print("Final Specific Boundary S:", S_final)
    print("Final General Boundary G:", G_final)
