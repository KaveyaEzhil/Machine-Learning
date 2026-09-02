from src.candidate_elimination import CandidateElimination

def test_candidate_elimination_boundaries():
    attr_names = ['Rainfall', 'Heat', 'Soil']
    domains = {
        'Rainfall': ['Low', 'Normal', 'High'],
        'Heat': ['Cool', 'Hot'],
        'Soil': ['Poor', 'Fertile']
    }
    
    ce = CandidateElimination(attr_names, domains)
    X = [
        ['Low', 'Hot', 'Poor'],
        ['Normal', 'Cool', 'Fertile'],
        ['Low', 'Hot', 'Fertile']
    ]
    y = [1, 0, 1]  # 1 = High Risk (Positive instance)
    
    S_final, G_final = ce.fit(X, y)
    
    assert len(S_final) > 0
    assert len(G_final) > 0
    
    # Specific boundary should generalize 'Soil' to '?'
    assert S_final == [['Low', 'Hot', '?']]
    # General boundary should specialize Rainfall and Heat
    assert ['Low', '?', '?'] in G_final
    assert ['?', 'Hot', '?'] in G_final
