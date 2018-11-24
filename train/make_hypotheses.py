def make_hypotheses():

    from train.hypothesis_wd1 import hypothesis_wd1
    from train.hypothesis_linear import hypothesis_linear
    
    return {
        'wide_and_deep': hypothesis_wd1,
        'linear': hypothesis_linear 
    }
