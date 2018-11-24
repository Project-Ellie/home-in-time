def sample_query(columns, total=100, lower=0, upper=1):
    col_string=", ".join(columns)
    return """
    SELECT
        {0}
    FROM 
        `going-tfx.examples.ATL_JUNE_SIGNATURE` 
    where
        MOD(ABS(FARM_FINGERPRINT(
            CONCAT(DATE,AIRLINE,ARR)
        )) + DEP_T, {1}) >= {2} 
    and
        MOD(ABS(FARM_FINGERPRINT(
            CONCAT( DATE, AIRLINE, ARR)
        )) + DEP_T, {1}) < {3} 
    """.format(col_string, total, lower, upper)
    
    
def sample_queries(columns, fractions, rate=0.1):
    start = 0
    total = int(sum(fractions) / rate)
    res = []
    for f in fractions:
        f_ = int(f) 
        q = sample_query(columns, total, start, start+f_)
        start = start + f_
        res.append(q)
    return dict(zip(['train', 'eval', 'test'], res))