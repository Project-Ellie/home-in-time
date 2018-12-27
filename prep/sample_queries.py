def sample_queries(columns, fractions=[80,10,10], rate=0.1):
    """Reproducible random sets of flights from Atlanta in any month of June.

    Requires Bigquery table ATL_JUNE_SIGNATURE to be present. This function is meant
    to encapsulate the aspect of reproducible random sets from the rest of the code.
    
    Creates three statements (strings), each of which will produce a non-intersecting
    random subset of the data in ATL_JUNE_SIGNATURE table. Will return a map with keys
    'train', 'eval', and 'test' to hold those statements. 
    
    Args:
        columns: A string holding a comma-separated list of column names to fetch.
        fractions: an array of exactly three doubles defining the precise splits.
        rate: an additional parameter determining the overall sample rate. 0.1 means 10% is
        being split into the three given fractions.
        
    Returns:
        a dict like {'train': stmt1, 'eval': stmt2, 'test': stmt3}, where stmt1, stmt2, stmt3
        are three strings that represent valid statements against bigquery.
    """
    
    def sample_query(columns, total, lower, upper):
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
    
    start = 0
    total = int(sum(fractions) / rate)
    res = []
    for f in fractions:
        f_ = int(f) 
        q = sample_query(columns, total, start, start+f_)
        start = start + f_
        res.append(q)
    return dict(zip(['train', 'eval', 'test'], res))
