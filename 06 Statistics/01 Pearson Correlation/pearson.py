def create_pearson_corr_query(x:str, y:str, x_on='id', y_on='id', group_by:list=[]):
    """
    @param x<str>: [table].[column]
    @param y<str>: [table].[column]
    @param x_on<str>: JOIN Column Name
    @param y_on<str>: JOIN Column Name
    """
    x_table, x_column = x.split('.')
    y_table, y_column = y.split('.')
    
    # GROUP BY
    group_columns = []
    group_sub_select = []
    group_select = []
    
    for i, g in enumerate(group_by):
        g_table, g_column = g.split('.')
        g_table = 'd1' if g_table == x_table else 'd2'
        group_columns.append(f'{g_table}.{g_column}')
        group_sub_select.append(f'{g_table}.{g_column} as name{i}')
        group_select.append(f'name{i}')
        
    group_columns = ', '.join(group_columns)
    group_sub_select = ', '.join(group_sub_select)
    group_select = ', '.join(group_select)
    
    group_by_sql = ''
    if group_columns:
        group_by_sql = f'GROUP BY {group_columns}'
        group_sub_select += ', '
        group_select += ', '
    
    query = '''SELECT {group_select}
(N * xy_psum - sum1 * sum2)/ 
    SQRT((N * sqsum1 - POW(sum1, 2)) * 
    (N * sqsum2 - POW(sum2, 2)))
FROM
    (SELECT {group_sub_select}
        COUNT(*) AS N,
        SUM(d1.{x} * d2.{y}) AS xy_psum,
        SUM(d1.{x}) AS sum1,
        SUM(d2.{y}) AS sum2,
        SUM(POW(d1.{x}, 2)) AS sqsum1,
        SUM(POW(d2.{y}, 2)) AS sqsum2
    FROM {x_table} AS d1
    LEFT JOIN {y_table} AS d2
    ON d1.{x_on} = d2.{y_on}
    {group_by_sql}) as pcorr
    '''.format(x_table=x_table, y_table=y_table, 
               x=x_column, y=y_column, 
               x_on=x_on, y_on=y_on, 
               
               group_select = group_select,
               group_sub_select= group_sub_select,
               group_by_sql=group_by_sql)
    return query