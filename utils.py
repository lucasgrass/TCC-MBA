# Based on IQR
def drop_outliers(df, column):
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    #print(lower_bound)
    #print(upper_bound)
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]