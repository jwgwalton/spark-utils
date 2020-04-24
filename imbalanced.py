# Utility functions for use in problems with imbalanced data

#TODO
# Add balancing code
# package 

def compute_class_weights(df, label):
    """
    Compute a class weight for a label column. Use in algorithms which support class_weight such as 
    LogisticRegression
    """
    y_collect = df.select(label).groupBy(label).count().collect()
    unique_y = [x[label] for x in y_collect]
    total_y = sum([x["count"] for x in y_collect])
    unique_y_count = len(y_collect)
    bin_count = [x["count"] for x in y_collect]
    class_weights = {i: ii for i, ii in zip(unique_y, float(total_y) / (unique_y_count * np.array(bin_count)))}
    return class_weights


