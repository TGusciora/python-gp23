import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from IPython.display import display, HTML


def var_boxplot(var, data, target, title, xrotate=0):
    """ Create boxplot of variable vs descending median of target
    and provide frequency table for variable values. Caution:
    target variable has to be numeric.

    Parameters
    ----------
    data : str
        DataFrame to plot and calculate frequencies
    var : str
        Variable name for analysis. Provide in quotation marks.
    target : str
        Target variable name for analysis. Provide in quotation marks.
    title : str
        Plot title
    xrotate : int
        Rotation parameter for plt.xticks. By default = 0.

    Notes
    -------------------
    Required libraries: \n
    * import matplotlib.pyplot as plt \n
    * import seaborn as sn \n
    * import pandas as pd \n
    * from IPython.display import display, HTML

    Returns
    -------
    Graphs : matplotlib plots
       Boxplot and frequency table.

    """
    plt.figure(figsize=(10, 5))
    sn.boxplot(x=var, y=target, data=data)
    plt.title(title)
    plt.xlabel(var)
    plt.ylabel('Median of '+str(target))
    plt.xticks(rotation=xrotate)
    plt.show()

    valueCounts = pd.concat([data[var].value_counts(),
                 data[var].value_counts(normalize=True).mul(100)], axis=1,
                 keys=('counts', 'percentage'))
    print(var, '\n', len(data[var].unique()), 'unique values: ')
    display(HTML(valueCounts.to_html()))
