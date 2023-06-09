import pandas as pd
import os


class MakeDataset:
    """
    Class creating pandas DataFrame from raw file_name.

    Creates MakeDataset.data attribute, which represents pandas DataFrame
    from imported project/data/raw/file_name file. Files from other directories
    or files that are not delimited by tabulation (delimiter = "t") will raise
    errors.

    Parameters
    -----------
    file_name : str
        Raw file name to be imported. Has to be in data/raw/ folder.

    Attributes
    ----------
    data : pandas DataFrame
        imported file
    delimiter :str
        Delimiter in imported raw file. Used as pandas read_csv parameter.

    Notes
    -------------------
    Imported raw file should be a .csv file. \n
    Required libraries: \n
    * import pandas as pd \n
    * import os

    Methods
    -------
    __init__(self, file_name)
        Constructor method.
    _import_dataset(self)
        Imports file_name and returns as self.data pandas DataFrame.
    """

    def __init__(self, file_name):
        """
        Constructor method
        """
        self.file_name = file_name
        self.delimiter = "\t"
        self.data = self._import_dataset()

    def _import_dataset(self):
        """
        Importing raw file from project/data/raw/ folder.

        Establishing raw data relative location based script location.

        Returns
        --------
        data : pandas DataFrame
            Imported file.
        """
        # Return main package / project directory
        absolute_path = os.path.abspath(os.path.join(__file__, "../../.."))
        # Subdirectory with raw data
        relative_path = "\\data\\raw\\"
        # Establish full path to raw file
        full_path = absolute_path + relative_path + self.file_name
        # Import raw file as data pandas DataFrame
        data = pd.read_csv(full_path, self.delimiter)
        return data
