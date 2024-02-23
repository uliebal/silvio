

from typing import Union, Generic, TypeVar, List
from dataclasses import dataclass
import pathlib
import pandas as pd
from openpyxl import load_workbook

from pandas import DataFrame, Series
import matplotlib.pyplot as plt



T = TypeVar('Value')



class SimulationException (Exception) :
    """ Base class for all simulation exceptions. """
    pass



@dataclass(frozen=True)
class Outcome (Generic[T]) :
    """ Return from operations with any type of result. """
    value: T
    error: Union[str,None] = None

    def has_error ( self ) -> bool :
        return self.error is not None

    def succeeded ( self ) -> bool :
        return not self.has_error()



@dataclass(frozen=True)
class DataOutcome (Outcome) :
    """
    Holds the dataframe of a simulation. Has methods to access whether it worked successfully, to
    print the data or to store it in files.
    """
    value: Union[DataFrame,Series]

    def __str__ ( self ) -> str :
        return str(self.value)

    # def has_error ( self ) -> bool :
    #     return self.error is not None

    # def succeeded ( self ) -> bool :
    #     return not self.has_error()

    def display_data ( self ) -> None :
        if self.value is not None :
            # old_max_rows = pd.options.display.max_rows
            # old_max_columns = pd.options.display.max_columns
            # pd.options.display.max_rows = 100
            # pd.options.display.max_columns = 30
            print(self.value)
            # pd.options.display.max_rows = old_max_rows
            # pd.options.display.max_columns = old_max_columns
        else :
            print("Outcome is empty.")

    def export_data ( self, filepath ) -> None :
        abs_filepath = pathlib.Path(filepath).resolve()
        abs_filepath.parent.mkdir(parents=True, exist_ok=True) # Build non-existing parent dirs.
        self.value.to_csv( abs_filepath, index=False )
        print("Data exported to: {}".format(abs_filepath))

    def append_data2xlsx ( self, filepath , sheet) -> None :
        abs_filepath = pathlib.Path(filepath).resolve()

        book = load_workbook(abs_filepath)
        writer = pd.ExcelWriter(abs_filepath, engine='openpyxl')
        writer.book = book

        #######################################################
        # Metadata fields
        # Extract range from sheet 'ref'
        ref_sheet = book['Reference']
        data_range = ref_sheet['A1:C14']  # Example range, adjust as needed
        # Convert the extracted range into a list of lists
        data_list = [[cell.value for cell in row] for row in data_range]

        # Convert the list of lists into a DataFrame
        meta_df = pd.DataFrame(data_list, columns=['Metadata: Key', 'Metadata: Value', 'Metadata: Comment'])  # Adjust column names as needed
        meta_df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        #######################################################

        # rename columns of dataframe myDat
        self.value.rename(columns={'t': 'time', 'S': 'Substrate'}, inplace=True)

        # Load the DataFrame 'myDat' into a new sheet in the Excel file
        self.value.to_excel(writer, sheet_name=sheet, index=False, startrow=0, startcol=4)

        # Save the changes
        writer.save()
        # print(f"Data appended to: {abs_filepath} in sheet {sheet}")



@dataclass(frozen=True)
class DataWithPlotOutcome (DataOutcome) :

    def make_plot ( self ) -> plt.Figure :
        """
        Plotting with pyplot is unfortunately unintuitive. You cannot display a single figure by
        using the object-oriented API. When you do `plt.subplots` (or create a plot by any other
        means) it will be stored in the global context. You can only display things from the glbbal
        context, and displaying it will remove it from there.
        """
        fig, ax = plt.subplots()
        self.value.plot( y=self.value.columns, use_index=True, subplots=True, ax=ax )
        return fig

    def display_plot ( self ) -> None :
        plt.rcdefaults()
        self.make_plot()
        plt.show()

    def export_plot ( self, filepath ) -> None :
        abs_filepath = pathlib.Path(filepath).resolve()
        abs_filepath.parent.mkdir(parents=True, exist_ok=True) # Build non-existing parent dirs.
        plt.rcdefaults()
        fig = self.make_plot()
        plt.savefig(abs_filepath)
        plt.close(fig)
        print("Plot exported to: {}".format(abs_filepath))



def combine_data ( outcomes:List[DataOutcome] ) -> DataOutcome :
    """
    Combine multiple DataOutcome into one big table. Can combine multiple Series and Dataframes.
    TODO: Should it extract the error of each outcome? Right now it is being ignored.
    """

    # First convert all outcomes into dataframes (convert Series and pass DataFrames)
    dfs = []
    for outcome in outcomes :
        if isinstance( outcome.value, pd.DataFrame ) :
            dfs.append( outcome.value )
        elif isinstance( outcome.value, pd.Series ) :
            dfs.append( pd.DataFrame([outcome.value]) )

    # Make the join and outcome a dataframe.
    cout = pd.concat(dfs)
    cout.reset_index( drop=True, inplace=True )
    return DataOutcome( value=cout )
