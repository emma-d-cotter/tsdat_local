import os
from typing import Dict

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tsdat.pipeline import IngestPipeline
from tsdat.utils import DSUtil

example_dir = os.path.abspath(os.path.dirname(__file__))
style_file = os.path.join(example_dir, "styling.mplstyle")
plt.style.use(style_file)


class Pipeline(IngestPipeline):
    """Example tsdat ingest pipeline used to process lidar instrument data from
    a buoy stationed at Morro Bay, California.

    See https://tsdat.readthedocs.io/ for more on configuring tsdat pipelines.
    """

    def hook_customize_raw_datasets(self, raw_dataset_mapping: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """-------------------------------------------------------------------
        Hook to allow for user customizations to one or more raw xarray Datasets
        before they merged and used to create the standardized dataset.  The
        raw_dataset_mapping will contain one entry for each file being used
        as input to the pipeline.  The keys are the standardized raw file name,
        and the values are the datasets.

        This method would typically only be used if the user is combining
        multiple files into a single dataset.  In this case, this method may
        be used to correct coordinates if they don't match for all the files,
        or to change variable (column) names if two files have the same
        name for a variable, but they are two distinct variables.

        This method can also be used to check for unique conditions in the raw
        data that should cause a pipeline failure if they are not met.

        This method is called before the inputs are merged and converted to
        standard format as specified by the config file.

        Args:
        ---
            raw_dataset_mapping (Dict[str, xr.Dataset])     The raw datasets to
                                                            customize.

        Returns:
        ---
            Dict[str, xr.Dataset]: The customized raw dataset.
        -------------------------------------------------------------------"""
        return raw_dataset_mapping

    def hook_customize_dataset(self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]) -> xr.Dataset:
        """-------------------------------------------------------------------
        Hook to allow for user customizations to the standardized dataset such
        as inserting a derived variable based on other variables in the
        dataset.  This method is called immediately after the apply_corrections
        hook and before any QC tests are applied.

        Args:
        ---
            dataset (xr.Dataset): The dataset to customize.
            raw_mapping (Dict[str, xr.Dataset]):    The raw dataset mapping.

        Returns:
        ---
            xr.Dataset: The customized dataset.
        -------------------------------------------------------------------"""

       # dataset['mass'].data = dataset['mass'].data + 5
       # dataset['mass'].attrs['correctionsapplied'] = 'added 5 because why not'

        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """-------------------------------------------------------------------
        Hook to apply any final customizations to the dataset before it is
        saved. This hook is called after quality tests have been applied.

        Args:
            dataset (xr.Dataset): The dataset to finalize.

        Returns:
            xr.Dataset: The finalized dataset to save.
        -------------------------------------------------------------------"""
        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset) -> None:
        """-------------------------------------------------------------------
        Hook to allow users to create plots from the xarray dataset after
        processing and QC have been applied and just before the dataset is
        saved to disk.

        To save on filesystem space (which is limited when running on the
        cloud via a lambda function), this method should only
        write one plot to local storage at a time. An example of how this
        could be done is below:

        ```
        filename = DSUtil.get_plot_filename(dataset, "sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(dataset["time"].data, dataset["sea_level"].data)
            fig.save(tmp_path)
            storage.save(tmp_path)

        filename = DSUtil.get_plot_filename(dataset, "qc_sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            DSUtil.plot_qc(dataset, "sea_level", tmp_path)
            storage.save(tmp_path)
        ```

        Args:
        ---
            dataset (xr.Dataset):   The xarray dataset with customizations and
                                    QC applied.
        -------------------------------------------------------------------"""
        # extract QC info and create masks
        # NOTE: this is hard coded for a max of 4 QC checks. Increase .zfill(4) to .zfill(x) if more QC checks are added
        mask_keys = dataset['qc_elevation'].attrs['flag_masks']
        mask_meanings = dataset['qc_elevation'].attrs['flag_meanings']

        masks_bin = [bin(int(x))[2:].zfill(4)[::-1] for x in mask_keys]
        masks_bin_idx = [x.find('1')+1 for x in masks_bin]

        masks = np.zeros((len(dataset['qc_elevation']),len(mask_keys)))
        for i, idx in enumerate(masks_bin_idx):
            qc_bin = [bin(int(x))[2:].zfill(4)[::-1] for x in dataset['qc_elevation'].values]
            masks[:,i] = [bool(int(x[idx])) for x in qc_bin]
        bad_data = np.sum(masks,1).astype(bool)

        # plot data that passed QC
        filename = DSUtil.get_plot_filename(dataset, "elevation", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3),constrained_layout=1)
            fig.suptitle('Tidal Elevation (after QC)')
            y = dataset['elevation']
            y[bad_data] = np.nan
            ax.plot(dataset['time'],y)
            fig.savefig(tmp_path,dpi=100)
            self.storage.save(tmp_path)
                        
        # plot QC results
        filename = DSUtil.get_plot_filename(dataset, "qc_elevation", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,3),constrained_layout=1)
            fig.suptitle('Tidal Elevation QC')
            for i, idx in enumerate(mask_keys):
                y = idx*np.ones(len(dataset['elevation']))
                mask = masks[:,i].astype(bool)
                y[np.invert(mask)] = np.nan
                ax.plot(dataset['time'],y,'o')

            y = np.zeros(len(dataset['elevation']))
            y[bad_data] = np.nan
            ax.plot(dataset['time'],y,'o')
            ax.set_ylim((0,np.max(masks_bin_idx)))
            ax.set_yticks(np.arange(0,np.max(masks_bin_idx)+1))
            ax.set_yticklabels(['good_data'] + mask_meanings)
            fig.savefig(tmp_path,dpi=100)
            self.storage.save(tmp_path)

        return
