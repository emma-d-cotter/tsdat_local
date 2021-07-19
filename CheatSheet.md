# setting up tsdat cheat sheet

1. Set up pipeline_config.yml
    * in pipeline:
        - location is where the data was collected
        - dataset_name is an identifier for the dataset
        - qualifier - the name of the sensor
        - temporal is the sampling frequency *emailed max about format*
        - data_level - leave this at a1
    * in attributes
        - set all relevant dimensions 
        - define each variable
            * name is the column header (exactly) in the file
2. Setup storage_config_dev.yml
    * specify file pattern to look for
    * add delimiter (sep), and header length
    * when adding time variable, make sure to check the time format
3. customize pipeline.py
    * hook_customize_raw_datasets --> this is where you can operate on the raw dataset before any data formatting, etc. Typically used to merge multiple files into one xarray
    * hook_customize_dataset --> this is where you can customize the dataset after reading and but before cleaning. For example, instrument is flipped, take opposite (*-1)
    *hook_finalize_dataset --> any customaization applied after data cleaning (QC)
    * hook_generate_and_persist_plots --> format and save any plots of the data here. filenames can be generated with DSUtil.get_plot_filename
4. run run_pipeline.py!

        