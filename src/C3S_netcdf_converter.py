# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # IGCC data formatting
#
# Indicators data reformatting form csv files in https://github.com/ClimateIndicator/data to IAMC and NetCDF files

# %% [markdown]
# ## Imports

# %%
from functools import partial

import numpy as np
import pandas as pd
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from pathlib import Path
from gcages.renaming import convert_variable_name
from pandas_openscm.io import load_timeseries_csv
from pandas_openscm.index_manipulation import update_index_levels_func

import subprocess
from enum import StrEnum
import os
# NetCDF
from datetime import datetime, timezone
from uuid import uuid4
import math
import shutil
from typing import Any
import xarray as xr
import yaml

import pyam

# %% [markdown]
# ## Parameters

# %%
C3S_FOLDER = Path(
    "../data/data/"
)
model = "IGCC (2025)"
scenario = "Reference"
region = "World"
submission_version = "IGCC-2025a"

# %%
# File Mapping
file_map = {
    "greenhouse_gas_emissions/greehouse_gas_emissions_co2eq.csv":"ghg-emissions_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}",
    "greenhouse_gas_concentrations/ghg_concentrations.csv":"ghg-concentration_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}",
    #MULTIFILE here
    "effective_radiative_forcing/ERF_best.csv":f"effective-radiative-forcing_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}", 
    #mv and rename file: anthropogenic_warming_by_species/ERF*.csv
    "effective_radiative_forcing/SmithHausfather_ERF_bysource_from1750_table.csv":f"effective-radiative-forcing_global_input-datasets_method-ch7_table_{submission_version}",
    "earth_energy_imbalance/earth_energy_imbalance.csv":f"earth-energy-imbalance_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}",
    "global_mean_temperatures/annual_averages.csv":f"temp-observed_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}",
    "global_mean_temperatures/decadal_averages.csv":f"temp-observed_global_igcc-assessment_synthesis_10yr-timeseries_{submission_version}",
    "global_mean_temperatures/twenty_year_averages.csv":f"temp-observed_global_igcc-assessment_synthesis_20yr-timeseries_{submission_version}",
    "anthropogenic_warming/Gillett_GMST_timeseries.csv":f"temp-attribution_global_input-datasets_method-rof_1yr-timeseries_{submission_version}",
    "anthropogenic_warming/Gillett_GMST_rates.csv":f"temp-attribution_global_input-datasets_method-rof_rate_{submission_version}",
    # anthropogenic_warming/Gillett_GMST_headlines.csv + anthropogenic_warming/Gillett_GSAT_headlines.csv
    "anthropogenic_warming/Gillett_GMST_headlines.csv":f"temp-attribution_global_input-datasets_method-rof_headline_{submission_version}",
    "anthropogenic_warming/Ribes_GMST_timeseries.csv":f"temp-attribution_global_input-datasets_method-kcc_1yr-timeseries_{submission_version}",
    "anthropogenic_warming/Ribes_GMST_rates.csv":f"temp-attribution_global_input-datasets_method-kcc_rate_{submission_version}",
    "anthropogenic_warming/Ribes_GMST_headlines.csv":f"temp-attribution_global_input-datasets_method-kcc_headline_{submission_version}",
    "anthropogenic_warming/Walsh_GMST_timeseries.csv":f"temp-attribution_global_input-datasets_method-gwi_1yr-timeseries_{submission_version}",
    "anthropogenic_warming/Walsh_GMST_rates.csv":f"temp-attribution_global_input-datasets_method-gwi_rate_{submission_version}",
    "anthropogenic_warming/Walsh_GMST_headlines.csv":f"temp-attribution_global_input-datasets_method-gwi_headline_{submission_version}",
    "anthropogenic_warming/Assessment-Update-2025_GMST_headlines.csv":f"temp-attribution_global_igcc-assessment_synthesis_headline_{submission_version}",
    "anthropogenic_warming/Assessment-Extrapolation-2026_GMST_headlines.csv":f"temp-attribution_global_igcc-assessment_synthesis_future-extrapolation_{submission_version}",
    #move anthropogenic_warming_by_species/temperature_relative_to_1750.csv → anthropogenic_warming/SmithHausfather_GSAT_bysource_from1750_table.csv
    # anthropogenic_warming/SmithHausfather_GSAT_bysource_from1750_table.csv (rename + move from anthropogenic_warming_by_species/temperature_relative_to_1750.csv)
    "anthropogenic_warming/SmithHausfather_GSAT_bysource_from1750_table.csv":f"temp-attribution_global_input-datasets_method-ch7_table_{submission_version}",
    # merge all 3
    "carbon_budget/budget_normal_magicc_True_fair_False_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.0_asym_False_hdT_1.24NonlinNonCO2_all_None_recEm213.csv":
    f"co2-budget_global_igcc-assessment_synthesis_table_{submission_version}",
    "extreme_temperatures/txx_timeseries.csv":f"temp-extremes-land_land_input-datasets-datasets-all_1yr-timeseries_{submission_version}",
    "extreme_temperatures/txx_decadal_means.csv":f"temp-extremes-land_land_input-datasets_datasets-all_10yr-timeseries_{submission_version}",
    # Occhio qui Thristram vuole mettere la media in un'altro file(temperature-heatwaves-marine_marine_igcc-assessment_synthesis_1yr-timeseries_IGCC-2025a.nc).
    # Vediamo cosa dice. 
    "marine_heatwave_days/MHW_days_per_year_combined.csv":f"temperature-heatwaves-marine_marine_input-datasets_datasets-all_1yr-timeseries_{submission_version}",
    # merge of: sea_level_rise/altimetry_ens.csv +sea_level_rise/altimetry_indiv_estimates.csv
    "sea_level_rise/altimetry_ens.csv":f"sea-level-rise_global_input-datasets_datasets-altimetry_1yr-timeseries_{submission_version}",
    "sea_level_rise/AR6_GMSL_altimeter_FGD.csv":f"sea-level-rise_global_input-datasets_ar6-altimetry_1yr-timeseries_{submission_version}",
    "sea_level_rise/AR6_GMSL_TG_ensemble_FGD.csv":f"sea-level-rise_global_input-datasets_ar6-tidegauge_1yr-timeseries_{submission_version}",
    "sea_level_rise/AR6_GMSL_reconstructions_FGD.csv":f"sea-level-rise_global_input-datasets_ar6-reconstruction_1yr-timeseries_{submission_version}",
    "sea_level_rise/IGCC_GMSL_ensemble.csv":f"sea-level-rise_global_igcc-assessment_synthesis_1yr-timeseries_{submission_version}",
    "global_land_precipitation/global_land_precipitation.csv":f"precipitation-land_land_input-datasets_datasets-all_1yr-timeseries_{submission_version}",
}

# %% [markdown]
# ## Download from https://github.com/ClimateIndicator/data

# %%
repo_url = "https://github.com/ClimateIndicator/data.git"
target_dir = "../data"

def update_repo():
    target_path = os.path.normpath(target_dir)
    
    if os.path.isdir(os.path.join(target_path, ".git")):
        # Repo exists → discard untracked files and pull updates
        
        # First, do a dry run to see what would be removed (optional, for safety)
        # subprocess.run(["git", "-C", target_path, "clean", "-n", "-fd"], check=True)
        
        # Discard all untracked files and directories
        subprocess.run(
            ["git", "-C", target_path, "clean", "-fd"],
            check=True
        )
        
        # Now pull updates
        subprocess.run(
            ["git", "-C", target_path, "pull"],
            check=True
        )
    else:
        # Repo doesn't exist → clone it
        subprocess.run(
            ["git", "clone", repo_url, target_path],
            check=True
        )

update_repo()

# %% [markdown]
# # NetCDF

# %%
SCRIPT_DIR = Path().parent
GENERAL_METADATA_FILE = SCRIPT_DIR / "metadata_general.yaml"
SPECIFIC_METADATA_FILE = SCRIPT_DIR / "metadata_specific.yaml"


# %%
def year2dates(df: pd.DataFrame, reference_year: int)-> pd.DataFrame:

    year = datetime(reference_year,1, 1, 0, 0)
    df_out = pd.DataFrame()
    
    if np.issubdtype(df["time"].dtype, np.floating):
        
        df_out["time"] = df["time"].map(lambda y: (datetime(int(y), 7, 1) - year).days)
        df_out["timebound_lower"] = df["timebound_lower"].map(lambda y: (datetime(int(y), 1, 1) - year).days)
        df_out["timebound_upper"] = df["timebound_upper"].map(lambda y: (datetime(int(y), 1, 1) - year).days)
        
    elif np.issubdtype(df["time"].dtype, np.integer):
        df_out = df.map(lambda y: (datetime(int(y), 1, 1) - year).days)
        
    else:
        raise ValueError

    return df_out
    
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def process_erf(file_path:Path, df: pd.DataFrame, df_time_in_days: pd.DataFrame)->xr.DataArray:

    folder = file_path.parent

    file_list = ["ERF_p05.csv","ERF_p05_aggregates.csv",
                "ERF_p95.csv","ERF_p95_aggregates.csv"]
    
    df_aggr_best = pd.read_csv(
            folder/"ERF_best_aggregates.csv",
            engine="python",
            comment="#",
            skip_blank_lines=True,
        )
    df_aggr_best.columns = df_aggr_best.columns.str.replace("-", "_", regex=False)
    df_median = pd.merge(df,df_aggr_best, how="outer")
    
    mask = df_median.columns.str.startswith("time")
    time_df = df_median.iloc[:,mask]
    df_time_in_days = year2dates(time_df, REF_YEAR)
    data_df_median = df_median.iloc[:,~mask].copy()
    data_df_median["time"] = df_time_in_days["time"]
    
    df_list = []
    for i in range(0,4,2):
        df = pd.read_csv(
            folder/file_list[i],
            engine="python",
            comment="#",
            skip_blank_lines=True,
        )
        df.columns = df.columns.str.replace("-", "_", regex=False)
        df_aggr = pd.read_csv(
            folder/file_list[i+1],
            engine="python",
            comment="#",
            skip_blank_lines=True,
        )
        df_aggr.columns = df_aggr.columns.str.replace("-", "_", regex=False)
        df_mrg = pd.merge(df,df_aggr, how="outer")
    
        mask = df_mrg.columns.str.startswith("time")
        df_mrg = df_mrg.iloc[:,~mask]
        df_mrg["time"] = df_time_in_days["time"]
    
        df_list.append(df_mrg)
    
    datasets = []
    
    for label, df_data in [
        ("p05", df_list[0]),
        ("median", data_df_median),
        ("p95", df_list[1]),
    ]:
        ds_tmp = xr.Dataset.from_dataframe(df_data.set_index("time"))
    
        ds_tmp = ds_tmp.assign_coords(
            time=("time", df_time_in_days["time"].squeeze().to_numpy())
        )
    
        datasets.append(ds_tmp)
    
    ds = xr.concat(
        datasets,
        dim=xr.DataArray(
            ["p05", "p50", "p95"],
            dims="percentile",
            name="percentile",
        ),
        join="outer"
    )
    return ds

def process_relative_to_1750(df: pd.DataFrame, general:dict, specific:dict, file: str, ref_year: int)->xr.DataArray:
    
    compounds_emitted = df['Compound emitted']
    mask = df.columns.str.contains("Total")
    df_totals = df[df.columns[mask]]
    df_compound = df[df.columns[~mask]]
    df_compound = df_compound.drop(columns={"Compound emitted"})

    variables = list(specific["files"][file]["variable"].keys())

    current_year = specific["files"][file]["current_year"]

    df_time_in_days = year2dates(
        pd.DataFrame({
            "time": [current_year+0.5],
            "timebound_lower": [current_year],
            "timebound_upper": [current_year+1],
        }), 
        ref_year
    )
    
    ds = xr.Dataset(
        {
            variables[0]: (('compound_emitted','percentile','time'),df_totals.values[:, :, None]),
            variables[1]: (('compound_emitted','forcing_agent','time'),df_compound.values[:, :, None])
        },
        coords={
            "compound_emitted": compounds_emitted,
            "forcing_agent": df_compound.columns,
            "percentile": ["p05","p50","p95"],
            "time": df_time_in_days["time"],
        },
    )

    # Attributes 
    ds["time"].attrs.update(general["dimensions"]["time"])
    
    # Bounds
    ds["time_bnds"] = (("time", "nbnds"), df_time_in_days[["timebound_lower","timebound_upper"]])

    # Encoding
    ds["time"].encoding = general["default_dimensions_encoding"]
    ds["time_bnds"].encoding = general["default_dimensions_encoding"]
    ds.encoding["unlimited_dims"] = {"time"}
    # # Add flexible global attrs
    # ds.attrs["time_coverage_start"] = datetime(int(df["time"].min()),math.ceil(df["time"].min()%1)*6+1,1).strftime("%Y-%m-%dT%H:%M:%SZ")
    # ds.attrs["time_coverage_end"] =   datetime(int(df["time"].max()),math.ceil(df["time"].max()%1)*6+1,1).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ds

def process_anthropogenic(df: pd.DataFrame, df_time_in_days: pd.DataFrame, percentiles: list[str])->xr.DataArray:
    
    datasets = []
    
    for p in percentiles:
        
        mask = df.columns.str.endswith(p)
        data = df.iloc[:, mask].copy()
        data.columns = data.columns.map(lambda c: "_".join(c.split("_")[:-1]))
        data["time"] = df_time_in_days["time"]
        data = data.melt(
            id_vars=["time"],
            value_vars=[c for c in data.columns],
            var_name="component",
            value_name="GMST"
        )
        
        ds_tmp = xr.Dataset.from_dataframe(data.set_index(["component","time"]))
        ds_tmp = ds_tmp.assign_coords(
            time=("time", df_time_in_days["time"].squeeze().to_numpy())
        )
    
        datasets.append(ds_tmp)
    
    ds = xr.concat(
        datasets,
        dim=xr.DataArray(
            percentiles,
            dims="percentile",
            name="percentile",
        ),
    )
    
    return ds

def process_emissions(df: pd.DataFrame, df_time_in_days: pd.DataFrame)->xr.DataArray:
    
    accounting_method = ["WGIII",]
    
    datasets = []
    
    for acc in accounting_method:
        
        data = data_df.copy()
        data["time"] = df_time_in_days["time"]
        
        ds_tmp = xr.Dataset.from_dataframe(data.set_index("time"))
        ds_tmp = ds_tmp.assign_coords(
            time=("time", df_time_in_days["time"].squeeze().to_numpy())
        )
    
        datasets.append(ds_tmp)
    
    ds = xr.concat(
        datasets,
        dim=xr.DataArray(
            accounting_method,
            dims="accounting_method",
            name="accounting_method",
        ),
    )

    return ds

def process_headers(file_path:Path, general:dict, specific:dict, ref_year: int)->xr.Dataset:

    name = file_path.stem
    
    df_gmst = pd.read_csv(
        file_path,
        engine="python",
        comment="#",
        skip_blank_lines=True,
        na_values=[" ", "  ", "\t"],
    )
    df_gmst.columns = df_gmst.columns.str.replace("-", "_", regex=False)
    
    last_col = df_gmst.columns[-1]
    
    if last_col.endswith("p95"):
        df_gmst_sr15 = pd.DataFrame()

        mask = (df_gmst["timebound_upper"]-df_gmst["timebound_lower"])==1
        df_gmst_annual = df_gmst[mask].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")] # Drop empty name columns
        
    else:
        df_gmst_sr15 = df_gmst[df_gmst[last_col].str.contains("SR", na=False)].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")] # Drop empty name columns
        
        mask = ((~df_gmst[last_col].str.contains("SR", na=False)) & (df_gmst["timebound_upper"]-df_gmst["timebound_lower"])==1)
        df_gmst_annual = df_gmst[mask].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")] # Drop empty name columns

    mask = ((df_gmst["timebound_upper"]-df_gmst["timebound_lower"])>1)
    df_gmst_ar6 = df_gmst[mask].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")] # Drop empty name columns
    
    df_list = [
        ("GMST","Annual",df_gmst_annual),
        ("GMST","SR1.5",df_gmst_sr15),
        ("GMST","AR6",df_gmst_ar6),
    ]
    
    if name == "Gillett_GMST_headlines":
        
        df_gsat = pd.read_csv(
            file_path.parent / "Gillett_GSAT_headlines.csv",
            engine="python",
            comment="#",
            skip_blank_lines=True,
            na_values=[" ", "  ", "\t"],
        )
        df_gsat.columns = df_gsat.columns.str.replace("-", "_", regex=False)
        
        last_col = df_gsat.columns[-1]
        df_gsat_sr15 = df_gsat[df_gsat[last_col].str.contains("SR", na=False)].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")] 
        
        mask = ((df_gsat["timebound_upper"]-df_gsat["timebound_lower"])>1)
        df_gsat_ar6 = df_gsat[mask].drop(columns=["notes"], errors="ignore").loc[:, lambda df: ~df.columns.str.contains("^Unnamed")]
        
        df_list = [
            ("GSAT","AR6",df_gsat_ar6),
            ("GSAT","SR1.5",df_gsat_sr15),
            ("GMST","Annual",df_gmst_annual),
            ("GMST","SR1.5",df_gmst_sr15),
            ("GMST","AR6",df_gmst_ar6),
        ]
        
    files_with_less_percentiles = ['assessment_6thIPCC_headlines',
                                   'Assessment-Extrapolation-2026_GMST_headlines',
                                   'Assessment-Update-2025_GMST_headlines']
    
    percentiles = ["p05", "p17", "p50", "p83", "p95"] if name not in files_with_less_percentiles else ["p05", "p50", "p95"]

    df_list = [(temp, descr,df) for temp, descr,df in df_list if not df.empty]
    
    datasets = []
    t_bounds_list = []
    gsat_nc = []
    gmst_nc = []
    
    for temp, definition, df in df_list:

        mask = df.columns.str.startswith("time")
        time_df = df.iloc[:,mask]
        data_df = df.iloc[:,~mask]
        
        df_time_in_days = year2dates(time_df, ref_year)
        t_bounds_list.append(df_time_in_days)
        
        for p in percentiles:
            
            mask = df.columns.str.endswith(p)
            data = df.iloc[:, mask].copy()
            data.columns = data.columns.map(lambda c: "_".join(c.split("_")[:-1]))
            data["time"] = df_time_in_days["time"]
            data["definition"]=definition
    
            data = data.melt(
                id_vars=["time","definition"],
                value_vars=[c for c in data.columns if c != "definition"],
                var_name="component",
                value_name=temp
            )
    
            ds_tmp = xr.Dataset.from_dataframe(data.set_index(["definition","component","time"]))
            # Assign coordiantes when assinging a tuple wants an array but if time is a single value 
            # we need to force it to be an array
            time_values = np.atleast_1d(df_time_in_days["time"].squeeze())
            ds_tmp = ds_tmp.assign_coords(
                time=("time", time_values),
            )

            datasets.append(ds_tmp)
            
        ds_aux = xr.concat(
            datasets,
            dim=xr.DataArray(
                percentiles,
                dims="percentile",
                name="percentile",
            ),
            join="outer"
        )

        if temp == "GSAT":
            gsat_nc.append(ds_aux)
        else:
            gmst_nc.append(ds_aux)
        datasets.clear()

    ds = xr.concat(gmst_nc, dim="definition", join="outer")
    
    if name == "Gillett_GMST_headlines":
        ds_gsat = xr.concat(gsat_nc, dim="definition", join="outer")
        ds = xr.merge([ds_gsat, ds], compat="no_conflicts", join="outer")
    
    # Attributes 
    ds["time"].attrs.update(general["dimensions"]["time"])
    
    # # Bounds
    bounds = pd.concat(t_bounds_list, ignore_index=True).drop_duplicates()
    ds["time_bnds"] = (("time", "nbnds"), bounds.set_index("time").reindex(ds["time"])[["timebound_lower", "timebound_upper"]])#np.unique(np.array(t_bounds_list),axis=0))
    
    # Encoding
    ds["time"].encoding = general["default_dimensions_encoding"]
    ds["time_bnds"].encoding = general["default_dimensions_encoding"]
    ds.encoding["unlimited_dims"] = {"time"}
    
    # Add flexible global attrs
    ds.attrs["time_coverage_start"] = datetime(int(df["timebound_lower"].min()),1,1).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["time_coverage_end"] =   datetime(int(df["timebound_upper"].max()),1,1).strftime("%Y-%m-%dT%H:%M:%SZ")

    return ds

def process_carbon_budget(file_path:Path)->xr.Dataset():

    path_list = [file_path,
                file_path.parent / "budget_normal_magicc_True_fair_False_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.0_asym_False_hdT_1.26NonlinNonCO2_all_None_recEm213.csv",
                file_path.parent / "budget_normal_magicc_True_fair_True_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.19_asym_False_hdT_1.24NonlinNonCO2_all_None_recEm213.csv",
    ]
    parameters_list = ["magicc_hdtbasis1.24_zecsd0.0",
                  "magicc_hdt_basis1.26_zecsd0.0",
                  "magiccfair_hdt_basis1.24_zecsd0.19",
                 ]
    df_list  = []

    for i in range(3):
        df = pd.read_csv(
                path_list[i],
                engine="python",
                comment="#",
                skip_blank_lines=True,
                na_values=[" ", "  ", "\t"],
            )
        df.columns = df.columns.str.replace("-", "_", regex=False)
        df_list.append((parameters_list[i],df))


    datasets = []
    nc_files = []
    
    for parameters, df in df_list:
        
        mask = df.columns.str.startswith("0")
        likelihood = np.array(df.columns[mask], dtype=float)
        dT_targets = df["dT_targets"]
        future_warming = df["Future_warming"]
        df = df.drop(columns="Future_warming")
        
        for dt in dT_targets:
            
            remaining_budget = (df[df["dT_targets"]==dt].iloc[0,1:])
            
            ds_tmp = xr.Dataset(
                data_vars={
                    "remaining_carbon_budget": ("likelihood", remaining_budget)
                }
            )
            ds_tmp = ds_tmp.assign_coords(
                likelihood=("likelihood", likelihood),
            )
            
            datasets.append(ds_tmp)
                
        ds_aux = xr.concat(
            datasets,
            dim=xr.DataArray(
                dT_targets,
                dims="dT_targets",
                name="dT_targets",
            ),
            join="exact"
        )
        datasets.clear()
        ds_aux["future_warming"] = (("dT_targets"), future_warming)
        ds_aux = ds_aux.assign_coords(
                parameters=("parameters", [parameters]),
            )
        
        nc_files.append(ds_aux)
    
    ds = xr.concat(nc_files, dim="parameters", join="outer",data_vars="all")

    return ds

REF_YEAR = 1750

# %%
general = load_yaml(GENERAL_METADATA_FILE)
specific = load_yaml(SPECIFIC_METADATA_FILE)

processing = general.get("processing", {})

output_dir = Path(processing.get("output_dir", "../data/netcdf"))
output_dir.mkdir(parents=True, exist_ok=True)

# %%
for file, metadata in specific.get("files", {}).items():

    file_path = C3S_FOLDER / file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    name = file_path.stem
    print(name)

    # Path("my_folder").mkdir(parents=True, exist_ok=True)
    file_dict = specific["files"][file]
    variables = list(file_dict["variable"].keys())
    
    df = pd.read_csv(
        file_path,
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )
    # df make-up and specifics
    if name == "MHW_days_per_year_combined":
        df.drop(columns=["mean"],inplace=True)
        name_copy = name + "_2.csv"
        shutil.copy(file_path, file_path.parent/ name_copy)

    elif name == "MHW_days_per_year_combined_2":
        df.drop(columns=["OISST","OSTIA","CRW","ERA5"],inplace=True)
        df.rename(columns={"mean":"mean"},inplace=True)
        name = name + "_mean"

    elif name == "IGCC_GMSL_ensemble":
        df.rename(columns={"mean":"GMSL_mean","std":"GMSL_standard_deviation"},inplace=True)
    elif name == "altimetry_ens":
        df.rename(columns={"mean":"GMSL_mean","std":"GMSL_standard_deviation"},inplace=True)
        df_ens = pd.read_csv(
            file_path.parent / "altimetry_indiv_estimates.csv",
            engine="python",
            comment="#",
            skip_blank_lines=True,
            na_values=[" ", "  ", "\t"],
        )
        df =  pd.concat([df,df_ens],axis=1,join="inner")
        df = df.loc[:,~df.columns.duplicated()]
        
    df.rename(columns={"year":"time"},inplace=True)
    if "time" in df.columns and df["time"].isna().any():
        print(f"File {file_path} has Nans times")
        df = df.dropna(subset=["time"]).copy()
        
    df.rename(columns={"GMST":"gmst_anomaly",},inplace=True)
    df.columns = df.columns.str.replace("-", "_", regex=False)
        
    # Adding coords
    if "time" in file_dict["dimensions"]:

        mask = df.columns.str.startswith("time")
        time_df = df.iloc[:,mask]
        data_df = df.iloc[:,~mask]
        
        df_time_in_days = year2dates(time_df, REF_YEAR)

        if name == "ERF_best":
            ds = process_erf(file_path, df, df_time_in_days)
            
        elif name in ["Gillett_GMST_timeseries","Ribes_GMST_timeseries","Walsh_GMST_timeseries",
                      "Gillett_GMST_rates","Ribes_GMST_rates","Walsh_GMST_rates"]:

            percentiles = ["p05", "p17", "p50", "p83", "p95"] if name != "Gillett_GMST_rates" else ["p05", "p50", "p95"]
            
            if name in ["Walsh_GMST_timeseries","Walsh_GMST_rates"]:
                df.rename(columns={"aerosol_radiation_interactions": "aerosol_radiation_interactions_p95"}, inplace=True)
                
            ds = process_anthropogenic(df, df_time_in_days,percentiles)

        elif name == "greehouse_gas_emissions_co2eq":
            ds = process_emissions(df, df_time_in_days)

        else: 
            if len(variables) == 1:
                
                coordinate = file_dict["dimensions"][0]
                
                da = xr.DataArray(
                    data_df.values.T.squeeze(),
                    dims= tuple(file_dict["dimensions"]),
                    coords={
                        coordinate: data_df.columns,
                        "time": df_time_in_days["time"],
                    },
                    name = variables[0]
                )
                ds = da.to_dataset()
                
            else:
                ds = xr.Dataset(
                    {
                        var: (
                            tuple(file_dict["dimensions"]),
                            data_df[var].values.T,
                        )
                        for var in variables
                    },
                    coords={
                        "time": df_time_in_days["time"],
                    },
                )
        
        # Bounds
        ds["time_bnds"] = (("time", "nbnds"), df_time_in_days[["timebound_lower","timebound_upper"]])
        # Attributes 
        ds["time"].attrs.update(general["dimensions"]["time"])
        # Encoding
        ds["time"].encoding = general["default_dimensions_encoding"]
        ds["time_bnds"].encoding = general["default_dimensions_encoding"]
        ds.encoding["unlimited_dims"] = {"time"}
        # Add flexible global attrs
        # ds.attrs["time_coverage_start"] = datetime(int(df["time"].min()),math.ceil(df["time"].min()%1)*6+1,1).strftime("%Y-%m-%dT%H:%M:%SZ")
        # ds.attrs["time_coverage_end"] =   datetime(int(df["time"].max()),math.ceil(df["time"].max()%1)*6+1,1).strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["time_coverage_start"] = datetime(int(df["timebound_lower"].min()),1,1).strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.attrs["time_coverage_end"] =   datetime(int(df["timebound_upper"].max()),1,1).strftime("%Y-%m-%dT%H:%M:%SZ")

    elif name in ['ERF_relative_to_1750','temperature_relative_to_1750']:
        ds = process_relative_to_1750(df, general, specific, file, REF_YEAR)

    elif name.endswith('headlines'):
        ds = process_headers(file_path=file_path, general=general, specific=specific, ref_year=REF_YEAR)
        
    elif name.startswith("budget_normal_magicc"):
        ds = process_carbon_budget(file_path)
    else:
        raise FileNotFoundError("File not found")

    # Add fixed global attrs
    ds.attrs.update(general["globals"])
    # Add flexible global attrs
    ds.attrs.update(file_dict["description"])
    ds.attrs["id"] = f"uuid:{uuid4()}"
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.attrs["date_created"] = run_time
    ds.attrs["history"] = f"Created at {run_time} using the xarray library in Python"

    encoding = {}
    # Add variable attrs and encoding
    for v in variables:
        ds[v].attrs.update(file_dict["variable"][v])
        encoding[v] = general["default_variable_encoding"]

    if "coordinate_description" in file_dict.keys():
        for c in file_dict["coordinate_description"].keys():
            ds[c].attrs.update(file_dict["coordinate_description"][c])
        
    if "dimension_description" in file_dict.keys():
        coord_descr = list(file_dict["dimension_description"].keys())[0]
        description = list(file_dict["dimension_description"].values())[0]
        ds[coord_descr] = ( (coordinate,), [description[s]["long_name"] for s in data_df.columns])

    name = name + ".nc"
    ds.to_netcdf(output_dir/name, encoding=encoding)

# %% [markdown]
# ## Variable like

# %%
ds = xr.open_dataset(output_dir / "altimetry_ens.nc")
ds

# %% [markdown]
# ## Table like with additional metadata coordinate for description

# %%
ds = xr.open_dataset(output_dir / "MHW_days_per_year_combined.nc")
ds

# %%
ds["dataset_description"].values

# %% [markdown]
# ## Table like

# %%
ds = xr.open_dataset(output_dir / "Gillett_GMST_headlines.nc")
ds

# %% [markdown]
# ## Mix

# %%
ds = xr.open_dataset(output_dir / "greehouse_gas_emissions_co2eq.nc")
ds
