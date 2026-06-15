# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Post process emissions
#
# Here we post process emissions arfter harmonization and infilling.

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
# repo_url = "https://github.com/ClimateIndicator/data.git"
# target_dir = "../data"

# def update_repo():
#     if os.path.isdir(os.path.join(target_dir, ".git")):
#         # Repo exists → pull updates
#         subprocess.run(["git", "-C", target_dir, "pull"], check=True)
#     else:
#         # Repo doesn't exist → clone it
#         subprocess.run(["git", "clone", repo_url, target_dir], check=True)

# update_repo()

# %% [markdown]
# ## Set up

# %%
"""
Variables database
"""
VARIABLES = pd.DataFrame(
    [# Concentations and emissions
        ("C2F6",   "PFC|C2F6"),
        ("C3F8",   "PFC|C3F8"),
        ("C8F18",  "PFC|C8F18",),
        ("CF4",    "PFC|CF4",),
        ("c-C4F8", "PFC|cC4F8",  ),
        ("n-C4F10","PFC|nC4F10", ),
        ("n-C5F12","PFC|nC5F12", ),
        ("n-C6F14","PFC|nC6F14", ),
        ("i-C6F14","PFC|iC6F14",),
        ("C7F16",  "PFC|C7F16",),
        #
        ("Halon-1211",  "Halon|1211",),
        ("Halon-1301",  "Halon|1301",),
        ("Halon-2402",  "Halon|2402",),
        #
        ("CH4",      "CH4", ),
        ("N2O",      "N2O", ),
        ("CO2",      "CO2", ),
        ("SF6",      "SF6", ),
        ("SO2F2",    "SO2F2", ),
        ("NF3",      "NF3",  ),
        #
        ("HFC-134a", "HFC|HFC134a"),
        ("HFC-23",   "HFC|HFC23",),
        ("HFC-32",   "HFC|HFC32",),
        ("HFC-125",  "HFC|HFC125",),
        ("HFC-143a", "HFC|HFC143a",),
        ("HFC-152a", "HFC|HFC152a",),
        ("HFC-227ea","HFC|HFC227ea",),
        ("HFC-236fa","HFC|HFC236fa",),
        ("HFC-245fa","HFC|HFC245fa",),
        ("HFC-365mfc","HFC|HFC365mfc",),
        ("HFC-43-10mee","HFC|HFC43-10",),
        #
        ("CFC-11",  "CFC|CFC11",),
        ("CFC-12",  "CFC|CFC12"),
        ("CFC-13",  "CFC|CFC13",),
        ("CFC-112", "CFC|CFC112",),
        ("CFC-112a","CFC|CFC112a"),
        ("CFC-113", "CFC|CFC113",),
        ("CFC-113a","CFC|CFC113a"),
        ("CFC-114", "CFC|CFC114",),
        ("CFC-114a","CFC|CFC114a",),
        ("CFC-115", "CFC|CFC115",),
        #
        ("CH3CCl3","CH3CCl3"),
        ("CCl4",   "CCl4"),
        ("CH3Cl",  "CH3Cl"),
        ("CH3Br",  "CH3Br"),
        ("CH2Cl2", "CH2Cl2"),
        ("CHCl3",  "CHCl3"),
        #
        ("HCFC-22",  "HCFC22"),
        ("HCFC-141b","HCFC141b"),
        ("HCFC-142b","HCFC142b"),
        #
        ("HCFC-133a","HCFC133a"),
        ("HCFC-31",  "HCFC31"),
        ("HCFC-124", "HCFC124"),		
        ("PFC[CF4-eq]",  "PFC [CF4-eq]"),
        ("HFC[HFC-134a-eq]",  "HFC [HFC134a-eq]"),
        ("CFC[CFC-12-eq]",  "CFC [CFC12-eq]"),
        # Emissions
        ("CO2-FFI",    "CO2|Fossil Fuels and Industry"),
        ("CO2-LULUCF", "CO2|LULUCF"),
        ("F-gases",    "F-Gases"),
        #
        ("volcanic", "Volcanic"),
        ("solar",    "Solar",   ),
        ("o3",       "O3",      ),
        ("n2o",      "N2O",     ),
        ("land_use", "Land Use",),
        ("halogen",  "Halogen", ),
        ("h2o",      "H2O",     ),
        ("h2o_strat","H2O-Strat",),
        ("contrails","Contrails",),
        ("co2",      "CO2",      ),
        ("ch4",      "CH4",      ),
        ("bc_snow",   "BC Snow", ),
        ("aerosol-radiation_interactions","Aerosol-Radiation Interactions",),
        ("aerosol-cloud_interactions",    "Aerosol-Cloud Interactions",    ),
        ("residual",           "Residual",),
        ("total",              "Total",),
        ("anthropogenic",      "Anthropogenic", ),
        ("ghg",                "GHGs",    ),
        ("natural",            "Natural", ),
        ("other_human_forcing","Other Human Forcings",),
        ("Multi-method mean","Multi-Method|Mean",),
        #
        ("GMST","Induced Warming",),
        #Carb Budg
        ("Future_warming",       "Future Warming",),
        ("0.1",       "Avoidance probability|10%",),
        ("0.17",      "Avoidance probability|17%",),
        ("0.33",      "Avoidance probability|33%",),
        ("0.5",       "Avoidance probability|50%",),
        ("0.67",      "Avoidance probability|67%",),
        ("0.83",      "Avoidance probability|83%",),
        ("0.9",       "Avoidance probability|90%",),
        # EEI
        ("atmosphere",   "Atmosphere",),
        ("cryosphere",   "Cryosphere",),
        ("land",         "Land",),
        ("ocean_0-700m",             "Ocean (0-700m)",),
        ("ocean_0-700m_error",       "Ocean (0-700m)|Error",),
        ("ocean_700-2000m",          "Ocean (700-2000m)",),
        ("ocean_700-2000m_error",    "Ocean (700-2000m)|Error",),
        ("ocean_2000-6000m",         "Ocean (2000-6000m)",),
        ("ocean_2000-6000m_error",   "Ocean (2000-6000m)|Error",),
        ("ocean_full-depth",         "Ocean Full Depth",),
        ("ocean_full-depth_error",   "Ocean Full Depth|Error",),
        # ERF
        ("anthro",      "Anthropogenic", ),
        ("BC_on_snow", "BC Snow", ),
        ("H2O_stratospheric","Stratospheric H2O",),
        ("nonco2wmghg",      "Non-CO2", ),
        ("O3",       "O3",      ),
        # EXTR-Temp TXX
        ("ERA5",        "ERA5",),
        ("Berkeley",    "Berkeley",),
        ("Berkeley_Earth",    "Berkeley",),
        ("HadEX3",      "HadEX3",),
        # MHW
        ("OISST",   "OISST",),
        ("OSTIA",   "OSTIA",),
        ("CRW",      "CRW",),
        # GLP
        ("GPCC",   "GPCC",),
        ("CRU",   "CRU",),
        ("GPCP",      "GPCP",),
        ("GHCN",      "GHCN",),
        # SLR
        ("mean",   "Mean",),
        ("std",    "Standard Deviation",),
        
    ],
    columns=["original", "iamc",],
)
"""
Database of emissions variables names according to different naming schemes
"""

ERF_VARIABLES = pd.DataFrame(
    [
        # ERF
        ("anthro",      "Anthropogenic", ),
        ("aerosol",     "Anthropogenic|Non-WMGHG|Aerosol",),
        ("aerosol-radiation_interactions","Anthropogenic|Non-WMGHG|Aerosol|Aerosol-Radiation Interactions",),
        ("aerosol-cloud_interactions",    "Anthropogenic|Non-WMGHG|Aerosol|Aerosol-Cloud Interactions",    ),
        ("CO2",        "Anthropogenic|WMGHG|CO2",      ),
        ("minor",      "Anthropogenic|Non-WMGHG|Minor", ),
        ("BC_on_snow", "Anthropogenic|Non-WMGHG|Minor|BC Snow", ),
        ("land_use",   "Anthropogenic|Non-WMGHG|Minor|Land Use", ),
        ("contrails",  "Anthropogenic|Non-WMGHG|Minor|Contrails", ),
        ("H2O_stratospheric","Anthropogenic|Non-WMGHG|Minor|Stratospheric H2O",),
        ("nonco2wmghg",      "Anthropogenic|WMGHG|Non-CO2", ),
        ("N2O",      "Anthropogenic|WMGHG|Non-CO2|N2O",     ),
        ("CH4",      "Anthropogenic|WMGHG|Non-CO2|CH4",     ),
        ("halogen",  "Anthropogenic|WMGHG|Non-CO2|Halogen", ),
        ("O3",       "Anthropogenic|Non-WMGHG|O3",      ),
        ("volcanic", "Natural|Volcanic"),
        ("solar",    "Natural|Solar",),
        ("natural",  "Natural",),
        ("wmghg",    "Anthropogenic|WMGHG",),
        ("anthro_nonwmghg", "Anthropogenic|Non-WMGHG",),
        ("total",    "Total",),
    ],
    columns=["original", "iamc",],
)
"""
Database of ERF variables names according to different naming schemes
"""
# Raw Surface Temperature (GMST)	K	Raw global-mean air ocean blended temperature change (GMST i.e. a blend of 2m air temperature over land and surface temperatures over the ocean; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)	K	Raw global-mean surface air temperature change (GSAT i.e. 2m air temperature; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|CO2	K	Raw global-mean surface air temperature change due to CO2 (note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|Non-CO2	K	Raw global-mean surface air temperature change due to non-CO2 climate drivers (note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|Residual	K	Raw Surface Temperature (GSAT) - Raw Surface Temperature (GSAT)|CO2 - Raw Surface Temperature (GSAT)|Non-CO2 i.e. the difference between the raw global-mean surface air temperature change and the sum of CO2 and non-CO2 contributions; the majority of this residual is due to natural climate forcers (e.g. volcanic eruptions and variations in the solar cycle; note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Surface Temperature (GSAT)	K	Global-mean surface air temperature change to be used for e.g. scenario categorisation (this output has been adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)

# %%
class SupportedNamingConventions(StrEnum):
    """Supported naming conventions"""

    IAMC = "iamc"
    """
    Integrated Assessment Modelling Consortium (IAMC) naming convention

    Not a perfect definition so the implementation here is a bit of a guess
    based on experience.
    https://github.com/IAMconsortium/common-definitions
    is a better source of truth, but it also moves more quickly,
    is not used universally and covers many more variables
    than we care about within the gcages context.
    """

    ORIGINAL = "original"

    C3S = "c3s"
    """
    Copernicus C3S format
    """

original2iamc = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.ORIGINAL,
    to_convention=SupportedNamingConventions.IAMC,
    database=VARIABLES,
)
original2iamc_erf = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.ORIGINAL,
    to_convention=SupportedNamingConventions.IAMC,
    database=ERF_VARIABLES,
)


# %% [markdown]
# ## Functions

# %%
def generic_csv2df(file_path,unit,mapper=original2iamc,prepend="",scenario = "Reference",year_column_name="time"):
    
    name = file_path.stem
    col_name = ['timebound_lower', "timebound_upper",year_column_name]
    df = pd.read_csv(file_path)
    
    if np.issubdtype(df[year_column_name].dtype, np.floating):
        df = df.drop(columns=[col_name[2],col_name[1]])
        df.rename(columns={col_name[0]: 'year'}, inplace=True)
    else:
        df = df.drop(columns=[col_name[2],col_name[0]])
        df.rename(columns={col_name[1]: 'year'}, inplace=True)
        df["year"] -= 1

    df['unit'] = unit
    df['model'] = "IGCC (2025)"
    df['scenario'] = scenario
    df['region'] = "World"
    
    indf=(df.melt(
                id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value'
            )
            .pivot_table(
                index=['model','scenario','region','variable','unit'], columns='year', values='value')
        )
    
    IAMC_from_C3S_df = update_index_levels_func(
        indf, {"variable": mapper}
    )
    
    res = IAMC_from_C3S_df.pix.format(
        variable=prepend+"{variable}",  # noqa: E501
        drop=True,
    )
        
    return res


# %%
def antropogenic(file_path,name,model,region,scenario):
    print(name)
    data_type = name.split("_")[-1]

    # DIFFERENT col name for Walsh_GMST_timeseries.csv ?? 
    #col_name = ['timebounds_lower', "timebounds_upper"] if name == "Walsh_GMST_timeseries" else ['timebound_lower', "timebound_upper"]
    col_name = ['timebound_lower', "timebound_upper"]
    
    if data_type == "timeseries":
        df = pd.read_csv(file_path).drop(columns=["time",col_name[1]], errors="ignore")
        df.rename(columns={col_name[0]: 'year'}, inplace=True)
        df['unit'] = "°C"
    elif data_type == "rates":
        df = pd.read_csv(file_path).drop(columns=["time",col_name[0]], errors="ignore")
        df.rename(columns={col_name[1]: 'year'}, inplace=True)
        cols = df.columns[~df.columns.str.contains("year")]
        df[cols] *= 10
        df["year"] -= 1
        df['unit'] = "°C/decade"
    else:
        raise ValueError
        
    df['model'] = model
    df['scenario'] = scenario if name == "multi_method_timeseries" else name.split("_")[0]
    df['region'] = region
    if name in ["Walsh_GMST_timeseries","Walsh_GMST_rates"]:
        df.rename(columns={"aerosol-radiation_interactions": "aerosol-radiation_interactions_p95"}, inplace=True)
    
    indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
    if name != "multi_method_timeseries":
        indf['percentile'] = indf.index.get_level_values('variable').str.split('_').str[-1].str[-2:]
        indf.set_index('percentile', append=True, inplace=True)
        indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
        indf.rename(index=lambda x: x+"th" if x[-1]!="3" else x+"rd", level='percentile', inplace=True)

    IAMC_from_C3S_df = update_index_levels_func(
        indf, {"variable": original2iamc}
    )

    if name == "multi_method_timeseries":
        res = IAMC_from_C3S_df.pix.format(
                variable="Induced Warming|{variable}",  # noqa: E501
                drop=True,
            )
    else:
        if data_type == "timeseries":
            res = IAMC_from_C3S_df.pix.format(
                variable="Induced Warming|{variable}|{percentile} Percentile",  # noqa: E501
                drop=True,
            )
        elif data_type == "rates":
            res = IAMC_from_C3S_df.pix.format(
                variable="Induced Warming [Rate]|{variable}|{percentile} Percentile",  # noqa: E501
                drop=True,
            )

        res.rename(index=lambda x: x.replace("50th Percentile","Median") , level='variable', inplace=True)
        res.rename(index=lambda x: x.replace("05th Percentile"," 5th Percentile") , level='variable', inplace=True)
        
    return res


# %% [markdown]
# # Process data

# %% [markdown]
# ### Antropogenic temperature

# %%
# Antropogenic temperature
folder = C3S_FOLDER / "anthropogenic_warming"

list_df = []
for file_path in folder.iterdir():

    if file_path.name.endswith("_timeseries.csv") or file_path.name.endswith("_rates.csv"):
        
        name = file_path.stem  # without .csv
        res = antropogenic(file_path=file_path,name=name,model=model,region=region,scenario=scenario)
        
        list_df.append(res)
        
        save_file_name = name.split(".")[0]
        save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"
        pyam.IamDataFrame(res).to_excel(save_file_name_path)

anthropogenic_warming_temperatures = pyam.concat(list_df)

# %% [markdown]
# ### Carbon Budget

# %%
# Carbon Budget
folder = C3S_FOLDER / "carbon_budget"

scenario_dict = {
    "budget_normal_magicc_True_fair_False_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.0_asym_False_hdT_1.24NonlinNonCO2_all_None_recEm213":
    "MAGICC 1.24",
    "budget_normal_magicc_True_fair_False_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.0_asym_False_hdT_1.26NonlinNonCO2_all_None_recEm213":
    "MAGICC 1.26",
    "budget_normal_magicc_True_fair_True_esf_7.1pm26.7_likeli_0.6827_nonCO2pc50.0_GtCO2_permaf_False_zecsd_0.19_asym_False_hdT_1.24NonlinNonCO2_all_None_recEm213":
    "MAGICC FaIR",
}
list_df = []
for file_path in folder.iterdir():
    if file_path.name.endswith(".csv"):
        
        name = file_path.stem  # without .csv
        print(name)
        
        df = pd.read_csv(file_path)
        # df.rename(columns={"dT_targets": "dT targets",'Future_warming':'Future Warming'}, inplace=True)
        df.rename(columns={"dT_targets": "dT targets"}, inplace=True)
        
        # df['unit'] = "°C"
        df['unit'] = "Gt CO2"
        df['model'] = model
        df['scenario'] = scenario_dict[name]
        df['region'] = region
        
        # indf=df.melt(id_vars=['Future Warming','dT targets','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(
        #     index=['model','scenario','region','variable','unit'], columns=['Future Warming','dT targets'], values='value'
        # )
        indf=df.melt(id_vars=['dT targets','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(
            index=['model','scenario','region','variable','unit'], columns='dT targets', values='value'
        )
        
        IAMC_from_C3S_df = update_index_levels_func(
            indf, {"variable": original2iamc}
        )
        
        res = IAMC_from_C3S_df.pix.format(
                variable="Carbon Budget|"+"{variable}",  # noqa: E501
                drop=True,
            )

        list_df.append(res)
        # Problemi time domains
        # carbon_budget = pyam.concat(list_df)
        
        # save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"
        # pyam.IamDataFrame(res).to_excel(save_file_name_path)
pd.concat(list_df)

# %% [markdown]
# ### Earth energy imbalance

# %%
# Earth energy imbalance
folder = C3S_FOLDER / "earth_energy_imbalance"

for file_path in folder.iterdir():
    if not file_path.name.endswith(".csv"):
        continue
    name = file_path.stem
    res = generic_csv2df(file_path,"ZJ",original2iamc,prepend="Earth Energy Imbalance|")
    
    ### DROPPING ERROR RIGHT NOW ###
    mask = res.index.get_level_values("variable").str.contains("Error")
    res = res[~mask]
    res.rename(index=lambda x: x.replace("Earth Energy Imbalance|Total","Earth Energy Imbalance") , level='variable', inplace=True)

    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"

    earth_energy_imbalance = pyam.IamDataFrame(res)
    earth_energy_imbalance.to_excel(save_file_name_path)

# %% [markdown]
# ### Effective Radiative Forcing

# %%
# Effective Radiative Forcing
folder = C3S_FOLDER / "effective_radiative_forcing"

scenario_map={"ERF_best_aggregates": "|Median",
             "ERF_p95_aggregates": "|95th Percentile",
             "ERF_p05_aggregates": "| 5th Percentile",
             "ERF_best": "|Median",}
          
for file_path in folder.iterdir():
    
    if not file_path.name.endswith(".csv"):
        continue
        
    name = file_path.stem
    print(name)
    if name.split("_")[-1] == "aggregates":
        res = generic_csv2df(file_path,"W/m^2",original2iamc_erf,prepend="Effective Radiative Forcing|")
    else:
        res = generic_csv2df(file_path,"W/m^2",original2iamc,prepend="Effective Radiative Forcing|")
    

    res = res.pix.format(
        variable="{variable}"+scenario_map[name],  # noqa: E501
        drop=True,
    )
    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

    effective_radiative_forcing_xlsx = pyam.IamDataFrame(res)
    effective_radiative_forcing_xlsx.to_excel(save_file_name_path)

    if name == "ERF_best_aggregates":
        res_out = res
        effective_radiative_forcing = effective_radiative_forcing_xlsx 

# %% [markdown]
# ### Extreme Temperatures

# %%
# Earth energy imbalance
folder = C3S_FOLDER / "extreme_temperatures"

list_df = []
for file_path in folder.iterdir():
    if not file_path.name.endswith(".csv"):
        continue
    name = file_path.stem
    print(name)
    
    data_type = name.split("_")[-1]
    
    res = generic_csv2df(file_path,"°C",original2iamc)

    if data_type == "timeseries":
        res = res.pix.format(
            scenario="{variable}",
            variable="Extreme Temperature",  # noqa: E501
            drop=True,
        )
    elif data_type == "means":
        res = res.pix.format(
            scenario="{variable}",  # noqa: E501
            variable="Extreme Temperature|10-Years Average",
            drop=True,
        )

    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

    pyam.IamDataFrame(res).to_excel(save_file_name_path)

    list_df.append(res)

extreme_temperatures = pyam.concat(list_df)

# %% [markdown]
# ### Global mean temperatures

# %%
# Global mean temperatures
folder = C3S_FOLDER / "global_mean_temperatures"

name2year_map = {"twenty_year_averages":"20-Years Average",
                "decadal_averages":"10-Years Average",
                "annual_averages":"1-Year Average",}
list_df = []

for file_path in folder.iterdir():

    if file_path.name.endswith(".csv"):
        name = file_path.stem  # without .csv
        print(name)
        col_name = ['timebound_lower', "timebound_upper","time"] if name == "annual_averages" else ["timebound_upper",'timebound_lower', "time"]
    
        df = pd.read_csv(file_path).drop(columns=[col_name[2],col_name[1]])
        
        df.rename(columns={col_name[0]: 'year'}, inplace=True)

        if name in ["decadal_averages","twenty_year_averages"]:
            df["year"] -= 1

        df['unit'] = "°C"
        df['model'] = model
        df['scenario'] = scenario
        df['region'] = region
        
        indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
        # indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
        
        IAMC_from_C3S_df = update_index_levels_func(
            indf, {"variable": original2iamc}
        )
        
        res = IAMC_from_C3S_df.pix.format(
                variable="{variable}",  # noqa: E501
                drop=True,
            )
        # 4 IAMS
        res4concat = IAMC_from_C3S_df.pix.format(
                variable="{variable}"+f"|Observed|{name2year_map[name]}",  # noqa: E501
                drop=True,
            )
        list_df.append(res4concat)
        global_mean_temperatures = pyam.concat(list_df)
        
        save_file_name = name.split(".")[0]
        save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"
        
        pyam.IamDataFrame(res).to_excel(save_file_name_path)

# %% [markdown]
# ### Global Land Precipitation

# %%
# Global Land Precipitation
folder = C3S_FOLDER / "global_land_precipitation"

for file_path in folder.iterdir():
    if not file_path.name.endswith(".csv"):
        continue
    name = file_path.stem
    print(name)
    
    res = generic_csv2df(file_path=file_path,unit="mm/year",mapper=original2iamc,prepend="")
    res = res.pix.format(
        scenario="{variable}",  # noqa: E501
        variable="Global Land Precipitation",
        drop=True,
    )
    #Adding mean 20260608 not needed anymore?
    # mean_row = res.mean(axis=0, numeric_only=True)
    # res.loc[("IGCC (2025)", "Reference", "World", "Global Land Precipitation", "mm/year")] = mean_row

    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

    global_land_precipitation = pyam.IamDataFrame(res)
    global_land_precipitation.to_excel(save_file_name_path)

# %% [markdown]
# ### GHG Emissions

# %%
# GHG Emissions
folder = C3S_FOLDER / "greenhouse_gas_emissions"
# model = "Human-Induced Greenhouse Gas Emissions"

for file_path in folder.iterdir():
    if file_path.name.endswith(".csv"):
        name = file_path.stem

        res = generic_csv2df(file_path=file_path,unit="Gt CO2-equiv",mapper=original2iamc,prepend="Human-Induced Emissions|",year_column_name="year")
        
        save_file_name = name.split(".")[0]
        save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

        ghg_emissions = pyam.IamDataFrame(res)
        ghg_emissions.to_excel(save_file_name_path)


# %% [markdown]
# ### GHG Concentration

# %%
# GHG Emissions
folder = C3S_FOLDER / "greenhouse_gas_concentrations"

for file_path in folder.iterdir():
    if not file_path.name.endswith(".csv"):
        continue
        
    name = file_path.stem
    print(name)
    
    res = generic_csv2df(file_path=file_path,unit="ppt",mapper=original2iamc,prepend="Concentration|")

    idx = res.index.to_frame(index=False)
    mask = idx["variable"].str.endswith("CO2")
    idx.loc[mask, "unit"] = "ppm"
    mask = idx["variable"].str.endswith("N2O") | idx["variable"].str.endswith("CH4")
    idx.loc[mask, "unit"] = "ppb"
    res.index = pd.MultiIndex.from_frame(idx)
    
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"

    greenhouse_gas_concentrations = pyam.IamDataFrame(res)
    greenhouse_gas_concentrations.to_excel(save_file_name_path)

# %% [markdown]
# ### Marine Heat Waves

# %%
# Marine Heat Waves
folder = C3S_FOLDER / "marine_heatwave_days"

for file_path in folder.iterdir():
    if not file_path.name.endswith(".csv"):
        continue
    name = file_path.stem
    print(name)
    
    res = generic_csv2df(file_path=file_path,unit="days",mapper=original2iamc,prepend="")
    res = res.pix.format(
        scenario="{variable}",  # noqa: E501
        variable="Marine Heat Waves",
        drop=True,
    )
    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

    marine_heatwave_days = pyam.IamDataFrame(res)
    marine_heatwave_days.to_excel(save_file_name_path)

# %% [markdown]
# ### Sea-level rise

# %%
# GHG Emissions
folder = C3S_FOLDER / "sea_level_rise"
# model = "Yearly Global Mean Sea Level Rise"

for file_path in folder.iterdir():
    if not file_path.name.endswith("GMSL_ensemble.csv"):
        continue
    name = file_path.stem
    print(name)
    
    res = generic_csv2df(file_path=file_path,unit="cm",mapper=original2iamc,prepend="Sea Level Rise|")
    
    # mm -> cm and set first mean value to 0 (as reference value)
    mask = res.index.get_level_values("variable").str.endswith("Mean")
    res[mask] = (res[mask].sub(res.iloc[:,0],axis=0))
    res = res * 0.1

    # METADATA
    meta = pd.DataFrame([[model, scenario]], columns=['model', 'scenario'])
    
    end_year_value = res.xs(key="Sea Level Rise|Mean", level="variable").loc[:,res.columns.max()].values
    for year in [1901,res.columns.max()-20,res.columns.max()-10]:
        start_year_value = res.xs(key="Sea Level Rise|Mean", level="variable").loc[:,year].values
        slr_since_year = end_year_value - start_year_value
        header = f"Sea Level Rise Since {year}"
        meta[header] = slr_since_year

    res = pyam.IamDataFrame(res)
    res.set_meta(meta)

    #Saving data
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{save_file_name}"+".xlsx"

    sea_level_rise = pyam.IamDataFrame(res)
    sea_level_rise.to_excel(save_file_name_path)

# %% [markdown]
# ### Final concat

# %%
list_df = [anthropogenic_warming_temperatures,
           earth_energy_imbalance,
           extreme_temperatures,
           effective_radiative_forcing,
           global_mean_temperatures,
           global_land_precipitation,
           ghg_emissions,
           greenhouse_gas_concentrations,
           marine_heatwave_days,
           sea_level_rise,
          ]
# list_df = [ghg_emissions,global_mean_temperatures,anthropogenic_warming_temperatures,sea_level_rise,]
pyam.concat(list_df).to_excel(C3S_FOLDER/ "IGCC_data.xlsx")

# %% [markdown]
# ## NetCDF

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
    
    df_aggr_best = pd.read_csv(
        folder/"ERF_best_aggregates.csv",
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )
    df_aggr_p05 = pd.read_csv(
        folder/"ERF_p05_aggregates.csv",
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )
    df_aggr_p95 = pd.read_csv(
        folder/"ERF_p95_aggregates.csv",
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )
    
    df_median = pd.merge(df,df_aggr_best, how="outer")
    
    mask = df_median.columns.str.startswith("time")
    time_df = df_median.iloc[:,mask]
    df_time_in_days = year2dates(time_df, REF_YEAR)
    data_df_median = df_median.iloc[:,~mask].copy()
    data_df_median["time"] = df_time_in_days["time"]
    
    mask = df_aggr_p05.columns.str.startswith("time")
    df_aggr_p05 = df_aggr_p05.iloc[:,~mask]
    df_aggr_p05["time"] = df_time_in_days["time"]
    
    
    mask = df_aggr_p95.columns.str.startswith("time")
    df_aggr_p95 = df_aggr_p95.iloc[:,~mask]
    df_aggr_p95["time"] = df_time_in_days["time"]

    datasets = []

    for label, df_data in [
        ("p05", df_aggr_p05),
        ("median", data_df_median),
        ("p95", df_aggr_p95),
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
    # ds["compound_emitted"].attrs.update({'long_name':'emitted compound', 'coverage_content_type': 'coordinate',
    #                                      'comment':'Species or emission category whose emissions produce the forcing response.'})
    # ds["forcing_agent"].attrs.update({'long_name':'forcing component', 'coverage_content_type': 'coordinate',
    #                                   'comment':'Component of effective radiative forcing attributable to the emitted compound.'})
    # ds["percentile"].attrs.update({'long_name':'percentile', 'units': "1", 'coverage_content_type': 'coordinate',
    #                                'comment':'Percentiles of the total effective radiative forcing distribution.'})
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

def process_anthropogenic(df: pd.DataFrame, df_time_in_days: pd.DataFrame)->xr.DataArray:
    
    percentiles = ["p05", "p17", "p50", "p83", "p95"]
    
    datasets = []
    
    for p in percentiles:
        
        mask = df.columns.str.endswith(p)
        data = df.iloc[:, mask].copy()
        data.columns = data.columns.map(lambda c: "_".join(c.split("_")[:-1]))
        data["time"] = df_time_in_days["time"]
        
        ds_tmp = xr.Dataset.from_dataframe(data.set_index("time"))
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

def process_headers(file_path:Path, general:dict, specific:dict, ref_year: int)->xr.DataArray:

    name = file_path.stem
    
    df_gmst = pd.read_csv(
        file_path,
        engine="python",
        comment="#",
        skip_blank_lines=True,
        na_values=[" ", "  ", "\t"],
    )
    
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

def process_carbon_budget(file_path:Path)->xr.array():

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

input_dir_setting = Path(processing.get("input_dir", "data"))

output_dir_setting = Path(processing.get("output_dir", "netcdf"))
if output_dir_setting.is_absolute():
    output_dir = output_dir_setting
else:
    output_dir = input_dir_setting.parent / output_dir_setting

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
            
        elif name in ["Gillett_GMST_timeseries","Ribes_GMST_timeseries","Walsh_GMST_timeseries","Gillett_GMST_rates","Ribes_GMST_rates","Walsh_GMST_rates"]:
            if name in ["Walsh_GMST_timeseries","Walsh_GMST_rates"]:
                df.rename(columns={"aerosol-radiation_interactions": "aerosol-radiation_interactions_p95"}, inplace=True)
                
            ds = process_anthropogenic(df, df_time_in_days)
            
        elif name == "altimetry_indiv_ensemble":
            # Processing: altimetry_indiv_ensemble and altimetry_ens
            ds = process_altimetry(file_path, df, df_time_in_days)

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

    # ds.to_netcdf(file.split(".")[0]+".nc")
    ds.to_netcdf(name + ".nc",encoding=encoding)

# %%
ds

# %%
file_dict["dimensions"]

# %%
