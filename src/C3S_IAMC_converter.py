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
from enum import StrEnum
import subprocess
import os

import pyam

## Set up

HERE = Path(__file__).parent
C3S_FOLDER = HERE.parent / "data/data/"
model = "IGCC (2025)"
scenario = "Reference"
region = "World"

## Download from https://github.com/ClimateIndicator/data

repo_url = "https://github.com/ClimateIndicator/data.git"
target_dir = HERE.parent / "data/"

def update_repo():
    if os.path.isdir(os.path.join(target_dir, ".git")):
        # Repo exists → pull updates
        subprocess.run(["git", "-C", target_dir, "pull"], check=True)
    else:
        # Repo doesn't exist → clone it
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
"""
Variables database
"""
VARIABLES = pd.DataFrame(
    [
        ("C2F6",   "PFC|C2F6",   "Concentration|PFC|C2F6",),
        ("C3F8",   "PFC|C3F8",   "Concentration|PFC|C3F8",),
        ("C8F18",  "PFC|C8F18",  "Concentration|PFC|C8F18",),
        ("CF4",    "PFC|CF4",    "Concentration|PFC|CF4",),
        ("c-C4F8", "PFC|cC4F8",  "Concentration|PFC|cC4F8",),
        ("n-C4F10","PFC|nC4F10", "Concentration|PFC|nC4F10",),
        ("n-C5F12","PFC|nC5F12", "Concentration|PFC|nC5F12",),
        ("n-C6F14","PFC|nC6F14", "Concentration|PFC|nC6F14",),
        ("i-C6F14","PFC|iC6F14", "Concentration|PFC|iC6F14",),
        ("C7F16",  "PFC|C7F16",  "Concentration|PFC|C7F16",),
        #
        ("Halon-1211",  "Halon|1211","Concentration|Halon|1211",),
        ("Halon-1301",  "Halon|1301","Concentration|Halon|1301",),
        ("Halon-2402",  "Halon|2402","Concentration|Halon|2402",),
        #
        ("CH4",      "CH4",         "Concentration|CH4",),
        ("N2O",      "N2O",         "Concentration|N2O",),
        ("CO2",      "CO2",         "Concentration|CO2",),
        ("SF6",      "SF6",         "Concentration|SF6",),
        ("SO2F2",    "SO2F2",       "Concentration|SO2F2",),
        ("NF3",      "NF3",         "Concentration|NF3",),
        #
        ("HFC-134a", "HFC|HFC134a",    "Concentration|HFC|HFC134a"),
        ("HFC-23",   "HFC|HFC23",      "Concentration|HFC|HFC23",),
        ("HFC-32",   "HFC|HFC32",      "Concentration|HFC|HFC32"),
        ("HFC-125",   "HFC|HFC125",    "Concentration|HFC|HFC125",),
        ("HFC-143a", "HFC|HFC143a",    "Concentration|HFC|HFC143a",),
        ("HFC-152a", "HFC|HFC152a",    "Concentration|HFC|HFC152a"),
        ("HFC-227ea","HFC|HFC227ea",   "Concentration|HFC|HFC227ea",),
        ("HFC-236fa","HFC|HFC236fa",   "Concentration|HFC|HFC236fa"),
        ("HFC-245fa","HFC|HFC245fa",   "Concentration|HFC|HFC245fa",),
        ("HFC-365mfc","HFC|HFC365mfc", "Concentration|HFC|HFC365mfc",),
        ("HFC-43-10mee","HFC|HFC43-10","Concentration|HFC|HFC43-10"),
        #
        ("CFC-11",  "CFC11",   "Concentration|CFC11"),
        ("CFC-12",  "CFC12",   "Concentration|CFC12"),
        ("CFC-13",  "CFC13",   "Concentration|CFC13"),
        ("CFC-112", "CFC112",  "Concentration|CFC112"),
        ("CFC-112a","CFC112a", "Concentration|CFC112a"),
        ("CFC-113", "CFC113",  "Concentration|CFC113"),
        ("CFC-113a","CFC113a", "Concentration|CFC113a"),
        ("CFC-114", "CFC114",  "Concentration|CFC114"),
        ("CFC-114a","CFC114a", "Concentration|CFC114a"),
        ("CFC-115", "CFC115",  "Concentration|CFC115"),
        #
        ("CH3CCl3","CH3CCl3","Concentration|CH3CCl3",),
        ("CCl4",   "CCl4",   "Concentration|CCl4",),
        ("CH3Cl",  "CH3Cl",  "Concentration|CH3Cl",),
        ("CH3Br",  "CH3Br",  "Concentration|CH3Br",),
        ("CH2Cl2", "CH2Cl2", "Concentration|CH2Cl2",),
        ("CHCl3",  "CHCl3",  "Concentration|CHCl3",),
        #
        ("HCFC-22",  "HCFC22",  "Concentration|HCFC22",),
        ("HCFC-141b","HCFC141b","Concentration|HCFC141b",),
        ("HCFC-142b","HCFC142b","Concentration|HCFC142b",),
        #
        ("HCFC-133a","HCFC133a","Concentration|HCFC133a",),
        ("HCFC-31",  "HCFC31",  "Concentration|HCFC31",),
        ("HCFC-124","HCFC124",  "Concentration|HCFC124",),
        #
        ("CO2-FFI",    "CO2|FFI",  None,),
        ("CO2-LULUCF", "CO2|LULUCF", None,),
        ("F-gases",    "F-gases", None,),
        #
        ("volcanic","Raw Surface Temperature (GMST)|Volcanic",None),
        ("solar","Raw Surface Temperature (GMST)|Solar",None),
        ("o3",                 "Raw Surface Temperature (GMST)|O3",None),
        ("n2o",                "Raw Surface Temperature (GMST)|N2O",None),
        ("land_use",           "Raw Surface Temperature (GMST)|Land Use",None),
        ("halogen",            "Raw Surface Temperature (GMST)|Halogen",None),
        ("h2o",                "Raw Surface Temperature (GMST)|H2O",None),
        ("h2o_strat",          "Raw Surface Temperature (GMST)|H2O-Strat",None),
        ("contrails",          "Raw Surface Temperature (GMST)|Contrails",None),
        ("co2",                "Raw Surface Temperature (GMST)|CO2",None),
        ("ch4",                "Raw Surface Temperature (GMST)|CH4",None),
        ("bc_snow",            "Raw Surface Temperature (GMST)|BC Snow",None),
        ("aerosol-radiation_interactions","Raw Surface Temperature (GMST)|Aerosol-Radiation Interactions",None),
        ("aerosol-cloud_interactions",    "Raw Surface Temperature (GMST)|Aerosol-Cloud Interactions",None),
        ("residual",           "Raw Surface Temperature (GMST)|Residual",None),
        ("total",              "Raw Surface Temperature (GMST)|Total",None),
        ("anthropogenic",      "Raw Surface Temperature (GMST)|Anthropogenic", None),
        ("ghg",                "Raw Surface Temperature (GMST)|GHGs",    None),
        ("natural",            "Raw Surface Temperature (GMST)|Natural", None),
        ("other_human_forcing","Raw Surface Temperature (GMST)|Other Human Forcings",None),
        #
        ("GMST","Raw Surface Temperature (GMST)",None),
    ],
    columns=["original", "iamc", "iamc_concentration"],
)
"""
Database of emissions variables names according to different naming schemes
"""

# Raw Surface Temperature (GMST)	K	Raw global-mean air ocean blended temperature change (GMST i.e. a blend of 2m air temperature over land and surface temperatures over the ocean; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)	K	Raw global-mean surface air temperature change (GSAT i.e. 2m air temperature; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|CO2	K	Raw global-mean surface air temperature change due to CO2 (note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|Non-CO2	K	Raw global-mean surface air temperature change due to non-CO2 climate drivers (note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Raw Surface Temperature (GSAT)|Residual	K	Raw Surface Temperature (GSAT) - Raw Surface Temperature (GSAT)|CO2 - Raw Surface Temperature (GSAT)|Non-CO2 i.e. the difference between the raw global-mean surface air temperature change and the sum of CO2 and non-CO2 contributions; the majority of this residual is due to natural climate forcers (e.g. volcanic eruptions and variations in the solar cycle; note that the breakdown of total warming into components is only approximate because of non-linearities and feedbacks; raw to distinguish it from the GSAT output which is adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)
# Surface Temperature (GSAT)	K	Global-mean surface air temperature change to be used for e.g. scenario categorisation (this output has been adjusted to match the WG1 best-estimate historical warming between 1850-1900 and 1995-2014)

"""
Variables database
"""
SLR_VARIABLES = pd.DataFrame(
    [
        ("mean",   "Sea Level Rise|Mean",),
        ("std",    "Sea Level Rise|Standard Deviation",),
    ],
    columns=["original", "sea_level_rise",],
)
"""
Database of emissions variables names according to different naming schemes for sea level rise
"""


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

    IAMC_CONCENTRATION = "iamc_concentration"

    C3S = "c3s"
    """
    Copernicus C3S format
    """

    SLR = "sea_level_rise"
    
original2iamc = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.ORIGINAL,
    to_convention=SupportedNamingConventions.IAMC,
    database=VARIABLES,
)
original2iamc_concentration = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.ORIGINAL,
    to_convention=SupportedNamingConventions.IAMC_CONCENTRATION,
    database=VARIABLES,
)
original2slr = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.ORIGINAL,
    to_convention=SupportedNamingConventions.SLR,
    database=SLR_VARIABLES,
)


# ## Functions

def antropogenic(file_path,name,model,region,scenario):

    # DIFFERENT col name for Walsh_GMST_timeseries.csv ?? 
    col_name = ['timebounds_lower', "timebounds_upper"] if name == "Walsh_GMST_timeseries" else ['timebound_lower', "timebound_upper"]

    df = pd.read_csv(file_path).drop(columns=["time",col_name[1]], errors="ignore")
    
    df.rename(columns={col_name[0]: 'year'}, inplace=True)
    df['unit'] = 'K'#"°C"
    df['model'] = model
    df['scenario'] = scenario
    df['region'] = region
    if name in ["Walsh_GMST_timeseries","Walsh_GMST_rates"]:
        df.rename(columns={"aerosol-radiation_interactions": "aerosol-radiation_interactions_p95"}, inplace=True)
    
    indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
    indf['percentile'] = indf.index.get_level_values('variable').str.split('_').str[-1].str[-2:]
    indf.set_index('percentile', append=True, inplace=True)
    indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
    indf.rename(index=lambda x: x+"th" if x[-1]!="3" else x+"rd", level='percentile', inplace=True)

    IAMC_from_C3S_df = update_index_levels_func(
        indf, {"variable": original2iamc}
    )
    
    res = IAMC_from_C3S_df.pix.format(
        variable="{variable}|{percentile} Percentile",  # noqa: E501
        drop=True,
    )
    # res.rename(index=lambda x: x.replace("50th Percentile","Median") , level='variable', inplace=True)
    save_file_name = name.split(".")[0]
    save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"
    
    pyam.IamDataFrame(res).to_excel(save_file_name_path)


# ## Process data

def main():
    # Update data if needed
    update_repo()
    # Antropogenic temperature
    folder = C3S_FOLDER / "anthropogenic_warming"

    for file_path in folder.iterdir():

        if file_path.name.endswith("_timeseries.csv") or file_path.name.endswith("_rates.csv"):
            name = file_path.stem  # without .csv
            antropogenic(file_path=file_path,name=name,model=model,region=region,scenario=scenario)


    # Global mean temperatures
    folder = C3S_FOLDER / "global_mean_temperatures"

    for file_path in folder.iterdir():

        if file_path.name.endswith(".csv"):
            name = file_path.stem  # without .csv
            print(name)
            col_name = ['timebound_lower', "timebound_upper","time"] if name == "annual_averages" else ["time",'timebound_lower', "timebound_upper"]
        
            df = pd.read_csv(file_path).drop(columns=[col_name[2],col_name[1]])
            
            df.rename(columns={col_name[0]: 'year'}, inplace=True)

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
            
            save_file_name = name.split(".")[0]
            save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"
            
            pyam.IamDataFrame(res).to_excel(save_file_name_path)

    # GHG Emissions
    folder = C3S_FOLDER / "greenhouse_gas_emissions"

    for file_path in folder.iterdir():
        if file_path.name.endswith(".csv"):
            name = file_path.stem
            col_name = ['timebound_lower', "timebound_upper","year"]
        
            df = pd.read_csv(file_path).drop(columns=[col_name[2],col_name[1]])
            
            df.rename(columns={col_name[0]: 'year'}, inplace=True)
            # df['unit'] = 'MtCO2/yr'
            df['unit'] = 'Gt CO2-equiv'
            df['model'] = model
            df['scenario'] = scenario
            df['region'] = region
            
            indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
            # indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
            
            
            IAMC_from_C3S_df = update_index_levels_func(
                indf, {"variable": original2iamc}
            )
            
            res = IAMC_from_C3S_df.pix.format(
                variable="Human-Induced Emissions|{variable}",  # noqa: E501
                drop=True,
            )
            res.rename(index=lambda x: x.replace("FFI","Fossil Fuels and Industry").replace("F-gases", "F-Gases") , level='variable', inplace=True)
            res = res*1000
            
            save_file_name = name.split(".")[0]
            save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"

            ghg_emissions = pyam.IamDataFrame(res)
            ghg_emissions.to_excel(save_file_name_path)

    # GHG Emissions
    folder = C3S_FOLDER / "greenhouse_gas_concentrations"

    for file_path in folder.iterdir():
        if not file_path.name.endswith(".csv"):
            continue
            
        name = file_path.stem
        col_name = ['timebound_lower', "timebound_upper","time"]
        df = pd.read_csv(file_path).drop(columns=[col_name[2],col_name[1]])
        
        df.rename(columns={col_name[0]: 'year'}, inplace=True)
        df['unit'] = 'aaaaa / yr'
        df['model'] = model
        df['scenario'] = scenario
        df['region'] = region
        
        indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
        # indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
        
        IAMC_from_C3S_df = update_index_levels_func(
            indf, {"variable": original2iamc_concentration}
        )
        
        res = IAMC_from_C3S_df.pix.format(
            # variable="IGCC (2025)|{variable}",  # noqa: E501
            variable="{variable}",
            drop=True,
        )
        
        save_file_name = name.split(".")[0]
        save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"
        
        pyam.IamDataFrame(res).to_excel(save_file_name_path)

    ### Sea-level rise

    folder = C3S_FOLDER / "sea_level_rise"

    for file_path in folder.iterdir():
        if not file_path.name.endswith("GMSL_ensemble.csv"):
            continue
            
        name = file_path.stem
        col_name = ['timebound_lower', "timebound_upper","time"]
        df = pd.read_csv(file_path).drop(columns=[col_name[2],col_name[1]])
        
        df.rename(columns={col_name[0]: 'year'}, inplace=True)
        df['unit'] = 'cm'
        df['model'] = model
        df['scenario'] = scenario
        df['region'] = region
        
        indf=df.melt(id_vars=['year','unit','model','scenario',"region"], var_name='variable', value_name='value').pivot_table(index=['model','scenario','region','variable','unit'], columns='year', values='value')
        # indf.rename(index=lambda x: "_".join(x.split("_")[:-1]), level='variable', inplace=True)
        
        IAMC_from_C3S_df = update_index_levels_func(
            indf, {"variable": original2slr}
        )
        
        res = IAMC_from_C3S_df.pix.format(
            variable="{variable}",  # noqa: E501
            drop=True,
        )
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
        save_file_name_path = os.path.dirname(file_path) +"/"+ f"{name}"+".xlsx"

        sea_level_rise = pyam.IamDataFrame(res)
        sea_level_rise.to_excel(save_file_name_path)

    ### Final concat
    list_df = [sea_level_rise,ghg_emissions]
    pyam.concat(list_df).to_excel(C3S_FOLDER/ "IGCC_data.xlsx")

if __name__ == "__main__":
    main()
