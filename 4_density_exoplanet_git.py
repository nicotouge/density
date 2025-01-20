# %%
#!/usr/bin/env python
# coding: utf-8
# (c) Charles Le Losq, Nicolas Tougeron
# Code modified after Le Losq et al. 2024
# see embedded licence file
# we use pathlib to handle paths

## Portabilité du code (à changer !!!)
from pathlib import Path

# Définir le chemin racine #Chemin à changer
racine = Path(r"C:\Users\")

## Library loading and additional function definition
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
import xgboost as xgb
from sklearn.base import BaseEstimator
import copy
from mapie.regression import MapieRegressor

# dataframes and arrays
import pandas as pd 
import numpy as np
import scipy

# machine learning
from sklearn.neighbors import KNeighborsRegressor

# error propagation
from uncertainties import unumpy

# for plotting
import matplotlib.pyplot as plt 

# For calculating surface temperature
# using the Carter et al. 2024 code
import carter

# import our code for helper functions
# you can install it with the traditional 'pip install gpvisc' command
import gpvisc

# import astropy
import astropy.units as u
from scipy.constants import Stefan_Boltzmann as sigma_sb

# we use pathlib to handle paths
from pathlib import Path
# we get the current path
_BASEPATH = Path(Path.cwd())

#loading XGBoost model trained in code 2
with open(racine / 'ANN_model.pkl', 'rb') as file:
    ANN_model = pickle.load(file)
    
#loading Mapie model trained in code 2
with open(racine / 'mapie_ANN_model.pkl', 'rb') as file:
    mapie_model = pickle.load(file)


###############################################################
# TEMPERATURE AT THE SURFACE OF A PLANET IRRADIATED BY A STAR #
###############################################################
def calculate_T_surface_carter(
        
        lon = np.arange(-180, 180, 0.1),
        planet_distance_AU = 0.00716,
        planet_radius_earthUnit = 1.51,
        star_radius_solUnit = 0.681,
        temperature_star_K = 4599.0,
        temperature_nightside_K = 950.0,
        show_figure = True):
    """calculate temperature profile at surface of an exoplanet
    

    Notes
    =====
    The calculation takes into account of the angular size of the star

    Default parameters are for K2-141 b from Malavolta et al. 2018

    This code uses the model and code from Carter 2024
    """
    
    # convert in si
    star_radius_si = (star_radius_solUnit  * u.solRad).si.value
    planet_radius_si = (planet_radius_earthUnit * u.earthRad).si.value
    planet_distance_si = (planet_distance_AU  * u.AU).si.value

    angular_diameter_rad = 2.0*np.arctan((2*star_radius_si)/(2*planet_distance_si))
    angular_diameter_degree = np.rad2deg(angular_diameter_rad)
    print("Star angular size: {}".format(angular_diameter_degree))
    print("Flux at surface assuming parallel beam:")
    print((star_radius_si/planet_distance_si)**2*sigma_sb*temperature_star_K**4)

    # calculate the limits of the different regions
    # formula 1 from Carter et al. 2024
    n_full = np.arcsin((star_radius_si + planet_radius_si)/planet_distance_si)
    n_pen = np.arcsin(planet_radius_si/planet_distance_si)
    n_un = np.arcsin((star_radius_si - planet_radius_si)/planet_distance_si)

    # star intensity 
    Io = sigma_sb * temperature_star_K**4

    # start calculations as in code Carter et al. 2024
    aRs, RpRs = carter.getScaled(planet_distance_AU, planet_radius_earthUnit, star_radius_solUnit)

    # star-planet seperation in units of Rp
    sep_unitsRp = aRs / RpRs

    # stellar radius in units of Rp
    Rs_unitsRp = 1.0 / RpRs
    print(Rs_unitsRp)

    # Ls in units of Rp
    # beware of this!
    Ls = np.pi * Rs_unitsRp**2 * Io

    # coordinates of center of source in units of Rp
    xs = 0
    ys = 0
    zs = sep_unitsRp

    # planet coordinates
    lat = np.zeros(len(lon))

    # get intensities (the interesting one is Ifinite)
    Ifull_divided_by_Ls= carter.getIfull(lat, lon, xs, ys, zs, Rs_unitsRp)
    Ifinite_divided_by_Ls = carter.getIfinite2(lat, lon, xs, ys, zs, Rs_unitsRp, 200)
    Iplane_divided_by_Ls = carter.getIplane(lat, lon, xs, ys, zs)

    # we renormalize the flux
    # as the intensity given by getIfinite2()
    # does not always match well the analytical
    # calculated intensity given by getIfull()
    Ifinite_divided_by_Ls = Ifinite_divided_by_Ls/np.max(Ifinite_divided_by_Ls) * np.max(np.nan_to_num(Ifull_divided_by_Ls))

    # flux in W / m^2
    Fp_stellar_W_m2 = Ifinite_divided_by_Ls.copy() * Ls

    # geothermal flux in W / m^2
    Fp_geoterm_W_m2 = sigma_sb * temperature_nightside_K**4

    # Total flux at planet surface
    Fp = Fp_stellar_W_m2 + Fp_geoterm_W_m2

    # temperature calculation
    T_planet = (Fp/sigma_sb)**(1/4.)

    if show_figure == True:
        # figure
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(lon, Ifull_divided_by_Ls*1e6, "k-", label="full")
        plt.plot(lon, Ifinite_divided_by_Ls*1e6, "r--", label="finite")
        plt.plot(lon, Iplane_divided_by_Ls*1e6, "b:", label="plane")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(lon, T_planet)

    print("Theta full (from substellar point): {:.1f}".format(90-np.rad2deg(n_full)))
    print("Theta pen (from substellar point): {:.1f}".format(90-np.rad2deg(n_pen)))
    print("Theta night (from substellar point): {:.1f}".format(90+np.rad2deg(n_un)))

    return T_planet

def whittaker(y,**kwargs):
    """smooth a signal with the Whittaker smoother

    Parameters
    ----------
    y : ndarray
        An array with the values to smooth (equally spaced).
    Lambda : float, optional
        The smoothing coefficient, the higher the smoother. Default = 10^5.

    Returns
    -------
    z : ndarray
        An array containing the smoothed values.

    References
    ----------
    P. H. C. Eilers, A Perfect Smoother. Anal. Chem. 75, 3631–3636 (2003).

    """
    # optional parameters
    lam = kwargs.get('Lambda',1.0*10**5)

    # starting the algorithm
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w*y)

    return z

def fill_un_pen_full_regions(y_min, y_max):
    """shaded areas for night, pen, day"""
    plt.fill_betweenx([y_min,y_max],-115,-89, color="grey", alpha=0.15, edgecolor=None)
    plt.fill_betweenx([y_min,y_max],89,116, color="grey", alpha=0.15, edgecolor=None)

    plt.fill_betweenx([y_min,y_max],-115,-180, color="grey", alpha=0.3, edgecolor=None)
    plt.fill_betweenx([y_min,y_max],116,180, color="grey", alpha=0.3, edgecolor=None)

###
# Temperature profile on K2-141 b 
# See paper for planetary parameters
###

lon = np.arange(-180, 180, 0.1)
T_planet = calculate_T_surface_carter(
    lon = lon,
    planet_distance_AU = 0.00716,
    planet_radius_earthUnit = 1.51,
    star_radius_solUnit = 0.681,
    temperature_star_K = 4599.0,
    temperature_nightside_K = 950.0,
    )

print("Maximum temperature is: {:.0f}".format(np.max(T_planet)))

###
# Figure for temperature
###

plt.figure(figsize=(4.22,3.22), dpi=150)

plt.errorbar(np.array([0., -135, 135]),
             np.array([2049, 956, 956]), 
             yerr=np.array([[359, 556, 556],[362, 489, 489]]),
             marker="s", 
             linestyle="none",
             color="k", label="Zieba et al. 2022")

plt.errorbar(0,3038,
             yerr=64,marker="o", mfc="none", 
             c = "black", linestyle="none", 
             label="Malavolta et al. 2018")

plt.legend(loc="upper right", fontsize=8)#bbox_to_anchor=(1.65, 1.0))

# plot our temperature profile
plt.plot(lon, T_planet, "-", linewidth=1.0, color="purple")

# shaded areas for night, pen, day
plt.fill_betweenx([500,5000],-115,-89, color="grey", alpha=0.15)
plt.fill_betweenx([500,5000],89,115, color="grey", alpha=0.15)

plt.fill_betweenx([500,5000],-116,-180, color="grey", alpha=0.4)
plt.fill_betweenx([500,5000],116,180, color="grey", alpha=0.4)

plt.xlim(-180, 180)
plt.ylim(500,4000)

plt.annotate("Nightside", xy=(-170, 3900), rotation=90, va="top", fontstyle="italic")
plt.annotate("Dayside", xy=(-80, 3900), rotation=90, va="top", fontstyle="italic")
plt.annotate("Penumbra", xy=(-110, 3900), rotation=90, va="top", fontstyle="italic")

plt.ylabel("Temperature, K")
plt.xlabel("Longitude, 0° = substellar point")

plt.tight_layout()
plt.savefig(_BASEPATH / "LeLosqetal_2024_K2141b_temperature_profile.pdf", bbox_inches="tight")

# %% [markdown]
# # Compositions, 1 bar atmosphere
# 
# The function below is for density.
# 
# Warning: you need to take into account of the crystals for density !
# 
# I modified the input file to have the proportions of the different phases, you will find the following new columns:
# 
# - prct_opx : percentage of orthopyroxene 
# - prct_cpx : percentage of clinopyroxene 
# - prct_ol : percentage of olivine 
# - prct_an : percentage of anorthite 
# - prct_sp : percentage of spinel 
# - prct_mel : percentage of melilite 

## Density of minerals (g/cm3)
d_opx=3.20
d_cpx=3.4
d_ol=3.3
d_an=2.75
d_sp=3.6
d_mel=2.95

## Density profile function 
def generate_density_profile(db, T_planet):
        """generate phase and viscosity longitudinal profile at the
        surface of a planet given composition (db), longitude (lon) and temperature profile (T_planet, K)
        
        db : pandas dataframe
                composition in weight percent
        T_planet : 1d numpy array containing the temperature profile as a function of lon
        """

        db_mol = gpvisc.wt_mol(gpvisc.chimie_control(db))
        db_mol["prct_c"] = 100 - db_mol["prct_l"]- db_mol["prct_g"] 

        ##############
        # KNR INTERP #
        ##############
        # we use a nearest neighbor algo to link temperature to all other variables
        clf = KNeighborsRegressor(n_neighbors=1, weights='distance')
        clf.fit(db_mol.loc[:, "T_K"].values.reshape(-1,1), 
                db_mol.loc[:, ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5', 'h2o',"prct_cpx","prct_opx","prct_ol",'prct_an', "prct_g","prct_l","prct_c"]].values)

        ##########
        # FIGURE #
        ##########
        # plot with an inverse X scale T_K versus prcl_l in compo_1bar
        plt.figure(figsize=(6,8/9*6))

        plt.plot(db_mol["T_K"], db_mol["prct_c"], "s", color="green", label="1 bar")
        plt.plot(db_mol["T_K"], db_mol["prct_l"], "o", color="blue", label="1 bar")
        plt.plot(db_mol["T_K"], db_mol["prct_g"], "d", color="purple", label="1 bar")

        # visualize the ML model
        fake_T = np.arange(1000,4000,5.0)
        plt.plot(fake_T, clf.predict(fake_T.reshape(-1,1))[:,18], "-", color="green")
        plt.plot(fake_T, clf.predict(fake_T.reshape(-1,1))[:,17], "-", color="blue")
        plt.plot(fake_T, clf.predict(fake_T.reshape(-1,1))[:,16], "-", color="purple")

        plt.annotate("liquid", xy=[2500,94], color="blue")
        plt.annotate("gas", xy=[3400,90], color="purple")
        plt.annotate("crystals", xy=[2000,20], color="green")

        # add legend
        plt.xlabel("Temperature (K)")
        plt.ylabel("Phase fraction")
        plt.tight_layout()
        plt.show()

        ########################
        # CALCULATE FOR PLANET #
        ########################

        # get the composition via the KNR interpolator
        values_planet = clf.predict(T_planet.reshape(-1,1))

        # declaring the output dataframe
        planet_result = pd.DataFrame(values_planet, columns= ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5', 'h2o',"prct_cpx","prct_opx","prct_ol",'prct_an', "prct_g","prct_l","prct_c"]) #à modifier #,"prct_sp","prct_mel"
        
        elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5','h2o']
        
        X_planet=planet_result.loc[:,elements_]
        
        planet_result["D_liquid_planet"]=ANN_model.predict(X_planet)
        
        
        error=mapie_model.predict(X_planet, alpha=0.95)
        planet_result["D_liquid_planet_pred"]=error[0]
        planet_result["D_liquid_planet_low"]=error[1][:, 0]
        planet_result["D_liquid_planet_up"]=error[1][:, 1]
        
        
        planet_result["D"]=(planet_result["prct_cpx"]*d_cpx+planet_result["prct_opx"]*d_opx+planet_result["prct_an"]*d_an+planet_result["prct_ol"]*d_ol+planet_result["prct_l"]*planet_result["D_liquid_planet"])/100
        planet_result["D_low"]=(planet_result["prct_cpx"]*d_cpx+planet_result["prct_opx"]*d_opx+planet_result["prct_an"]*d_an+planet_result["prct_ol"]*d_ol+planet_result["prct_l"]*(planet_result["D_liquid_planet_low"]))/100
        planet_result["D_up"]=(planet_result["prct_cpx"]*d_cpx+planet_result["prct_opx"]*d_opx+planet_result["prct_an"]*d_an+planet_result["prct_ol"]*d_ol+planet_result["prct_l"]*(planet_result["D_liquid_planet_up"]))/100
        
        
        return planet_result

def generate_density_profile1(db, T_planet):
        """generate phase and viscosity longitudinal profile at the
        surface of a planet given composition (db), longitude (lon) and temperature profile (T_planet, K)
        
        db : pandas dataframe
                composition in weight percent
        T_planet : 1d numpy array containing the temperature profile as a function of lon
        """

        db_mol = gpvisc.wt_mol(gpvisc.chimie_control(db))
        db_mol["prct_c"] = 100 - db_mol["prct_l"]- db_mol["prct_g"] 

        ##############
        # KNR INTERP #
        ##############
        # we use a nearest neighbor algo to link temperature to all other variables
        clf1 = KNeighborsRegressor(n_neighbors=1, weights='distance')
        clf1.fit(db_mol.loc[:, "T_K"].values.reshape(-1,1), 
                db_mol.loc[:, ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5','h2o',"prct_sp","prct_mel",'prct_an', "prct_g","prct_l","prct_c"]].values)

        ##########
        # FIGURE #
        ##########
        # plot with an inverse X scale T_K versus prcl_l in compo_1bar
        plt.figure(figsize=(6,8/9*6))

        plt.plot(db_mol["T_K"], db_mol["prct_c"], "s", color="green", label="1 bar")
        plt.plot(db_mol["T_K"], db_mol["prct_l"], "o", color="blue", label="1 bar")
        plt.plot(db_mol["T_K"], db_mol["prct_g"], "d", color="purple", label="1 bar")

        # visualize the ML model
        fake_T = np.arange(1000,4000,5.0)
        plt.plot(fake_T, clf1.predict(fake_T.reshape(-1,1))[:,17], "-", color="green")
        plt.plot(fake_T, clf1.predict(fake_T.reshape(-1,1))[:,16], "-", color="blue")
        plt.plot(fake_T, clf1.predict(fake_T.reshape(-1,1))[:,15], "-", color="purple")

        plt.annotate("liquid", xy=[2500,94], color="blue")
        plt.annotate("gas", xy=[3400,90], color="purple")
        plt.annotate("crystals", xy=[2000,20], color="green")

        # add legend
        plt.xlabel("Temperature (K)")
        plt.ylabel("Phase fraction")
        plt.tight_layout()
        plt.show()

        ########################
        # CALCULATE FOR PLANET #
        ########################

        # get the composition via the KNR interpolator
        values_planet = clf1.predict(T_planet.reshape(-1,1))
        
        #density_planet=xgboost_model.predict(values_planet)

        # declaring the output dataframe
        planet_result = pd.DataFrame(values_planet, columns= ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5', 'h2o',"prct_sp","prct_mel",'prct_an', "prct_g","prct_l","prct_c"]) #à modifier #,"prct_sp","prct_mel"
        
        elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3','mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5','h2o']
        
        X_planet=planet_result.loc[:,elements_]
        
        planet_result["D_liquid_planet"]=ANN_model.predict(X_planet)
        error=mapie_model.predict(X_planet, alpha=0.95)
        planet_result["D_liquid_planet_pred"]=error[0]
        planet_result["D_liquid_planet_low"]=error[1][:, 0]
        planet_result["D_liquid_planet_up"]=error[1][:, 1]


        planet_result["D"]=(planet_result["prct_sp"]*d_sp+planet_result["prct_mel"]*d_mel+planet_result["prct_an"]*d_an+planet_result["prct_l"]*planet_result["D_liquid_planet"])/100
        planet_result["D_low"]=(planet_result["prct_sp"]*d_sp+planet_result["prct_mel"]*d_mel+planet_result["prct_an"]*d_an+planet_result["prct_l"]*(planet_result["D_liquid_planet_low"]))/100
        planet_result["D_up"]=(planet_result["prct_sp"]*d_sp+planet_result["prct_mel"]*d_mel+planet_result["prct_an"]*d_an+planet_result["prct_l"]*(planet_result["D_liquid_planet_up"]))/100


        return planet_result

##########################
# VISCOSITY CALCULATIONS #
##########################

# load compositions
BSE_1bar_in = pd.read_excel(_BASEPATH / "Lelosq_et_al_2024_condensation_sequence.xlsx", sheet_name="BSE_1bar")
FBSE_1bar_in = pd.read_excel(_BASEPATH / "Lelosq_et_al_2024_condensation_sequence.xlsx", sheet_name="FBSE_1bar")
CAI_1bar_in = pd.read_excel(_BASEPATH / "Lelosq_et_al_2024_condensation_sequence.xlsx", sheet_name="CAI_1bar")

# return results
Result_BSE = generate_density_profile(BSE_1bar_in, T_planet)
Result_FBSE = generate_density_profile(FBSE_1bar_in, T_planet)
Result_CAI = generate_density_profile1(CAI_1bar_in, T_planet)


# %% [markdown]
# ### Code for zooms
# 
# from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html

# %%
from matplotlib.transforms import (Bbox, TransformedBbox,
                                   blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.1,
            "clip_on": False,
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    ax1
        The main axes.
    ax2
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

    prop_patches = {**kwargs, "alpha": 1.}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

# %%
plt.figure(figsize=(6.22,3.22))
ax1 = plt.subplot(1,2,1)
plt.plot(lon, Result_BSE.loc[:,"prct_c"], "-", linewidth=0.5)
plt.plot(lon, Result_BSE.loc[:,"prct_l"], "-", linewidth=0.5)
#plt.plot(Longitude_axis, Result_BSE.loc[:,"prct_g"], "-", linewidth=0.5)

plt.plot(lon, Result_FBSE.loc[:,"prct_c"], "-", linewidth=0.5, color="C0")
plt.plot(lon, Result_FBSE.loc[:,"prct_l"], "-", linewidth=0.5, color="C1")
#plt.plot(lon, Result_FBSE.loc[:,"prct_g"], "-", linewidth=0.5, color="C2")

plt.plot(lon, Result_CAI.loc[:,"prct_c"], "-", linewidth=0.5, color="C0")
plt.plot(lon, Result_CAI.loc[:,"prct_l"], "-", linewidth=0.5, color="C1")
plt.xlabel("Longitude, 0° = substellar point")
plt.ylabel("Fraction of the phase")
plt.annotate("Crystals", xy=(-180, 94), xycoords="data", color="C0", ha="left")
plt.annotate("Melt", xy=(0, 92), xycoords="data", color="C1", ha="center")
#plt.annotate("Gas", xy=(0, 3), xycoords="data", color="C2", ha="center")

ax2 = plt.subplot(1,2,2)
plt.plot(lon, Result_BSE.loc[:,"prct_c"], "-", linewidth=0.5)
plt.plot(lon, Result_BSE.loc[:,"prct_l"], "-", linewidth=0.5)
#plt.plot(Longitude_axis, Result_BSE.loc[:,"prct_g"], "-", linewidth=0.5)

plt.plot(lon, Result_FBSE.loc[:,"prct_c"], "-.", linewidth=0.5, color="C0")
plt.plot(lon, Result_FBSE.loc[:,"prct_l"], "-.", linewidth=0.5, color="C1")
#plt.plot(Longitude_axis, Result_FBSE.loc[:,"prct_g"], "-.", linewidth=0.5, color="C2")

plt.plot(lon, Result_CAI.loc[:,"prct_c"], ":", linewidth=0.5, color="C0")
plt.plot(lon, Result_CAI.loc[:,"prct_l"], ":", linewidth=0.5, color="C1")
plt.xlabel("Longitude, 0° = substellar point")
plt.annotate("BSE", xy=(-80, 67), xycoords="data", color="k", ha="left")
plt.annotate("Fe-BSE", xy=(-80, 79), xycoords="data", color="k", ha="center")
plt.annotate("CAI", xy=(-82, 89), xycoords="data", color="k", ha="center")

# shaded areas for night, pen, day
fill_un_pen_full_regions(0,100)

# limits
plt.xlim(-110, -70)

plt.tight_layout()
plt.savefig(_BASEPATH / "Phases_profile_density.pdf")

# %%
from matplotlib.patches import Rectangle

plt.figure(figsize=(3.22,6))
ax1 = plt.subplot(2,1,1)
plt.plot(lon, Result_BSE.loc[:,"prct_c"], "-", linewidth=0.5, color="grey")
plt.plot(lon, Result_BSE.loc[:,"prct_l"], "-", linewidth=0.5, color="C3")
#plt.plot(lon, Result_BSE.loc[:,"prct_g"], "-", linewidth=0.5)

plt.plot(lon, Result_FBSE.loc[:,"prct_c"], "--", linewidth=0.5, color="grey")
plt.plot(lon, Result_FBSE.loc[:,"prct_l"], "--", linewidth=0.5, color="C4")
#plt.plot(lon, Result_FBSE.loc[:,"prct_g"], "-", linewidth=0.5, color="C2")

plt.plot(lon, Result_CAI.loc[:,"prct_c"], ":", linewidth=0.5, color="grey")
plt.plot(lon, Result_CAI.loc[:,"prct_l"], ":", linewidth=0.5, color="C5")

plt.annotate("Crystals", xy=(120, 92), xycoords="data", color="k", ha="left")
plt.annotate("Melt", xy=(40, 92), xycoords="data", color="k", ha="center")
#plt.annotate("Gas", xy=(0, 3), xycoords="data", color="C2", ha="center")

# shaded areas for night, pen, day
fill_un_pen_full_regions(-5,105)
ax1.annotate("Nightside", (125,50), ha="center", va="center", rotation=90, fontsize=8, fontstyle="italic")
ax1.annotate("Dayside", (10,50), ha="center", va="center", rotation=90, fontsize=8, fontstyle="italic")
ax1.annotate("Penumbra", (107,50), ha="center", va="center", rotation=90, fontsize=8, fontstyle="italic")

# axes stuffs
plt.xlabel("Longitude, 0° = substellar point")
ax1.set_ylabel("Fraction of the phase")

# set limits
ax1.set_xlim(0, 180)
ax1.set_ylim(-5,105)

# annotations
plt.annotate("BSE", xy=(-80, 67), xycoords="data", color="k", ha="left")
plt.annotate("Fe-BSE", xy=(-80, 79), xycoords="data", color="k", ha="center")
plt.annotate("CAI", xy=(-82, 89), xycoords="data", color="k", ha="center")

#plt.tight_layout()
plt.savefig(_BASEPATH / "Phases_profile_density1.pdf", bbox_inches='tight')

# %% [markdown]
# ## Density as a function of longitude

# %%
plt.figure(figsize=(4.22,3.22), dpi=200)

# BSE composition
plt.plot(lon, Result_BSE["D"], "-", linewidth=0.5, color="C3", label="BSE")
plt.fill_between(lon, 
                 Result_BSE["D_low"],
                 Result_BSE["D_up"],
                 alpha=0.3, color="C3", label='Mapie 95%')

# Fe-rich BSE
plt.plot(lon, Result_FBSE["D"], "--", linewidth=0.5, color="C4", label="Fe-rich BSE")
# plt.fill_between(lon, 
#                  Result_FBSE["D_low"],
#                  Result_FBSE["D_up"],
#                  alpha=0.3, color="C4")

# CAI compo
plt.plot(lon, Result_CAI["D"], ":", linewidth=0.5, color="C5", label="CAI")
# plt.fill_between(lon, 
#                  Result_CAI["D_low"],
#                  Result_CAI["D_up"],
#                  alpha=0.3, color="C5")

# shaded areas and limits
fill_un_pen_full_regions(-1, 5)
plt.ylim(2.25, 4)
plt.xlim(-180, 180)

# shaded areas for night, pen, day
#fill_un_pen_full_regions(-5,105)
ax1.annotate("Nightside", (125,200), ha="center", va="center", rotation=90, fontsize=20, fontstyle="italic")
ax1.annotate("Dayside", (10,200), ha="center", va="center", rotation=90, fontsize=20, fontstyle="italic")
ax1.annotate("Penumbra", (107,200), ha="center", va="center", rotation=90, fontsize=200, fontstyle="italic")

# axes labels
plt.xlabel("Longitude, 0° = substellar point")
plt.ylabel("Density at surface, g/cm³")

# add legend
plt.legend(loc="upper center")
plt.tight_layout()

plt.savefig(_BASEPATH / "Density_profile_full.pdf")

# %%
plt.figure(figsize=(4.22,3.22), dpi=200)

# BSE composition
plt.plot(lon, Result_BSE["D"], "-", linewidth=0.5, color="C3", label="BSE")
# plt.fill_between(lon, 
#                  Result_BSE["D_low"],
#                  Result_BSE["D_up"],
#                  alpha=0.3, color="C3", label='Mapie 95%',edgecolor="none")

# Fe-rich BSE
plt.plot(lon, Result_FBSE["D"], "--", linewidth=0.5, color="C4", label="Fe-rich BSE")
# plt.fill_between(lon, 
#                  Result_FBSE["D_low"],
#                  Result_FBSE["D_up"],
#                  alpha=0.3, color="C4",edgecolor="none")

# CAI compo
plt.plot(lon, Result_CAI["D"], ":", linewidth=0.5, color="C5", label="CAI")
# plt.fill_between(lon, 
#                  Result_CAI["D_low"],
#                  Result_CAI["D_up"],
#                  alpha=0.3, color="C5",edgecolor="none")

# shaded areas and limits
fill_un_pen_full_regions(-1, 4)
plt.xlim(-110,-90)
plt.ylim(2.8,3.4)

# axes labels
plt.xlabel("Longitude, 0° = substellar point")
plt.ylabel("Density at surface, g/cm³")

# set legend
plt.legend(loc="lower left")

plt.tight_layout()

plt.savefig(_BASEPATH / "Density_profile_shore.pdf")

# %%
# plt.figure(figsize=(4.22,3.22), dpi=200)
# plt.plot(lon, Result_BSE.visco, "-", linewidth=2.0, color="C3", label="BSE")
# plt.fill_between(lon, 
#                  Result_BSE.visco-Result_BSE.visco_std,
#                  Result_BSE.visco+Result_BSE.visco_std,
#                  alpha=0.3, color="C3", edgecolor="none")
# 
# plt.fill_betweenx([-2,23],-180,-90, color="grey", alpha=0.2)
# plt.fill_betweenx([-2,23],90,180, color="grey", alpha=0.2)
# plt.legend(loc=9)
# plt.xlabel("Longitude, 0° = substellar point")
# plt.ylabel("Viscosity at surface, log$_{10}$ Pa$\cdot$s")
# plt.xlim(-70,70)
# plt.ylim(-4,0)
# plt.tight_layout()
# 
# plt.savefig(_BASEPATH / "Viscosity_profile_center_1.pdf")
# 
# plt.plot(lon, Result_FBSE.visco, "--", linewidth=2.0, color="C4", label="Fe-rich BSE")
# plt.fill_between(lon, 
#                  Result_FBSE.visco-Result_FBSE.visco_std,
#                  Result_FBSE.visco+Result_FBSE.visco_std,
#                  alpha=0.1, facecolor="C4", edgecolor="none")
# plt.legend(loc=9)
# 
# plt.savefig(_BASEPATH / "Viscosity_profile_center_2.pdf")
# 
# 
# plt.plot(lon, Result_CAI.visco, ":", linewidth=2.0, color="C5", label="CAI")
# plt.fill_between(lon, 
#                  Result_CAI.visco-Result_CAI.visco_std,
#                  Result_CAI.visco+Result_CAI.visco_std,
#                  alpha=0.3, color="C5", edgecolor="none")
# plt.legend(loc=9)
# 
# plt.savefig(_BASEPATH / "Viscosity_profile_center_3.pdf")
