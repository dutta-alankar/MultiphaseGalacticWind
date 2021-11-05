################################################################
# Author: Drummond Fielding
# Reference: Fielding & Bryan (2021)
# Date: 08 Aug 2021
# Brief: This code calculates the structure of multiphase galactic winds.
#
# Execution:
# >> python singlePhaseCloudEvol.py
#
# Output: a 9 panel figure showing the properties of a multiphase galactic wind relative to a single phase galactic wind 
# 
# Overview:
# - First the code calculates the structure of a single phase galactic wind in the manner of Chevalier and Clegg (1985). 
# - Then the code calculates the structure of a multiphase galactic wind. 
# - The default values are:
#   - SFR            = 20 Msun/yr   (star formation rate)
#   - eta_E          = 1            (energy loading)
#   - eta_M          = 0.1          (initial hot phase or single pahse mass loading)
#   - eta_M_cold     = 0.2          (initial cold phase mass loading)
#   - M_cloud_init   = 10^3 Msun    (initial cloud mass)
#   - v_cloud_init   = 10^1.5 km/s  (initial cloud velocity)
#   - r_sonic        = 300 pc       (sonic radius)
#   - Z_wind_init    = 1 * Z_solar  (initial wind metallicity)
#   - Z_cloud_init   = 0.1 * Z_solar      (initial cloud metallicity)
#   - v_circ0        = 150 km/s     (circular velocity of external isothermal gravitational potential)
#
# Edits (Ritali, Alankar, Prateek): 
#   - Solving for the time evolution of relative velocity, mass, and metallicity for "single cloud framework".
#   - Adding metallicity dependence in the calculation of mean molecular mass.
################################################################


import numpy as np
import h5py 
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
#import cmasher as cmr
from matplotlib.lines import Line2D
import seaborn as sns

sns.set()
## Plot Styling
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 2.75
matplotlib.rcParams['xtick.major.size'] = 2.75
matplotlib.rcParams['ytick.minor.size'] = 1.75
matplotlib.rcParams['xtick.minor.size'] = 1.75
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.labelright'] = False
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')
plt.rc('legend',fontsize=5) # using a named size

## Defining useful constants
gamma   = 5/3.
kb      = 1.3806488e-16
mp      = 1.67373522381e-24
km      = 1e5
s       = 1
yr      = 3.1536e7
Myr     = 3.1536e13
Gyr     = 3.1536e16
pc      = 3.086e18
kpc     = 1.0e3 * pc
Msun    = 2.e33
#mu      = 0.62 
#muH     = 1/0.75
#X_solar, Y_solar, Z_solar = 0.71, 0.27, 0.02
X_solar, Y_solar, Z_solar    = 0.7154, 0.2703, 0.0143 # solar metallicity
#mu        = 1./(2*X+0.75*Y+0.5625*Z)

# ********************************************************************************* #
# addition of metallicity dependence on mu
def fractionMetallicity(fracZ):
    #fracZ is fraction of Solar metallicity (linear)
       
    Xp, Yp, Zp = X_solar/(X_solar+Y_solar+fracZ*Z_solar), Y_solar/(X_solar+Y_solar+fracZ*Z_solar), fracZ*Z_solar/(X_solar+Y_solar+fracZ*Z_solar)
    mup        = 1./(2*Xp+0.75*Yp+0.5625*Zp)
    muHp       = mup/Xp
    return [Xp, Yp, Zp, mup, muHp] # corrected values
    
# ********************************************************************************* ##


"""
Cooling curve as a function of density, temperature, metallicity
"""
Cooling_File = "./CoolingTables/z_0.000.hdf5" ### From Wiersma et al. (2009) appropriate for z=0 UVB
f            = h5py.File(Cooling_File, 'r')
i_X_He       = -3 
Metal_free   = f.get('Metal_free')
Total_Metals = f.get('Total_Metals')
log_Tbins    = np.array(np.log10(Metal_free['Temperature_bins']))
log_nHbins   = np.array(np.log10(Metal_free['Hydrogen_density_bins']))
Cooling_Metal_free   = np.array(Metal_free['Net_Cooling'])[i_X_He] ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
Cooling_Total_Metals = np.array(Total_Metals['Net_cooling'])
HHeCooling           = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Metal_free)
ZCooling             = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Total_Metals)
f.close()
Zs          = np.logspace(-2,1,31)
Lambda_tab  = np.array([[[HHeCooling.ev(lT,ln)+Z*ZCooling.ev(lT,ln) for Z in Zs] for lT in log_Tbins] for ln in log_nHbins])
Lambda_z0   = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs), Lambda_tab, bounds_error=False, fill_value=-1e-30)

def tcool_P(T,P, metallicity):
    """
    cooling time function
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    """
    T = np.where(T>10**8.98, 10**8.98, T)
    T = np.where(T<10**2, 10**2, T)
    _, _, _, mu, muH = fractionMetallicity(metallicity)
    nH_actual = P/T*(mu/muH)
    nH = np.where(nH_actual>1, 1, nH_actual)
    nH = np.where(nH<10**-8, 10**-8, nH)
    return (1/(gamma-1)) * (muH/mu) * kb * T / ( nH_actual * Lambda_z0((np.log10(nH),np.log10(T), metallicity))) #correction of pre-factor

def Lambda_T_P(T,P, metallicity):
    """
    cooling curve function as a function of
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence 
    """
    _, _, _, mu, muH = fractionMetallicity(metallicity)
    nH = P/T*(mu/muH)
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH),np.log10(T), metallicity))
Lambda_T_P  = np.vectorize(Lambda_T_P)

def Lambda_P_rho(P, rho, metallicity):
    """
    cooling curve function as a function of
    P in units of erg * cm**-3
    rho in units of g * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence 
    """
    _, _, _, mu, muH = fractionMetallicity(metallicity)
    nH = rho / (muH * mp)
    T  = P/kb / (rho/(mu*mp))
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH),np.log10(T), metallicity))
Lambda_P_rho  = np.vectorize(Lambda_P_rho)

def Onephase_Cloud_Evo(t, state):
    """
    Calculates the time derivative of M_cloud, v_cloud, and Z_cloud. 
    Used with solve_ivp. 
    """
    global v_wind
    global rho_wind 
    global Pressure 
    global Z_wind
    global T_cloud
    global T_wind
    global f_turb0
    global TurbulentVelocityChiPower
    global ColdTurbulenceChiPower
    global M_cloud_min
    global drag_coeff

    M_cloud    = state[0]
    v_cloud    = state[1]
    Z_cloud    = state[2]

    _, _, _, mu, muH = fractionMetallicity(Z_cloud/Z_solar)

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind              # density contrast
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud
    
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)

    # cloud evolution
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

    vdot_cloud = ((p_dot_drag + v_rel*Mdot_grow ) / M_cloud ) * np.where(M_cloud>M_cloud_min, 1, 0)

    Zdot_cloud = ((Z_wind-Z_cloud) * Mdot_grow / M_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    return np.r_[Mdot_cloud, vdot_cloud, Zdot_cloud]

def cloud_ksi(r, state):
    """
    function to calculate the value of ksi = t_mix / t_cool
    """
    global v_wind
    global rho_wind 
    global Pressure 
    global Z_wind
    global T_cloud
    global rho_cloud
    global chi
    global T_wind
    global f_turb0
    global TurbulentVelocityChiPower

    M_cloud    = state[0]
    v_cloud    = state[1]
    Z_cloud    = state[2]

    # cloud transfer rates
    
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    return ksi
    
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
Mdot_coefficient            = 1.0/3.0
drag_coeff                  = 0.5
f_turb0                     = 10**0
M_cloud_min                 = 1e-2*Msun     ## minimum mass of clouds

# cold cloud initial properties
T_cloud             = 1e4
log_M_cloud_init    = 3
M_cloud_init        = 10**log_M_cloud_init * Msun
Z_cloud_init        = 0.1 * Z_solar 
v_cloud_init        = 0. * km/s 
Pressure            = 1.0e3 * kb
_, _, _, mu, muH    = fractionMetallicity(Z_cloud_init/Z_solar)
rho_cloud           = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium

# wind intial properties
Z_wind              = 1.0 * Z_solar
v_wind              = 1.0e3 * km/s 
T_wind              = 1.0e6                   
chi                 = 100              # density contrast
rho_wind            = rho_cloud / chi

T_mix        = (T_wind*T_cloud)**0.5
Z_mix        = (Z_wind*Z_cloud_init)**0.5

xi0 = np.logspace(np.log10(1.e-2), np.log10(1.e2), 9)

fig1, axs = plt.subplots(figsize=(6,18), nrows= 3, ncols=2)#, sharex=True)
plt.suptitle(
    "Cloud growth modelling, $C_D$=%.1f, $v_{rel}$ = %d km/s, $Z_{cl}=%.1fZ_\odot$"\
        %(drag_coeff, int((v_wind-v_cloud_init)/(km/s)), (Z_cloud_init/Z_solar) ) )

for xi in xi0:
    t_cool_layer  = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer  = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    R_cloud_init  = xi * f_turb0 * (v_wind-v_cloud_init) *  t_cool_layer
    M_cloud_init  = 4 * np.pi/3 * rho_cloud * (R_cloud_init**3) 
    t_cc0         = chi**.5 * R_cloud_init / (v_wind-v_cloud_init)

    print("R_cl0 = %.2e pc" %(R_cloud_init/pc))
    print("t_cc  = %.2e Myr"%(t_cc0/Myr))

    #### ICs 
    initial_conditions = np.r_[M_cloud_init, v_cloud_init, Z_cloud_init]
    t_init = 0.
    t_stop = 5.0e3 * Myr
    teval = np.logspace(np.log10(1.e-2), np.log10(t_stop), 1000)

    ### integrate!
    sol = solve_ivp(Onephase_Cloud_Evo, [t_init, t_stop], initial_conditions, t_eval=teval ,dense_output=True, rtol=1e-6)
    print(sol.message)

    ## gather solution and manipulate into useful form
    time            = sol.t
    M_cloud_sol     = sol.y[0]
    v_cloud_sol     = sol.y[1]
    Z_cloud_sol     = sol.y[2]
    

    axs[0,0].plot(time/t_cc0, M_cloud_sol/M_cloud_init, label=r"$\xi=$ %.2f " %xi)
    axs[0,0].set_yscale("log")
    axs[0,0].set_xscale("linear")
    axs[0,0].set_xlim(1.e-5,20)
    axs[0,0].set_ylim(ymin=1.e-2,ymax=12)
    axs[0,0].set_ylabel(r"$\rm M_{cl}/M_{cl,0}$")
    axs[0,0].set_xticklabels([])

    axs[1,0].plot(time/t_cc0, (v_wind-v_cloud_sol) / (v_wind-v_cloud_init), label=r"$\xi=$ %.2f " %xi)
    axs[1,0].set_ylim(0., 1.2)
    axs[1,0].set_xscale("linear")
    axs[1,0].set_xlim(1.e-5,20)
    axs[1,0].set_ylabel(r"$\rm v_{rel}/v_{wind}$")
    axs[1,0].set_xticklabels([])

    axs[2,0].plot(time/t_cc0, Z_cloud_sol/Z_wind, label=r"$\xi=$ %.2f " %xi)
    axs[2,0].set_xlim(1.e-5,20)
    axs[2,0].set_ylim(0., 1.1)
    axs[2,0].set_xscale("linear")
    axs[2,0].set_ylabel(r"$\rm Z_{cl}/Z_{wind}$")
    axs[2,0].set_xlabel(r"$\rm t/t_{cc}$")
    
    axs[0,1].plot(time/Myr, M_cloud_sol/Msun, label=r"$\xi=$ %.2f " %xi)
    axs[0,1].set_yscale("log")
    axs[0,1].set_ylim(1,5.e13)
    axs[0,1].set_xscale("log")
    axs[0,1].set_xlim(1.e-1,5.e3)
    axs[0,1].set_ylabel(r"$\rm M_{cl}$ $\rm[M_\odot]$")
    axs[0,1].set_xticklabels([])

    axs[1,1].plot(time/Myr, v_cloud_sol / (km/s), label=r"$\xi=$ %.2f " %xi)
    axs[1,1].set_xscale("log")
    axs[1,1].set_xlim(1.e-1,5.e3)
    axs[1,1].set_ylabel(r"$\rm v_{cl}$ [km/s]")
    axs[1,1].set_xticklabels([])

    axs[2,1].plot(time/Myr, Z_cloud_sol/Z_solar, label=r"$\xi=$ %.2f " %xi)
    axs[2,1].set_xscale("log")
    axs[2,1].set_xlim(1.e-1,5.e3)
    axs[2,1].set_ylabel(r"$\rm Z_{cl}$ $\rm[Z_\odot]$")
    axs[2,1].set_xlabel("Time (Myr)")

axs[2,0].legend()
plt.show()
