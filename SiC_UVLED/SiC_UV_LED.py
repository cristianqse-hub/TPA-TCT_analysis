#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:10:27 2023

ANALYSIS AND RESULTS _ ROOT FORMAT

@author: cristian
"""

#%%
import sys
import importlib

import ROOT
from ROOT import gStyle, gPad, gROOT
gROOT.SetBatch(ROOT.kTRUE)


#%%

import glob
import numpy as np
from scipy.optimize import curve_fit
import time

from graphical_lib import new_Canvas, plot_StackPlts
from analysis_lib_3rdC import ProcessedFile, process_files

# PHYSICAL Parameters -----------------------------------------

# SiC_zfac = 3.134 # theoretical objective
SiC_zfac = 2.83  # real, Raul's fit
Area = 3.0e-3**2 # SI
SiC_epsilon = 8.854e-12 * 9.6 # SI # Permittivity 9.6 / 9.66 ~
# http://www.qualitymaterial.net/news_list85.html
qe = 1.602e-19
dt = 300/3000 * 1e-9 # ns/sample

colors = np.array([ROOT.kBlack, ROOT.kRed+1, ROOT.kBlue, ROOT.kGreen+2,
                  ROOT.kOrange+1, ROOT.kMagenta-7, ROOT.kCyan+1, ROOT.kGray+2, 
                  ROOT.kMagenta+2, ROOT.kRed+3, ROOT.kYellow-2,  ROOT.kCyan+3,])



# marks = ["solid","solid","solid","solid","solid","solid","solid",
#         "dot","dot","dot","dot","dot","dot","dot"]

def func_ChargeP(x,a,b,c,d,e):
    return a * (np.arctan((c-x)/b) + np.arctan((x-d)/b)) + e # e -> SPA

def func_ChargePH(x,a,b,c,e):
    return a * np.arctan((x-c)/b) + e # e -> SPA

#%% --------------------------------------------------------------------------- ZSCANs
#%%  ---------------- PARAMETERS AND FILES -----------------------------------
#       Get FILES    ------------------------------------------
#  ROOT file open and parameters: -----------------------------
# Files source dir
source_dir = "Data_groups/" # files folder

parameters = {
                "skip_lines": 20,
                "dt": 0.1,
                "tBL": 20.1,
                "tLeft": 51.50,
                "tRight": 54.1,
                "SPA_NorQ": 0.0,
                "TPA_CorrF": 2.0,
                "FitRangePercen": 50.0,
                "mode": "AC",
                "scan_group": 0,
                "scan_num": 0,
                "dzSiC": SiC_zfac,
                "avrLP": 0.023,
                "WPC_T": 0.6,
                "ToA_CCF": 200,
                "SPA_WF_idx": 0,
                "SPA_WF_N": 15,
                "Amplitude_cut": 0.04,
                "AnalysisMode": "STD",
                "PrintTest": "NO",
                "ToT_th": 0.3,
                "wf_add0s":0.0,
                "z_sign": -1,
            }   



# Get Filenames with glob
files = glob.glob(source_dir+"LED_UV_F2W1/*") # FILES
files = sorted(files, key=str.lower) # Short alphabetically

for i, _file in enumerate(files):
    print(f"[{i} - {_file}]")

data = process_files(files, parameters)
data_fresh = ProcessedFile(parameters, "/home/cquintana/Software/Docker/TPA-TCT/Data_groups/LED_UV_1MW2/20240704_1811_1MW2_-1000V_1mm beam_400nm_LED_0_lamp_off_reps2_zscan_baseline_substrated")
max_chargeFresh = np.max(np.abs(data_fresh.ChargeCSPA_Avr))

print("done")
#%% NORM LED POWER
#%% ----------------------------------- Norm CHARGE
# Averga profile study

led_bias = np.array([3, 3.1, 3.15, 3.2])
led_power = np.array([0.234, 1.09, 1.81, 2.60])
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "UV LED bias voltage [V]" , "Measured power [mW]"]

fig[0].cd()

zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        led_bias,  
                        led_power, 
                        titles=mtitle, 
                            lims=[2.95, 3.25,0,3],
                        plt_ops=["ap"]+[int(colors[0]),2,int(colors[0]),20], 
                        addlg=[False, ""])


fap = ROOT.TF1("fap","[0]*TMath::Exp([1]*x + [2] + [3])",3,3.20,4)
g = zsc_plots[0][0]

g.Fit(fap,"R")

fig[0].Draw()
fig[0].SaveAs(f"Images/LED/UV_NORM_POWER_EMISSION.png")


#%%

plops = ["al"] + ["ls"] * 200
mtitle = ["", "z (SiC) [um]" , "Absolute Charge [fC]"]
lgAdd =  True
max_charge = np.zeros(len(files))
p1_charge = np.zeros(len(files))
p2_charge = np.zeros(len(files))
zSiC_p1 = -data[0].zSiC[17] *-1.0 -38
zSiC_p2 = -data[0].zSiC[28] *-1.0 -38

total_charge = np.zeros(len(files))
ledBias = ["0.00V","3.00V","3.10V","3.15V","3.20V"]
ledRPower = ["0.00"] + [f"{led_power[i]/2.6:.2f}" for i in range(4)]
dutBias = np.array([-850, -500, -100, 100, 500, 850, 1000]).astype(float)
figs = [new_Canvas(f"fig{i}", lgPos=[["",0.17,0.65,0.40,0.88]]) for i in range(7) ]
con = -1
for k, fig in enumerate(figs):
    fig[0].cd()
    zsc_plots = [[], 1]
    #lgAdd = True if k==0 else False
    for i in range(5):
        con += 1
        total_charge[con] = np.sum(data[k*5+i].Charge_Avr)
        zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                                    -data[k*5+i].zSiC[:40]*-1.0-38,  
                                    np.abs(data[k*5+i].Charge_Avr), 
                                    titles=mtitle, 
                                    lims=[0,0,0,160],
                                plt_ops=[plops[i]]+[int(colors[i]),2,1,7], 
                                    addlg=[lgAdd, "LED Rel. Power = "+ledRPower[i]])
        
        text = ROOT.TLatex(20,150,f"F2W1 - VBias = {data[k*5+i].label}")
        text.Draw()

    fig[0].Draw()
    fig[0].SaveAs(f"Images/LED/ChargeProfile_{k}.png")

figs = [new_Canvas(f"fig{i}", lgPos=[["",0.17,0.60,0.50,0.88]]) for i in range(7) ]

#CORR
con = -1
mtitle = ["", "z (SiC) [um]" , "Absolute TPA Charge [fC]"]
for k, fig in enumerate(figs):
    fig[0].cd()
    zsc_plots = [[], 1]
    #lgAdd = True if k==0 else False
    for i in range(5):
        con += 1
        total_charge[con] = np.abs(np.sum(data[k*5+i].ChargeCSPA_Avr))
        max_charge[con] = np.max(np.abs(data[k*5+i].ChargeCSPA_Avr))
        p1_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[17])
        p2_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[28])

        zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                                    -data[k*5+i].zSiC[:40] *-1.0 -38,  
                                    np.abs(data[k*5+i].ChargeCSPA_Avr), 
                                    titles=mtitle, 
                                    lims=[0,0,0,160],
                                plt_ops=[plops[i]]+[int(colors[i]),2,1,7], 
                                    addlg=[lgAdd, "LED Rel. Power = "+ledRPower[i]])
        
        text = ROOT.TLatex(20,160,f"F2W1 - VBias = {data[k*5+i].label}")
        text.Draw()

    fig[0].Draw()
    fig[0].SaveAs(f"Images/LED/ChargeProfile_SPACorr_{k}.png")

# Norm respect no LED
con = -1
figs = [new_Canvas(f"fig{i}", lgPos=[["",0.17,0.60,0.50,0.88]]) for i in range(7) ]
mtitle = ["", "z (SiC) [um]" , " TPA Charge increment [%]"]
for k, fig in enumerate(figs):
    fig[0].cd()
    zsc_plots = [[], 1]
    #lgAdd = True if k==0 else False
    noLed_norm = 0 
    for i in range(5):
        if i == 0:
            con += 1
            total_charge[con] = np.abs(np.sum(data[k*5+i].ChargeCSPA_Avr))
            max_charge[con] = np.max(np.abs(data[k*5+i].ChargeCSPA_Avr))
            p1_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[17])
            p2_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[28])

            noLed_norm = np.abs(data[k*5+i].ChargeCSPA_Avr)

        else: 
            con += 1
            total_charge[con] = np.abs(np.sum(data[k*5+i].ChargeCSPA_Avr))
            max_charge[con] = np.max(np.abs(data[k*5+i].ChargeCSPA_Avr))
            p1_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[17])
            p2_charge[con] = np.abs(data[k*5+i].ChargeCSPA_Avr[28])

            zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                                        -data[k*5+i].zSiC[:40] *-1.0 -38,  
                                        (np.abs(data[k*5+i].ChargeCSPA_Avr)/noLed_norm-1)*100, 
                                        titles=mtitle, 
                                        lims=[0,50,-10,600],
                                    plt_ops=[plops[i-1]]+[int(colors[i]),2,1,7], 
                                        addlg=[lgAdd, "LED Rel. Power = "+ledRPower[i]])
            
            text = ROOT.TLatex(28,605,f"F2W1 - VBias = {data[k*5+i].label}")
            text.Draw()

    fig[0].Draw()
    fig[0].SaveAs(f"Images/LED/ChargeProfile_NormNoled_{k}.png")

max_charge = np.reshape(max_charge, (5,7), order="F")
p1_charge = np.reshape(p1_charge, (5,7), order="F")
p2_charge = np.reshape(p2_charge, (5,7), order="F")

total_charge = np.reshape(total_charge, (5,7), order="F")

#%% ----------------------------------- Total CHARGE
# Averga profile study
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Total TPA Charge [fC]"]

fig[0].cd()

plops = ["apl"] + ["lps"] * 200
plops2 = ["ap"] + ["sp"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            total_charge[i,:], 
                            titles=mtitle, 
                            lims=[-1000,1100,-1,5000],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            total_charge[i,:3], 
                            titles=mtitle, 
                            lims=[-1000,1100,-1,5000],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            total_charge[i,3:], 
                            titles=mtitle, 
                            lims=[-1000,1100,-1,5000],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])

fig[0].Draw()
fig[0].SaveAs(f"Images/LED/TotalCharge_comparison.png")
#%% ----------------------------------- Norm Total CHARGE

# Averga profile study
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , " Total TPA Charge (Norm. fresh)"]

fig[0].cd()

plops = ["apl"] + ["lps"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            total_charge[i,:], 
                            titles=mtitle, 
                                lims=[0,0,0,4],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            total_charge[i,:3]/np.sum(data_fresh.Charge_Avr), 
                            titles=mtitle, 
                                lims=[0,0,0,4],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            total_charge[i,3:]/np.sum(data_fresh.Charge_Avr), 
                            titles=mtitle, 
                                lims=[0,0,0,4],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
text = ROOT.TLatex(500,0.94,f"<- 1MW2(fresh) ref.")
text.SetTextSize(0.04)
text.Draw()
fig[0].Draw()
fig[0].SaveAs(f"Images/LED/NormTotalCharge_comparison.png")
fig[0].SaveAs(f"Images/LED/NormTotalCharge_comparison.pdf")


#%% ----------------------------------- Peaks  CHARGE
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Norm. TPA Charge at P1"]

fig[0].cd()
plops = ["apl"] + ["lps"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            p1_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            p1_charge[i,:3]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            p1_charge[i,3:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])

text = ROOT.TLatex(100,2.8,f"z (SiC) = {zSiC_p1:.2f} um" )
text.Draw()
fig[0].Draw()
fig[0].SaveAs(f"Images/LED/Peak_1_NormTotalCharge_comparison.png")
#fig[0].SaveAs(f"Images/LED/Peak_1_NormTotalCharge_comparison.pdf")

#%% ------------------------------------------------------------- peaks a 0V


lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.65,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Norm. TPA Charge at P1 - P2"]

fig[0].cd()
plops = ["apl"] + ["lps"] * 200

for i in range(1):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            p1_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops2[i]]+[int(colors[1]),2,int(colors[1]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            p1_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops[i+1]]+[int(colors[1]),2,int(colors[1]),20], 
                            addlg=[True, f"Q at z = {zSiC_p1:.2f} um"])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            p1_charge[i,3:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops[i+1]]+[int(colors[1]),2,int(colors[1]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            p2_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops2[i+1]]+[int(colors[2]),2,int(colors[2]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            p2_charge[i,:3]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops[i+1]]+[int(colors[2]),2,int(colors[2]),20], 
                            addlg=[True, f"Q at z = {zSiC_p2:.2f} um"])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            p2_charge[i,3:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,2.5],
                            plt_ops=[plops[i+1]]+[int(colors[2]),2,int(colors[2]),20], 
                            addlg=[False, f"Q at z = {zSiC_p2}"])

text = ROOT.TLatex(100,2.3,f"LED Rel. Power = "+ledRPower[0] )
text.Draw()
fig[0].Draw()
fig[0].SaveAs(f"Images/LED/PeakS_0V_NormTotalCharge_comparison.png")

#%%%-----------------

lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Norm. TPA Charge at P2"]

fig[0].cd()
plops = ["apl"] + ["lps"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            p2_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            p2_charge[i,:3]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            p2_charge[i,3:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])

text = ROOT.TLatex(100,2.8,f"z (SiC) = {zSiC_p2:.2f} um" )
text.Draw()
fig[0].Draw()
fig[0].SaveAs(f"Images/LED/Peak_2_NormTotalCharge_comparison.png")
#fig[0].SaveAs(f"Images/LED/Peak_2_NormTotalCharge_comparison.pdf")

#%% ----------------------------------- max2 CHARGE
# Averga profile study
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Max. TPA Charge [fC]"]

fig[0].cd()
plops = ["apl"] + ["lps"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            max_charge[i,:], 
                            titles=mtitle, 
                                lims=[0,0,0,450],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            max_charge[i,:3], 
                            titles=mtitle, 
                                lims=[0,0,0,450],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            max_charge[i,3:], 
                            titles=mtitle, 
                                lims=[0,0,0,450],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])

fig[0].Draw()
fig[0].SaveAs(f"Images/LED/MaxCharge_comparison.png")

#%% ----------------------------------- Norm CHARGE
# Averga profile study
lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.17,0.60,0.44,0.88]]) 
mtitle = ["", "DUT bias voltage [V]" , "Max. Norm. TPA Charge"]

fig[0].cd()
plops = ["apl"] + ["lps"] * 200

for i in range(5):
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:],  
                            max_charge[i,:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops2[i]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])
    
    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[:3],  
                            max_charge[i,:3]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[True, "LED Rel. Power = "+ledRPower[i]])

    zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                            dutBias[3:],  
                            max_charge[i,3:]/max_chargeFresh, 
                            titles=mtitle, 
                                lims=[0,0,0,3],
                            plt_ops=[plops[i+1]]+[int(colors[i]),2,int(colors[i]),20], 
                            addlg=[False, "LED Rel. Power = "+ledRPower[i]])

fig[0].Draw()
fig[0].SaveAs(f"Images/LED/MaxNormCharge_comparison.png")


lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.20,0.70,0.47,0.88]]) 
mtitle = ["", "Z (SiC) [um]" , "Corrected Charge [fC]"]


fig[0].cd()

zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        data[0].zSiC[:40]-38,  
                        data[0].ChargeCSPA_Avr , 
                        titles=mtitle, 
                            lims=[0,0,0,300],
                        plt_ops=["al"]+[int(colors[3]),2,int(colors[0]),20], 
                        addlg=[True, "F2W1, bias = -850V"])

zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        data[25].zSiC[:40]-39,  
                        -data[25].ChargeCSPA_Avr, 
                        titles=mtitle, 
                            lims=[0,0,0,300],
                        plt_ops=["sl"]+[int(colors[4]),2,int(colors[0]),20], 
                        addlg=[True, "F2W1, bias = +850V"])

zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        data_fresh.zSiC[:40]-32,  
                        data_fresh.ChargeCSPA_Avr, 
                        titles=mtitle, 
                            lims=[0,0,0,300],
                        plt_ops=["sl"]+[int(colors[0]),2,int(colors[0]),20], 
                        addlg=[True, "1MW2, bias = -1kV"])

fig[0].Draw()
fig[0].SaveAs(f"Images/ProfileComparison/F2W1_-850_850V.png")

#%%


lgAdd =  True
zsc_plots = [[], 1]
fig = new_Canvas(f"fig", lgPos=[["",0.20,0.70,0.47,0.88]]) 
mtitle = ["", "Z (SiC) [um]" , "Corrected Charge [fC]"]


fig[0].cd()


zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        data[25].z[:40],  
                        data[25].Charge_Avr, 
                        titles=mtitle, 
                            lims=[0,0,0,0],
                        plt_ops=["al"]+[int(colors[4]),2,int(colors[0]),20], 
                        addlg=[False, "F2W1, bias = +850V"])

fig[0].Draw()
fig[0].SaveAs(f"Images/ProfileComparison/F2W1_Iscan_max.png")


zsc_plots = plot_StackPlts(fig, zsc_plots[0], 
                        data[25].zSiC[:40]-39,  
                        data[25].Charge_Avr, 
                        titles=mtitle, 
                            lims=[0,0,0,0],
                        plt_ops=["al"]+[int(colors[4]),2,int(colors[0]),20], 
                        addlg=[False, "F2W1, bias = +850V"])

fig[0].Draw()
fig[0].SaveAs(f"Images/ProfileComparison/F2W1_Iscan_2_max.png")
