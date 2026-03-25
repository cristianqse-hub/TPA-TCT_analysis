#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:00:20 2022

@author: cristian
"""


import numpy as np
import plotly.graph_objects as go
import plotly.colors
from PIL import ImageColor
import ROOT
from ROOT import gStyle, gPad


#%%


#       PLOT FUNCTIONS    -------------------------------------------

def draw_LinePlot(fig,X,Y,ccolor,legend,my_width=3, lmod = "solid", mrow= 1, mcol =1): # legend = [True, "title"]
    fig.add_trace(go.Scatter(x=X, 
                             y=Y,
                        mode='lines', # 'lines', 'markers', 'lines+markers'
                        name=legend[1],
                        line_shape='linear', # 'spline'
                        showlegend = legend[0],
                        line = dict(
                                    color=ccolor,
                                    width=my_width,                               
                                    dash=lmod # 'dash', 'dot', and 'dashdot'
                                    ),
                        ) , row=mrow, col=mcol)
    return fig




def draw_ScatterPlot(fig,X,Y,ccolor,legend,my_width=3, msymbol='circle',
                     err_x = [], err_y = [], mrow= 1, mcol =1): # legend = [True, "title"]
    err_x = np.array(err_x)
    err_y = np.array(err_y)
    show_err_x = True # ERROR HANDLING
    show_err_y = True
    if len(err_x) == 0:
        show_err_x = False
        exp = X*0
        exn = X*0
    else:
        exn = err_x[0,:]
        exp = err_x[1,:]
    if len(err_y) == 0:
        show_err_y = False
        eyp = X*0
        eyn = X*0
    else:
        eyn = err_y[0,:]
        eyp = err_y[1,:]
    fig.add_trace(go.Scatter(x=X, 
                             y=Y,
                        mode='markers', # 'lines', 'markers', 'lines+markers'
                        name=legend[1],
                        showlegend = legend[0],
                        marker_symbol=msymbol,
                        marker_line_color=ccolor,
                        marker_color=ccolor,
                        marker_line_width=0, 
                        marker_size=my_width,
                        error_x=dict(
                                type='data',
                                visible = show_err_x,
                                symmetric=False,
                                array=exp,
                                arrayminus=exn),
                        error_y=dict(
                                type='data',
                                visible = show_err_y,
                                symmetric=False,
                                array=eyp,
                                arrayminus=eyn)               
                        ), row=mrow, col=mcol)
    return fig

def draw_layout(fig, labels, w = 700, h = 600, txtsize = 14, lg_pos = [0.8,1], mrow= 1, mcol =1): #labels = ["title","x axis","y axis","legend"]
    fig.update_layout(
        title=labels[0],
        xaxis_title=labels[1],
        width = w, height = h,
        yaxis_title=labels[2],
        legend_title=labels[3],
        legend=dict(
        x=lg_pos[0],
        y=lg_pos[1]),
        font=dict(
            family="Courier 10 Pitch, monospace",
            size=txtsize,
            color="Black"
        ) 
    )
    return fig
        
def draw_setLimits(fig,limits, mrow= 1, mcol =1): # limits = [x0,xl,y0,yl]
    if limits[0] != limits[1]:
        fig.update_xaxes(range=[limits[0],limits[1]], row=mrow, col=mcol) 
    if limits[2] != limits[3]:
        fig.update_yaxes(range=[limits[2],limits[3]], row=mrow, col=mcol) 
    return fig


def draw_heatmap(value_map, color_map, mytitle, legend_title, save = False):
    
    evMap = eval(value_map)
    fig = go.Figure(data=go.Heatmap(
                   z=evMap,
                   dy=0.18,
                   y0=0,
                   dx=0.18,
                   x0=0,
                   colorscale = color_map,
                   colorbar=dict(
                                  title=legend_title,
                              )))
    fig.update_layout(margin = dict(t=80,r=20,b=80,l=20),
        showlegend = False,
        width = 900, height = 900,
        autosize = False,
        title = mytitle,
        font=dict(size=18))
    
    fig.update_yaxes(title = "Y [mm]",
                     scaleanchor = "x",
                      scaleratio = 1)
    fig.update_xaxes(title = "X [mm]")
    fig.show()
    
    if save:
        mytitle = mytitle.replace("/", "-")
        mytitle = mytitle.replace(" ", "_")
        value_map = value_map.replace(":", "-")
        fig.write_image("Auto_Figures/heatmap-"+mytitle+'_'+value_map+"_.png")   


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)
        

# Identical to Adam's answer


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )    



#%% --------------------  PLOT SCANS FUNCTIONS   ------------------------------


def plot_WFs(WFarray, fig, ccolor, plt_leg = False, legend="", row=1, col=1):
    dim1, dim2 = WFarray.shape
    for i in range(1,dim1):
        if i == 1:
            fig = draw_LinePlot(fig, WFarray[0,:], WFarray[i,:], ccolor, 
                                [plt_leg, legend], my_width = 1 , mcol=col, mrow=row)
        else:
            fig = draw_LinePlot(fig, WFarray[0,:], WFarray[i,:], ccolor, 
                                [False, legend],  my_width = 1, mcol=col, mrow=row)
    return fig
    

#%% --------------------  ROOT PLOTs   ------------------------------
# import ROOT
# import numpy as np

def new_Canvas(ID, title="New canvas", W=700, H=500, div=[], scale = False, lgPos=[]): # 
    if scale == True:
        H *= div[1]
        W *= div[0]
        
    cv = ROOT.TCanvas(ID, title, W, H)
    
    if len(div) == 2:
        cv.Divide(div[0],div[1])
    elif len(div) == 4:
        cv.Divide(div[0],div[1],div[2],div[3])
    else:
        div = [1,1]
        
    lg = []
    for i in range(int(div[0]*div[1])):
        if len(lgPos) == 0:
            idLg = ROOT.TLegend(0.6,0.7,0.9,0.9)
            lg.append(idLg)
        else:
            header, x1,y1,x2,y2 = lgPos[i]
            idLg = ROOT.TLegend(x1,y1,x2,y2)
            if header == "":
                #idLg.SetHeader("","C")
                pass
            else:
                idLg.SetHeader(header,"C")
            lg.append(idLg)
        
    return cv, lg
        

def plot_ScaLine(cl, x, y, titles = [""], plt_ops = ["alp"], lims = [0,0,0,0], 
                 pos=1, addlg = [False, ""], lgPos = [], errs=[]):
    
    cl[0].cd(pos)
    try:
        n = x.shape[0]
    except:
        x1 = np.array(x.copy()).astype(float)
        n = x1.shape[0]
        

    if len(errs) == 0:
        plot = ROOT.TGraph(n, x, y)
    else:
        plot = ROOT.TGraphErrors(n, x, y, errs[0], errs[1])
    
    # SET TITLES
    if len(titles)>1:
        plot.SetTitle(titles[0]+";"+titles[1]+";"+titles[2])
    else:
        plot.SetTitle("Graph title;X ;Y ")
        
    plot.GetXaxis().SetLabelSize(0.04)
    plot.GetYaxis().SetLabelSize(0.04)
    plot.GetXaxis().SetTitleSize(0.05)
    plot.GetYaxis().SetTitleSize(0.05)
    gPad.SetLeftMargin(0.15)
    gPad.SetBottomMargin(0.15)

    # SET LIMITS 
    if (lims[0] != 0) or (lims[1] != 0) : # Set x limits:
        plot.GetXaxis().SetRangeUser(lims[0],lims[1])
    if (lims[2] != 0) or (lims[3] != 0) : # Set y limits:
        plot.GetYaxis().SetRangeUser(lims[2],lims[3])
        
    # DRAWING OPTIONS
    # Fixed
    gPad.SetGrid(1)
    # From opts
    if len(plt_ops)==5:
        plot.SetLineColor(plt_ops[1])
        plot.SetLineWidth(plt_ops[2])
        plot.SetMarkerColor(plt_ops[3])
        plot.SetMarkerStyle(plt_ops[4])
    elif  len(plt_ops)==6:
        plot.SetLineColor(plt_ops[1])
        plot.SetLineWidth(plt_ops[2])
        plot.SetMarkerColor(plt_ops[3])
        plot.SetMarkerStyle(plt_ops[4])
        plot.SetMarkerSize(plt_ops[5])

    # DRAW THE PLOT
    plot.Draw(plt_ops[0])
    # Legend:
    if addlg[0] == True:
        cl[1][pos-1].AddEntry(plot, addlg[1], plt_ops[0])
        cl[1][pos-1].Draw()
        # gPad.Update()

        # if len(lgPos) > 0:
        #     cl[1][pos].SetX1NDC(lgPos[0])
        #     cl[1][pos].SetX2NDC(lgPos[0]+lgPos[2])
        #     cl[1][pos].SetY1NDC(lgPos[1])
        #     cl[1][pos].SetY2NDC(lgPos[1]-lgPos[3])
            
        # gPad.Modified()

    return [plot, pos]


def plot_StackPlts(cl, Plots, x, y, titles = [""], plt_ops = ["alp"], lims = [0,0,0,0], 
                 pos=1, addlg = [False, ""], lgPos = [], errs=[]):
    
    cl[0].cd(pos)
    try:
        n = x.shape[0]
    except:
        x1 = np.array(x.copy()).astype(float)
        n = x1.shape[0]
        print("Problem when try x.shape")
        
    try:
        if len(errs) == 0:
            plot = ROOT.TGraph(n, x, y)
        else:
            plot = ROOT.TGraphErrors(n, x, y, errs[0], errs[1])
    except:
        plot = ROOT.TGraph(x, y)
        
    # SET TITLES
    if len(titles)>1:
        plot.SetTitle(titles[0]+";"+titles[1]+";"+titles[2])
    else:
        plot.SetTitle("Graph title;X ;Y ")
        
    plot.GetXaxis().SetLabelSize(0.04)
    plot.GetYaxis().SetLabelSize(0.04)
    plot.GetXaxis().SetTitleSize(0.05)
    plot.GetYaxis().SetTitleSize(0.05)
    gPad.SetLeftMargin(0.15)
    gPad.SetBottomMargin(0.15)

    # SET LIMITS 
    if (lims[0] != 0) or (lims[1] != 0) : # Set x limits:
        plot.GetXaxis().SetRangeUser(lims[0],lims[1])
    if (lims[2] != 0) or (lims[3] != 0) : # Set y limits:
        plot.GetYaxis().SetRangeUser(lims[2],lims[3])
        
    # DRAWING OPTIONS
    # Fixed
    gPad.SetGrid(1)
    # From opts
    if len(plt_ops)==5:
        plot.SetLineColor(plt_ops[1])
        plot.SetLineWidth(plt_ops[2])
        plot.SetMarkerColor(plt_ops[3])
        plot.SetMarkerStyle(plt_ops[4])
    elif  len(plt_ops)==6:
        plot.SetLineColor(plt_ops[1])
        plot.SetLineWidth(plt_ops[2])
        plot.SetMarkerColor(plt_ops[3])
        plot.SetMarkerStyle(plt_ops[4])
        plot.SetMarkerSize(plt_ops[5])
        print("SetMarker")

    # DRAW THE PLOT
    plot.Draw(plt_ops[0])
    # Legend:
    if addlg[0] == True:
        cl[1][pos-1].AddEntry(plot, addlg[1], plt_ops[0])
        cl[1][pos-1].Draw()
        # gPad.Update()

        # if len(lgPos) > 0:
        #     cl[1][pos].SetX1NDC(lgPos[0])
        #     cl[1][pos].SetX2NDC(lgPos[0]+lgPos[2])
        #     cl[1][pos].SetY1NDC(lgPos[1])
        #     cl[1][pos].SetY2NDC(lgPos[1]-lgPos[3])
            
        # gPad.Modified()
        
    if len(Plots) == 0:
        Plots = [plot]
    else:
        Plots.append(plot)

    return [Plots, pos]

def plot_ExpPlot(cl, file, exp, titles = [""], plt_ops = ["alp"], lims = [0,0,0,0], 
                 pos=1, addlg = [False, ""], lgPos = []):
    cl[0].cd(pos)
    
    n = file.ch0.Draw(exp[0], exp[1], exp[2])
    plot = ROOT.TGraph(n, file.ch0.GetV2(), file.ch0.GetV1())
    plot.SetName(titles[0])
    # SET TITLES
    if len(titles)>1:
        plot.SetTitle(titles[0]+";"+titles[1]+";"+titles[2])
    else:
        plot.SetTitle("Graph title;X ;Y ")
        
    plot.GetXaxis().SetLabelSize(0.04)
    plot.GetYaxis().SetLabelSize(0.04)
    plot.GetXaxis().SetTitleSize(0.05)
    plot.GetYaxis().SetTitleSize(0.05)
    gPad.SetLeftMargin(0.15)
    gPad.SetBottomMargin(0.15)
        
    # SET LIMITS 
    if (lims[0] != 0) or (lims[1] != 0) : # Set x limits:
        plot.GetXaxis().SetRangeUser(lims[0],lims[1])
    if (lims[2] != 0) or (lims[3] != 0) : # Set y limits:
        plot.GetYaxis().SetRangeUser(lims[2],lims[3])
        
    # DRAWING OPTIONS
    # Fixed
    gPad.SetGrid(1)
    # From opts
    if len(plt_ops)>1:
        plot.SetLineColor(plt_ops[1])
        plot.SetLineWidth(plt_ops[2])
        plot.SetMarkerColor(plt_ops[3])
        plot.SetMarkerStyle(plt_ops[4])
        
    # DRAW THE PLOT
    plot.Draw(plt_ops[0])
    
    # Legend:
    # if addlg[0] == True:
    #     cl[1][pos-1].AddEntry(plot, addlg[1], plt_ops[0])
    #     cl[1][pos-1].Draw()
    #     # gPad.Update()

        # if len(lgPos) > 0:
        #     cl[1][pos].SetX1NDC(lgPos[0])
        #     cl[1][pos].SetX2NDC(lgPos[0]+lgPos[2])
        #     cl[1][pos].SetY1NDC(lgPos[1])
        #     cl[1][pos].SetY2NDC(lgPos[1]-lgPos[3])
            
        # gPad.Modified()

    return [plot, pos]


def plot_Update(cl, Plot, titles = [""], plt_ops = ["ald"], lims = [0,0,0,0]):
    cl[0].cd(Plot[1])
    # SET TITLES
    if len(titles)>1:
        Plot[0].SetTitle(titles[0]+";"+titles[1]+";"+titles[2])
    else:
        Plot[0].SetTitle("Graph title;X ;Y ")
        
        
    # SET LIMITS 
    if (lims[0] != 0) or (lims[1] != 0) : # Set x limits:
        Plot[0].GetXaxis().SetRangeUser(lims[0],lims[1])
    if (lims[2] != 0) or (lims[3] != 0) : # Set y limits:
        Plot[0].GetYaxis().SetRangeUser(lims[2],lims[3])
            
        
    # DRAWING OPTIONS
    # Fixed
    gPad.SetGrid(1)
    # From opts
    if len(plt_ops)>1:
        Plot[0].SetLineColor(plt_ops[1])
        Plot[0].SetLineWidth(plt_ops[2])
        Plot[0].SetMarkerColor(plt_ops[3])
        Plot[0].SetMarkerStyle(plt_ops[4])
        
    # DRAW THE PLOT
    Plot[0].Draw(plt_ops[0])
    
    return Plot



# plotbox = [xpos, ypos, xlen, ylen, color]
def plot_Fit(cl, Plot, fitFcn="pol1", mrange = [0,0], plotBox = [0,0,0,0,1]): 
    cl[0].cd(Plot[1])
    if mrange[0]==mrange[1]:
        fit = Plot[0].Fit(fitFcn, "S")
    else:
        fit = Plot[0].Fit(fitFcn, "S","",mrange[0],mrange[1])
    gStyle.SetOptFit()
    
    if plotBox[2]*plotBox[3] > 0:
        fit_plt = Plot[0].GetListOfFunctions().FindObject(fitFcn)
        fit_plt.SetLineColor(plotBox[4])
        stats = Plot[0].GetListOfFunctions().FindObject("stats")
        stats.SetTextColor(plotBox[4])
        stats.SetX1NDC(plotBox[0])
        stats.SetX2NDC(plotBox[0]+plotBox[2])
        stats.SetY1NDC(plotBox[1])
        stats.SetY2NDC(plotBox[1]-plotBox[3])
    return fit








