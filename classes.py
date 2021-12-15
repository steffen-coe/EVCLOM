# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:58:26 2021

@author: Steffen Coenen and Ekin Ugurel
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

import utils as pu

def init():
    import matplotlib
    
    matplotlib.pyplot.rcdefaults()
    matplotlib.pyplot.style.use(u"config/project.mplstyle")
    matplotlib.pyplot.close("all")


class EVChargingLoadModel():
    """
    This is the main class that contains all necessary variables and 
    functions related to the EV charging load model.
    """
    
    def __init__(self, configfile):
        self.configfile = configfile
        
        self.read_config()
        self.def_general_variables()
        
        self.read_base_elec_rate()
        self.read_base_charg_load()
        self.read_target_charg_load()
        
        # print(self.df.head())
    
    def __str__(self):
        return "\nmodel info:\n\tname           = {0:s} \
                \n\tstartdate      = {1} \
                \n\tenddate        = {2} \
                \n\ttime step      = {3:.0f} minute(s) \
                \n\tbase_elec_rate_file    = {4:s} \
                \n\tbase_charg_load_file   = {5:s} \
                \n\ttarget_charg_load_file = {6:s}\n".format(self.name, self.startdate, self.enddate, self.time_step, self.base_elec_rate_file, self.base_charg_load_file, self.target_charg_load_file)
    
    def get_time_from_bin(self, i):
        """
        Returns the datetime object that corresponds to the time bin with index i.

        Parameters
        ----------
        i : datetime
            Time bin index of desired time.
        """
        
        return self.startdate + pd.Timedelta(hours=24*i/self.n_bins_per_day)
    
    def read_config(self):
        """
        Retrieves information from the configuration file (self.configfile) 
        and saves it into the corresponding class variables.
        """
        
        self.cfg = pu.read_config_file(self.configfile)
        cfg = self.cfg["General Info"]
        
        self.name   = cfg["name"]
        self.folder = cfg["folder"] #output folder or similar
        self.seed   = None if cfg["seed"]=="None" else cfg["seed"]
        self.solver = cfg["solver"]
        # self.n_EVs  = cfg["n_EVs"]
        
        self.startdate = pd.to_datetime(cfg["startdate"])
        self.enddate   = pd.to_datetime(cfg["enddate"])
        self.time_step = int(cfg["time_step"]) #[minutes]
        
        self.duration = self.enddate - self.startdate
        self.n_days = int(np.ceil(self.duration.total_seconds()/(60*60*24))) #number of (partial) days
        self.n_bins_per_day = int(24*60/self.time_step) #number of bins in one day
        self.n_bins = int(self.duration.total_seconds() / (self.time_step*60)) #self.n_days * self.n_bins_per_day #number of bins in simulation time frame
        self.bins = np.array(range(self.n_bins))
        self.xticks = [self.startdate + pd.Timedelta(minutes=self.time_step*i) for i in self.bins]
        
        # self.alpha = float(cfg["alpha"]) #response rate between electricity rate change and charging load change
        self.alpha_factor = float(cfg["alpha_factor"]) #factor to be multiplied with alpha_electricity_rate_increases and alpha_electricity_rate_decreases (to potentially enhance the response to price changes)
        self.alpha_electricity_rate_increases = self.alpha_factor * float(cfg["alpha_electricity_rate_increases"])
        self.alpha_electricity_rate_decreases = self.alpha_factor * float(cfg["alpha_electricity_rate_decreases"])
        
        self.tolerance_total_charged_energy_up   = float(cfg["tolerance_total_charged_energy_up"])
        self.tolerance_total_charged_energy_down = float(cfg["tolerance_total_charged_energy_down"])
        
        self.max_electricity_rate_change_up   = float(cfg["max_electricity_rate_change_up"])
        self.max_electricity_rate_change_down = float(cfg["max_electricity_rate_change_down"])
        
        self.mu_plugin, self.sigma_plugin = cfg["mu_plugin"], cfg["sigma_plugin"]
        self.mu_plugout, self.sigma_plugout = cfg["mu_plugout"], cfg["sigma_plugout"]
        
        #baseline electricity rate
        self.base_elec_rate_folder = cfg["base_elec_rate_folder"]
        self.base_elec_rate_file    = cfg["base_elec_rate_file"]
        self.base_elec_rate_scale  = float(cfg["base_elec_rate_scale"])
        
        #baseline charging load
        self.base_charg_load_folder = cfg["base_charg_load_folder"]
        self.base_charg_load_file    = cfg["base_charg_load_file"]
        self.base_charg_load_scale  = float(cfg["base_charg_load_scale"])
        
        #target charging load
        self.target_charg_load_folder = cfg["target_charg_load_folder"]
        self.target_charg_load_file    = cfg["target_charg_load_file"]
        self.target_charg_load_scale  = float(cfg["target_charg_load_scale"])
        
        # cfg[""]
    
    def def_general_variables(self):
        """
        Defines some general class variables, such as the main DataFrame 
        object (self.df) or colors and labels in plots. 
        Is called only once when instantiating the class object.
        """
        
        #df
        self.df = pd.DataFrame(index = self.bins, 
                               columns = ["time", 
                                          "base_elec_rate" , 
                                          "opt_elec_rate", 
                                          # "base_n_EVs_charging", 
                                          # "res_n_EVs_charging", 
                                          "base_charg_load", 
                                          "target_charg_load", 
                                          "res_charg_load"])
        self.df["time"] = [self.get_time_from_bin(i) for i in self.df.index]
        self.df.index += 1 #prepare for pyomo being 1-indexed
        
        # self.datecut = str(self.startdate.date())+"_"+str(self.enddate.date())
        self.datecut = ( str(self.startdate) + "_" + str(self.enddate) ).replace(":","-")
        
        #hardcoded variables
        self.figsize = (8,4) #legend next to plots
        self.x = self.df["time"]
        self.xlabel = "time"
        
        self.labels = {"opt_elec_rate": "Optimized electricity rate", "base_elec_rate": "Baseline electricity rate", "res_charg_load": "Resulting charging load", "base_charg_load": "Baseline charging load", "target_charg_load": "Target charging load", "rel_charg_load": r"$\frac{\mathrm{Resulting\;charging\;load}}{\mathrm{Baseline\;charging\;load}} - 1$"}
        self.labels.update({"CMA_opt_elec_rate": "Optimized electricity rate (CMA)", "CMA_base_elec_rate": "Baseline electricity rate (CMA)", "CMA_res_charg_load": "Resulting charging load (CMA)", "CMA_base_charg_load": "Baseline charging load (CMA)", "CMA_target_charg_load": "Target charging load (CMA)", "CMA_rel_charg_load": r"$\frac{\mathrm{Resulting\;charging\;load}}{\mathrm{Baseline\;charging\;load}} - 1$"})
        
        settings = self.cfg["Plot Settings"]
        self.colors = {"opt_elec_rate": settings["col_OER"], "base_elec_rate": settings["col_BER"], "res_charg_load": settings["col_RCL"], "base_charg_load": settings["col_BCL"], "target_charg_load": settings["col_TCL"], "rel_charg_load": settings["col_rel_charg_load"]}
        self.colors.update({"CMA_opt_elec_rate": settings["col_OER"], "CMA_base_elec_rate": settings["col_BER"], "CMA_res_charg_load": settings["col_RCL"], "CMA_base_charg_load": settings["col_BCL"], "CMA_target_charg_load": settings["col_TCL"], "CMA_rel_charg_load": settings["col_rel_charg_load"]})
        
        self.zorders = {"opt_elec_rate": float(settings["zor_OER"]), "base_elec_rate": float(settings["zor_BER"]), "res_charg_load": float(settings["zor_RCL"]), "base_charg_load": float(settings["zor_BCL"]), "target_charg_load": float(settings["zor_TCL"]), "rel_charg_load": float(settings["zor_RCL"])}
        self.zorders.update({"CMA_opt_elec_rate": float(settings["zor_OER"]), "CMA_base_elec_rate": float(settings["zor_BER"]), "CMA_res_charg_load": float(settings["zor_RCL"]), "CMA_base_charg_load": float(settings["zor_BCL"]), "CMA_target_charg_load": float(settings["zor_TCL"]), "CMA_rel_charg_load": float(settings["zor_RCL"])})
        
        self.elec_rates  = ["opt_elec_rate", "base_elec_rate"]
        self.charg_loads = ["res_charg_load", "base_charg_load", "target_charg_load"]
    
    def read_base_elec_rate(self):
        """
        Reads-in the baseline electricity rate from the respective data file 
        (self.base_elec_rate_file), by time of the day.
        
        Scales the numbers given in that file with self.base_elec_rate_scale.
        """
        
        base_elec_rate = pd.read_csv(self.base_elec_rate_folder + self.base_elec_rate_file, sep=";", dtype={"electricity rate": float})
        base_elec_rate.index += 1 #prepare for pyomo being 1-indexed
        
        if self.time_step > 1:
            for index in self.df.index:
                index2 = index - ((index-1)//self.n_bins_per_day)*self.n_bins_per_day
                self.df.loc[index, "base_elec_rate"] = base_elec_rate.loc[index2*self.time_step, "electricity rate"]
        else:
            # for i in range(self.n_days-1):
            #     base_elec_rate = base_elec_rate.append(base_elec_rate, ignore_index=True)
            # self.df["base_elec_rate"] = base_elec_rate["electricity rate"]
            base_elec_rate = pd.concat([base_elec_rate]*self.n_days, ignore_index=True)
            base_elec_rate.index += 1 #prepare for pyomo being 1-indexed
            self.df["base_elec_rate"] = base_elec_rate.loc[base_elec_rate.index <= self.n_bins, "electricity rate"]
        
        self.df["base_elec_rate"] *= self.base_elec_rate_scale
        
        print("Read baseline electricity rate.")
    
    def read_base_charg_load(self):
        """
        Reads-in the baseline charging load from the respective data file 
        (self.base_charg_load_file), by date and time of the day.
        
        Scales the numbers given in that file with self.base_charg_load_scale.
        Also computes the baseline total charged energy.
        """
        
        base_charg_load = pd.read_csv(self.base_charg_load_folder + self.base_charg_load_file, sep=";", dtype={"charging load": float}, parse_dates=["time"])
        base_charg_load.index += 1 #prepare for pyomo being 1-indexed
        
        if self.time_step > 1:
            for index in self.df.index:
                self.df.loc[index, "base_charg_load"] = base_charg_load.loc[index*self.time_step, "charging load"]
        else:
            self.df["base_charg_load"] = base_charg_load.loc[base_charg_load["time"]<self.enddate, "charging load"]
        
        self.df["base_charg_load"] *= self.base_charg_load_scale
        
        # self.df["res_charg_load"] = self.df["base_charg_load"]#+0.01*np.random.random(self.n_bins) #initialize resulting charging load (will be updated in the optimization process) with baseline charging load
        
        self.base_total_charged_energy = sum(self.df["base_charg_load"])
        
        print("Read baseline charging load.")
    
    def read_target_charg_load(self): #by time of the day
        """
        Reads-in the target charging load from the respective data file 
        (self.target_charg_load_file), by date and time of the day.
        
        Scales the numbers given in that file with self.target_charg_load_scale.
        Also scales the target charging load to equal the baseline total energy demand.
        """
        
        target_charg_load = pd.read_csv(self.target_charg_load_folder + self.target_charg_load_file, sep=";", dtype={"charging load": float})
        target_charg_load.index += 1 #prepare for pyomo being 1-indexed
        
        if self.time_step > 1:
            for index in self.df.index:
                index2 = index - ((index-1)//self.n_bins_per_day)*self.n_bins_per_day
                self.df.loc[index, "target_charg_load"] = target_charg_load.loc[index2*self.time_step, "charging load"]
        else:
            target_charg_load = pd.concat([target_charg_load]*self.n_days, ignore_index=True)
            target_charg_load.index += 1 #prepare for pyomo being 1-indexed
            self.df["target_charg_load"] = target_charg_load.loc[target_charg_load.index <= self.n_bins, "charging load"]
        
        self.df["target_charg_load"] *= self.target_charg_load_scale
        
        #scale target charging load to total energy demand
        self.df["target_charg_load"] *= self.base_total_charged_energy/sum(self.df["target_charg_load"])
        
        print("Read target charging load.")
    
    def run(self):
        """
        Umbrella function to run the optimization model.
        """
        
        #show and save initial dataframe
        # print(self.df)
        # print(self.df.sum())
        self.df.to_csv(self.folder+"df_initial.csv", sep=";")
        
        self.model = pyo.ConcreteModel()
        
        self.model.t = pyo.RangeSet(self.n_bins)
        
        # decision variable
        self.model.opt_elec_rate     = pyo.Var(self.model.t, domain=pyo.NonNegativeReals)
        
        #initialize decision variable's values
        for i in self.model.t:
            self.model.opt_elec_rate[i] = self.df.loc[i,"base_elec_rate"]
            # self.model.opt_elec_rate[i] = 0.1 #+ 0.1*np.random.random()
        
        # view initial decision variables' values
        # self.model.opt_elec_rate.pprint()
        
        #parameters
        #TODO: maybe simplify the initialize argument passed
        self.model.base_elec_rate    = pyo.Param(self.model.t, initialize=pd.to_numeric(self.df["base_elec_rate"]   ).to_dict(), domain=pyo.NonNegativeReals)
        self.model.base_charg_load   = pyo.Param(self.model.t, initialize=pd.to_numeric(self.df["base_charg_load"]  ).to_dict(), domain=pyo.NonNegativeReals)
        self.model.target_charg_load = pyo.Param(self.model.t, initialize=pd.to_numeric(self.df["target_charg_load"]).to_dict(), domain=pyo.NonNegativeReals)
        # self.model.res_charg_load    = pyo.Param(self.model.t, initialize=pd.to_numeric(self.df["res_charg_load"]   ).to_dict(), domain=pyo.NonNegativeReals, mutable=True)
        
        # self.model.alpha = pyo.Param(initialize=self.alpha, domain=pyo.NonNegativeReals)
        alpha_init = {}
        for i in self.model.t:
            alpha_init[i] = self.alpha(i)
        self.alpha = list(alpha_init.values())
        # print(alpha_init)
        self.model.alpha = pyo.Param(self.model.t, initialize=alpha_init, domain=pyo.NonNegativeReals)
        self.model.tolerance_total_charged_energy_up   = self.tolerance_total_charged_energy_up
        self.model.tolerance_total_charged_energy_down = self.tolerance_total_charged_energy_down
        self.model.max_electricity_rate_change_up   = self.max_electricity_rate_change_up
        self.model.max_electricity_rate_change_down = self.max_electricity_rate_change_down
        
        # add objective function to the model
        self.model.OBJ = pyo.Objective(rule=objective)
        
        self.initial_objective = self.model.OBJ()
        print("initial objective = {0:.2f}".format(self.initial_objective))
        
        # add constraints to the model
        self.model.Constraint1 = pyo.Constraint(rule=constraint_sustain_total_charged_energy_up)
        self.model.Constraint2 = pyo.Constraint(rule=constraint_sustain_total_charged_energy_down)
        self.model.Constraint3 = pyo.Constraint(self.model.t, rule=constraint_max_electricity_rate_change_up)
        self.model.Constraint4 = pyo.Constraint(self.model.t, rule=constraint_max_electricity_rate_change_down)
        
        # solve the optimization problem
        print("solver = {0:s}".format(self.solver))
        solution = pyo.SolverFactory(self.solver).solve(self.model)
        # print(solution)
        
        print("Done solving the optimization problem.")
        
        # log feasibility/infeasibility
        # from pyomo.util.infeasible import log_infeasible_constraints
        # import logging
        # log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
        # logging.basicConfig(filename='example.log', level=logging.INFO)
        
        self.final_objective = self.model.OBJ()
        self.obj_ratio = self.final_objective/self.initial_objective
        print("final objective   = {0:.2f}".format(self.final_objective))
        print("objective value change (final/initial-1) = {0:.2f}%".format((self.obj_ratio-1)*100))
        
        # retrieve decision variables' and parameters' final values
        self.df["opt_elec_rate"] = [self.model.opt_elec_rate[i].value for i in self.model.t]
        
        self.df["base_elec_rate"]    = [self.model.base_elec_rate[i] for i in self.model.t]
        self.df["base_charg_load"]   = [self.model.base_charg_load[i] for i in self.model.t]
        self.df["target_charg_load"] = [self.model.target_charg_load[i] for i in self.model.t]
        # self.df["res_charg_load"] = [pyo.value(self.model.res_charg_load[i]) for i in self.model.t]
        # self.df["res_charg_load"] = self.df["base_charg_load"] * (1 - self.model.alpha.value*(self.df["opt_elec_rate"]-self.df["base_elec_rate"])/self.df["base_elec_rate"])
        self.df["res_charg_load"] = self.df["base_charg_load"] * (1 - self.alpha*(self.df["opt_elec_rate"]-self.df["base_elec_rate"])/self.df["base_elec_rate"])
        self.df["rel_charg_load"] = self.df["res_charg_load"]/self.df["base_charg_load"] - 1
        
        #add central moving averages (CMAs) to df
        window = int(self.cfg["CMA"]["CMA_window"])
        min_periods = min(int(self.cfg["CMA"]["CMA_min_periods"]), window)
        std = window
        for column in ["opt_elec_rate", "base_elec_rate", "base_charg_load", "target_charg_load", "res_charg_load", "rel_charg_load"]:
            self.df["CMA_"+column] = self.df[column].rolling(window, min_periods=min_periods, win_type="gaussian", center=True).mean(std=std) #Gaussian-distributed weights within one window
        
        #show and save final dataframe
        # print(self.df)
        # print(self.df.sum())
        self.df.to_csv(self.folder+"df_final.csv", sep=";")
        
        self.res_total_charged_energy = self.df["res_charg_load"].sum()
        print("sum(BCL) = {0:.2f}".format(self.base_total_charged_energy))
        print("sum(RCL) = {0:.2f}".format(self.res_total_charged_energy))
        print("sum(RCL)/sum(BCL) = {0:.4f}".format(self.res_total_charged_energy/self.base_total_charged_energy))
    
    def plot_BCL(self, plot_CMAs=False, save=False):
        fig,ax = plt.subplots()
        
        plot_type = "base_charg_load"
        
        y = plot_type
        ylabel = "Baseline charging load [kW]"
        ylim = (0, None)
        ax.set_xlim(self.startdate, self.startdate+pd.Timedelta(days=1))
        
        if plot_CMAs:
            y = "CMA_" + y
        
        #plot
        ax.plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        ax.set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        # ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #           mode="expand", borderaxespad=0)
        plt.tight_layout()
        ax.grid()
        ax.set_ylim(ylim)
        
        if save:
            alph = "alpha_in={0:.1f}_alpha_de={1:.1f}/".format(self.alpha_electricity_rate_increases, self.alpha_electricity_rate_decreases)
            add = "step={0:d}_BER={1:s}_BCL={2:s}_TCL={3:s}_".format(self.time_step, self.base_elec_rate_file[:-4], self.base_charg_load_file[:-4], self.target_charg_load_file[:-4])
            CMA = "CMA_" if plot_CMAs else ""
            filename = self.folder + self.datecut + "/" + alph + CMA + add + plot_type + "_only" + ".png"
            pu.save_figure(fig, filename)
    
    def plot_charg_loads_rel(self, plot_CMAs=False, save=False):
        fig,axs = plt.subplots(2, 1, sharex=True, figsize=(self.figsize[0], 1.6*self.figsize[1]), gridspec_kw={"height_ratios": [2, 1]})
        
        #plot charging loads
        y_cols = self.charg_loads.copy()
        ylabel = "Charging load [kW]"
        ylim = (0, None)
        
        if plot_CMAs:
            for k in range(len(y_cols)):
                y_cols[k] = "CMA_" + y_cols[k]
        
        #plot
        for y in y_cols:
            axs[0].plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        # axs[1].set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        axs[0].set_ylabel(ylabel)
        axs[0].legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # axs[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #               mode="expand", borderaxespad=0)
        plt.tight_layout()
        # ax.grid()
        axs[0].set_ylim(ylim)
        
        #plot resulting/baseline - 1
        y = "rel_charg_load"
        ylabel = "Resulting/Baseline - 1"
        ylim = (None, None)
        
        if plot_CMAs:
            y = "CMA_" + y
        
        #plot
        axs[1].plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        axs[1].set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        axs[1].set_ylabel(ylabel)
        # axs[1].legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # plt.tight_layout()
        # ax.grid()
        axs[1].set_ylim(ylim)
        
        axs[1].axhline(0, color="grey", alpha=0.6, zorder=-1)
        
        if save:
            alph = "alpha_in={0:.1f}_alpha_de={1:.1f}/".format(self.alpha_electricity_rate_increases, self.alpha_electricity_rate_decreases)
            add = "step={0:d}_BER={1:s}_BCL={2:s}_TCL={3:s}_".format(self.time_step, self.base_elec_rate_file[:-4], self.base_charg_load_file[:-4], self.target_charg_load_file[:-4])
            CMA = "CMA_" if plot_CMAs else ""
            plot_type = "charg_loads_rel"
            filename = self.folder + self.datecut + "/" + alph + CMA + add + plot_type + ".png"
            pu.save_figure(fig, filename)
    
    def plot_all_in_one(self, plot_CMAs=False, save=False):
        fig,axs = plt.subplots(3, 1, sharex=True, figsize=(self.figsize[0], 2*self.figsize[1]), gridspec_kw={"height_ratios": [2, 1, 2]})
        
        #plot charging loads
        y_cols = self.charg_loads.copy()
        ylabel = "Charging load [kW]"
        ylim = (0, None)
        
        if plot_CMAs:
            for k in range(len(y_cols)):
                y_cols[k] = "CMA_" + y_cols[k]
        
        #plot
        for y in y_cols:
            axs[0].plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        # axs[1].set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        axs[0].set_ylabel(ylabel)
        axs[0].legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # axs[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #               mode="expand", borderaxespad=0)
        plt.tight_layout()
        # ax.grid()
        axs[0].set_ylim(ylim)
        
        #plot resulting/baseline - 1
        y = "rel_charg_load"
        ylabel = "Resulting/Baseline - 1"
        ylim = (None, None)
        
        if plot_CMAs:
            y = "CMA_" + y
        
        #plot
        axs[1].plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        axs[1].set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        axs[1].set_ylabel(ylabel)
        # axs[1].legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # plt.tight_layout()
        # ax.grid()
        axs[1].set_ylim(ylim)
        
        axs[1].axhline(0, color="grey", alpha=0.6, zorder=-1)
        
        #plot electricity rates
        y_cols = self.elec_rates.copy()
        ylabel = "Electricity rate [USD/kWh]"
        ylim = (0, None)
        
        if plot_CMAs:
            for k in range(len(y_cols)):
                y_cols[k] = "CMA_" + y_cols[k]
        
        #plot
        for y in y_cols:
            axs[2].plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        # axs[1].set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        axs[2].set_ylabel(ylabel)
        axs[2].legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # axs[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #               mode="expand", borderaxespad=0)
        plt.tight_layout()
        # ax.grid()
        axs[2].set_ylim(ylim)
        
        if save:
            alph = "alpha_in={0:.1f}_alpha_de={1:.1f}/".format(self.alpha_electricity_rate_increases, self.alpha_electricity_rate_decreases)
            add = "step={0:d}_BER={1:s}_BCL={2:s}_TCL={3:s}_".format(self.time_step, self.base_elec_rate_file[:-4], self.base_charg_load_file[:-4], self.target_charg_load_file[:-4])
            CMA = "CMA_" if plot_CMAs else ""
            plot_type = "all"
            filename = self.folder + self.datecut + "/" + alph + CMA + add + plot_type + ".png"
            pu.save_figure(fig, filename)
    
    def plot(self, plot_type="elec_rate", plot_CMAs=False, save=False):
        fig,ax = plt.subplots(figsize=self.figsize)
        
        if plot_type=="elec_rates":
            y_cols = self.elec_rates.copy()
            ylabel = "Electricity rate [USD/kWh]"
            ylim = (0, None)
        elif plot_type=="charg_loads":
            y_cols2 = self.charg_loads.copy()
            y_cols = y_cols2
            ylabel = "Charging load [kW]"
            ylim = (0, None)
        elif plot_type=="rel_charg_load":
            y_cols = [plot_type]
            ylabel = "Resulting/Baseline - 1"
            ylim = (None, None)
        
        if plot_CMAs:
            for k in range(len(y_cols)):
                y_cols[k] = "CMA_" + y_cols[k]
        
        #plot
        for y in y_cols:
            ax.plot(self.x, self.df[y], label=self.labels[y], color=self.colors[y], zorder=self.zorders[y])
        
        ax.set_xlabel(self.xlabel)
        plt.xticks(rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        # ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #           mode="expand", borderaxespad=0)
        plt.tight_layout()
        # ax.grid()
        ax.set_ylim(ylim)
        
        if plot_type=="rel_charg_load":
            ax.axhline(0, color="grey", alpha=0.6, zorder=-1)
            ax.get_legend().remove()
        
        if save:
            alph = "alpha_in={0:.1f}_alpha_de={1:.1f}/".format(self.alpha_electricity_rate_increases, self.alpha_electricity_rate_decreases)
            add = "step={0:d}_BER={1:s}_BCL={2:s}_TCL={3:s}_".format(self.time_step, self.base_elec_rate_file[:-4], self.base_charg_load_file[:-4], self.target_charg_load_file[:-4])
            CMA = "CMA_" if plot_CMAs else ""
            filename = self.folder + self.datecut + "/" + alph + CMA + add + plot_type + ".png"
            pu.save_figure(fig, filename)
    
    def print_results(self, plot_CMAs=False, save=False):
        self.plot_BCL(plot_CMAs, save)
        self.plot("elec_rates", plot_CMAs, save)
        self.plot("charg_loads", plot_CMAs, save)
        self.plot("rel_charg_load", plot_CMAs, save)
        self.plot_charg_loads_rel(plot_CMAs, save)
        self.plot_all_in_one(plot_CMAs, save)
    
    # def alpha(self, BCL, TCL):
    def alpha(self, i):
        BCL = self.model.base_charg_load[i]
        TCL = self.model.target_charg_load[i]
        cond = BCL >= TCL
        
        if cond: #BCL>=TCL, OER>=BER
            alpha = self.alpha_electricity_rate_increases #0.2
        else: #BCL<TCL, OER<BER
            alpha = self.alpha_electricity_rate_decreases #1.0
        return alpha

# def RCL(model, i):
#     res_charg_load = model.base_charg_load[i] * (1 - model.alpha*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
#     return res_charg_load

# def RCL(alpha, BCL, OER, BER):
    # return BCL * (1 - alpha*((OER-BER)/BER))

def objective(model):
    """
    The objective function of the model.

    Parameters
    ----------
    model : pyomo model object
        The pyomo model that the optimization shall be run on.

    Returns
    -------
    dev : float
        The summed squared deviation between the resulting charging load 
        (which is RCL = BCL * (1 - alpha*((OER-BER)/BER)) at each bin) and 
        the target charging load.
    """
    
    dev = 0
    for i in model.t:
        # model.res_charg_load[i] = model.base_charg_load[i] * (1 - model.alpha*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
        # dev += (model.res_charg_load[i] - model.target_charg_load[i])**2
        
        # dev += (model.base_charg_load[i] * (1 - model.alpha*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i]) - model.target_charg_load[i])**2
        dev += (model.base_charg_load[i] * (1 - model.alpha[i]*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i]) - model.target_charg_load[i])**2
    
    return dev

def constraint_sustain_total_charged_energy_up(model):
    """
    Constraints the relative change of the total charged energy to the 
    baseline total charged energy to be at most 
    1 + model.tolerance_total_charged_energy_up.
    """
    
    base_total_charged_energy = 0
    res_total_charged_energy  = 0
    for i in model.t:
        base_total_charged_energy += model.base_charg_load[i]
        # res_total_charged_energy  += model.base_charg_load[i] * (1 - model.alpha*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
        res_total_charged_energy  += model.base_charg_load[i] * (1 - model.alpha[i]*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
    
    frac = res_total_charged_energy/base_total_charged_energy
    
    return frac <= 1+model.tolerance_total_charged_energy_up

def constraint_sustain_total_charged_energy_down(model):
    """
    Constraints the relative change of the total charged energy to the 
    baseline total charged energy to be at least 
    1 - model.tolerance_total_charged_energy_down.
    """
    
    base_total_charged_energy = 0
    res_total_charged_energy  = 0
    for i in model.t:
        base_total_charged_energy += model.base_charg_load[i]
        # res_total_charged_energy  += model.base_charg_load[i] * (1 - model.alpha*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
        res_total_charged_energy  += model.base_charg_load[i] * (1 - model.alpha[i]*(model.opt_elec_rate[i]-model.base_elec_rate[i])/model.base_elec_rate[i])
    
    frac = res_total_charged_energy/base_total_charged_energy
    
    return frac >= 1-model.tolerance_total_charged_energy_down

def constraint_max_electricity_rate_change_up(model, i):
    """
    Constraints the electricity rate *increase* induced by the model to a 
    certain amount above the baseline electricity price.
    """
    
    return (model.opt_elec_rate[i] - model.base_elec_rate[i]) <= model.max_electricity_rate_change_up

def constraint_max_electricity_rate_change_down(model, i):
    """
    Constraints the electricity rate *decrease* induced by the model to a 
    certain amount above the baseline electricity price.
    """
    
    return -(model.opt_elec_rate[i] - model.base_elec_rate[i]) <= model.max_electricity_rate_change_down
