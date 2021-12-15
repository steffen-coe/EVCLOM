# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:07:49 2021

@author: steff
"""

import timeit
start = timeit.default_timer()

import classes

classes.init()

my_model = classes.EVChargingLoadModel("config.cfg")

print(my_model)

my_model.run()

my_model.print_results(plot_CMAs=1, save=0)


stop = timeit.default_timer()
print("\nruntime = {0:.2f} s".format(stop-start))
