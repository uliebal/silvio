=======
History
=======

0.1.0 (2021-10-17)
------------------

* First release on PyPI.

0.1.4 (2022-04-07)
------------------

* add catalog with RecExpSim functions in src

0.1.5 (2022-04-07)
------------------

* add __init__.py to catalog folder

0.1.6 (2022-04-07)
------------------

* in RecExperiment: round print failure rate to two decimals
* in RecExperiment.simulate_growth: separate argument progress bar waiting

0.1.7 (2022-05-03)
------------------

* remove requirement cobra

0.1.8 (2022-05-03)
------------------

* remove cobra code dependencies

0.1.8 (2022-05-03)
------------------

* add cobra code dependencies
* remove undelete_gene

0.2.0 (2023-03-29)
------------------

* add GroExpSim, a class to simulate growth experiments

0.2.1 (2023-08-20)
------------------

* add storage of simulated data to Data folder

0.2.2 (2023-09-02)
------------------

* GroExpSim with: 
    * measure_DryWeight: measure the OD to DW conversion factor
    * measure_TemperatureGrowth: measure the growth curve at different temperatures
    * measure_BiomassSubstrateExp: measure the growth curve and substrate concentrations
    * check_Results: check the results of the parameters

0.2.2 (2023-09-02)
------------------

* GroExpSim, nightshift must be within 15h of experiment

0.2.5 (2024-02-22)
------------------

* GroExpSim, export single growth experiments to existing reference excel sheet

0.2.6 (2024-04-23)
------------------

* RecExpSim, add umax argument to 'make' in 'RecHost' for new argument demands of function 'Make_TempGrowthExp' in 'extesions/modules/growth_behaviour.py'