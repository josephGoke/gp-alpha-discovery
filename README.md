# gp-alpha-discovery
A genetic programming system for discovering multi-tree alpha factors from market data, with Pareto Hall of Fame selection and out-of-sample evaluation.


To-do:
    - Rewrite main.jl, Engine.jl 

Future Work(Extension):
    Extend into a multi-layer search system comprising of:
        First discovering single alphas(factors), ensure sustainability using out-of-sample evaluation techniques(e.g MCC, Walk-Forward evaluation...)
        Find possible combinations for alphas from previous stage, also testing for sustainability by filtering bad combinations....
        