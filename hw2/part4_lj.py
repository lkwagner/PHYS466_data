from HW2_solutions import * 
import pandas as pd


if __name__=="__main__":
    data = run_lj_solid(timestep=0.01)
    pd.DataFrame(data).to_csv("data/first_run.csv")
    