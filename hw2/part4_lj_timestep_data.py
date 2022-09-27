from HW2_solutions import * 
import pandas as pd


if __name__=="__main__":
    for timestep in [0.01, 0.02, 0.04, 0.1, 0.2, 0.4 ]:
        print("running timestep", timestep)
        data = run_lj_solid(timestep=0.01)
        pd.DataFrame(data).to_csv(f"data/timestep{timestep}.csv")
    