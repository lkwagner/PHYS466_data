import pyscf
import pandas as pd
import numpy as np

def run_he(r):
    mol = pyscf.M(
        atom = f'He 0 0 0; He 0 0 {r}',
        basis = 'ccpvqz')

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    et = mycc.ccsd_t()
    print('CCSD(T) energy', mycc.e_tot+et)
    return mycc.e_tot+et

if __name__=="__main__":
    data =[]
    for r in np.linspace(1.1, 6.0, 50):
        data.append({"r":r, "energy":run_he(r)})
    pd.DataFrame(data).to_csv("he_energy.csv", index=False)
