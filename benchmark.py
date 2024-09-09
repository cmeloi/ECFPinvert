import multiprocessing as mp
from search import ECFPInvert
import utils
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import random
import numpy as np
import sys
import argparse
import os
import fcntl
import math
from pathlib import Path


def invert(fp, smi, max_steps, max_time,max_children):
    s, info = inv.run_search(fp, max_steps, max_time, max_children=max_children, miniOutput=True)
    with open(filename, "r+") as file:  # write with locking
        fcntl.lockf(file, fcntl.LOCK_EX)
        file.readlines()
        file.write(
            f'{smi},{Chem.MolToSmiles(info["bestpartial"])},'
            f'{info["searchpath"]},{info["tanimoto"]},'
            f'{info["failurereason"]},{info["time"]}'
            + "\n"
        )
        file.flush()
        fcntl.lockf(file, fcntl.F_UNLCK)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for ECFPinvert")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="input file. should be one smiles per line and nothing else",
        required=True,
    )
    parser.add_argument("-o", "--output", type=str, default="", help="output file")
    parser.add_argument("-r", "--radius", type=int, default=3, help="radius of ECFP")
    parser.add_argument("-l", "--length", type=int, default=4096, help="length of ECFP")
    parser.add_argument(
        "-n",
        "--ncpus",
        type=int,
        default=os.cpu_count(),
        help="number of cpus to run jobs on concurrently",
    )
    parser.add_argument(
        "-c", "--corpus", type=str, default="CHEMBL", help="corpus of atoms to use"
    )
    parser.add_argument(
        "-mt",
        "--maxtime",
        type=float,
        default=100,
        help="maxtime in sec after which to halt an individual search",
    )
    parser.add_argument(
        "-ms",
        "--maxsteps",
        type=int,
        default=1000,
        help="max amount of expansions after which to halt an individual search",
    )
    parser.add_argument(
        "-mc",
        "--maxchildren",
        type=int,
        default=0,
        help="max children per expansion. aka beam width",
    )

    args = vars(parser.parse_args())
    n = args["ncpus"]
    if args["output"] == "":
        inputname = Path(args["input"].split(".")[0]).stem
        filename = f'results/{inputname}_{args["radius"]}_{args["length"]}_{args["corpus"]}_{args["maxtime"]}_{args["maxsteps"]}_{args["maxchildren"]}.txt'
    else:
        filename = args["output"]
    assert args["corpus"] in ["GDB11", "GDB17", "CHEMBL", "FULL"], "mode not allowed"
    utils.initialize_atomtypes(mode=args["corpus"])
    utils.set_fp_settings(args["radius"], args["length"])

    path = Path("results")
    path.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:  # write with locking
        file.write("insmi,outsmi,steps,ts,failure,time" + "\n")
        file.flush()

    inv = ECFPInvert()
    pool = mp.Pool(processes=n, maxtasksperchild=1)
    all_mols = [m for m in Chem.SmilesMolSupplier(args["input"],titleLine=False)]
    fps = [utils.get_fp(m) for m in all_mols]
    all_smis = [Chem.MolToSmiles(m) for m in all_mols]
    all_mols = None
    JOBSIZE = 1000  # make this lower if you run out of RAM. and higher to be more efficient
    for i in range(math.ceil(len(all_smis) / JOBSIZE)):
        begin = i * JOBSIZE
        end = min((i + 1) * JOBSIZE, len(all_smis))
        pool.starmap(
            invert,
            zip(
                fps[begin:end],
                all_smis[begin:end],
                [args["maxsteps"]] * len(fps),
                [args["maxtime"]] * len(fps),
                [args["maxchildren"]] * len(fps),
            ),
        )
    pool.close()
