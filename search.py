from rdkit import Chem, DataStructs
from heapq import heappop, heappush
import time
import utils
import os


class ECFPInvert:
    """
    this class is used for performing A* search to invert ECFP.
    """

    def __init__(self):
        """
        initialize inverter class. this is so the atom types do not have to be
        initialized every time.
        """
        self.reinitialize()

    def reinitialize(self):
        """
        build corpus using functino from utils, get the atoms and their
        invariants. in case you change the corpus (e.g. from chembllike to
        GDB-like) you should call this one.
        """
        self.corpus = utils.ATOMTYPES 
        self.time = time.time()

    def run_search(
        self,
        targetfp,
        max_steps=1000,
        max_time=50,
        max_queue=5000,
        max_children=0,
        visualize_path=False,
        miniOutput=False,
    ):
        """
        this function runs the a-star search.
        remark the Node class has a custom comparator using both depth and
        score. This is what makes this A star search.
        """
        cc = [a for a in self.corpus if utils.get_invariants(a) in targetfp.GetOnBits()]
        queue = [utils.mol_partial(fromFeatures=f, targetfp=targetfp) for f in cc]
        queue = sorted(queue,key=lambda x:x.score)
        searchpath = []
        visited = set([])
        self.time = time.time()
        step = 0
        solution = None
        while (
            step <= max_steps
            and solution == None
            and len(queue) > 0
            and time.time() - self.time < max_time
        ):
            if len(queue) > max_queue:
                queue = queue[
                    :max_queue
                ]  # otherwise there can be catastrophical memory issues
            N = heappop(queue)
            while N.smiles in visited and len(queue) > 0:
                N = heappop(queue)
            N.step.append(step)
            visited.add(N.smiles)
            searchpath.append(N)
            children = N.expand_node(corpus=cc)
            if max_children>0:
                children = sorted(children,key=lambda x:x.score)[:max_children] 
            for child in children:
                heappush(queue, child)
                if len(child.fp)==len(targetfp):
                    if DataStructs.TanimotoSimilarity(targetfp,utils.get_fp(child.mol)) == 1.0:
                        solution = child.mol
                        searchpath.append(child)
                        break
            step += 1
        if solution:
            utils.clean_aromaticity(solution)
        infodigest = {}
        if miniOutput:
            infodigest["searchpath"] = len(searchpath)
        else:
            infodigest["searchpath"] = searchpath  # [n.mol for n in searchpath]
        infodigest["time"] = round(time.time() - self.time, 3)
        if solution:
            infodigest["failurereason"] = "Success"
            infodigest["bestpartial"] = solution
            infodigest["tanimoto"] = 1.0
        else:
            bm, bts = utils.get_best_partial_solution(searchpath, targetfp)
            infodigest["bestpartial"] = bm
            infodigest["tanimoto"] = bts
            if step <= max_steps:
                if infodigest["time"] < max_time:
                    infodigest["failurereason"] = "Not in search path"
                else:
                    infodigest["failurereason"] = "Timeout"
            else:
                infodigest["failurereason"] = "Max steps exceeded"
        if visualize_path == True:
            infodigest["queue"] = queue
        else:
            infodigest["queue"] = None
        return solution, infodigest


if __name__ == "__main__":
    targetmol = Chem.MolFromSmiles(
        "FS(F)(F)(F)(F)c1ccc(C(F)(F)F)cc1 "
    )
    utils.set_fp_settings(3, 4096)
    utils.initialize_atomtypes(mode="GDB17")
    tfp = utils.get_fp(targetmol)
    inv = ECFPInvert()
    s, info = inv.run_search(tfp, 200, 100)
    if s == None:
        print(
            "failure this was the path",
            [Chem.MolToSmiles(g.mol) for g in info["searchpath"]],
        )
        print(
            f'best partial mol was {Chem.MolToSmiles(info["bestpartial"])} with tanimoto of {info["tanimoto"]}'
        )
        print("failed due to:", info["failurereason"])
    else:
        utils.clean_aromaticity(s)
        print(Chem.MolToSmiles(s), len(info["searchpath"]), "steps")
