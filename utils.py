from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, PeriodicTable, rdqueries
import math, itertools
import copy
from copy import deepcopy

RDLogger.DisableLog("rdApp.*")


RADIUS = 3
FPSIZE = 4096
MODE = "FULL"

SCORE_WEIGHTS = [0.44294695077527485,1.6796863000150526,1.062249537151004,0.20668299539294513] #found by optuna
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE)
AO = rdFingerprintGenerator.AdditionalOutput()
AO.AllocateBitInfoMap()

DUMMY_PATTERN = rdqueries.AtomNumEqualsQueryAtom(0)
GET_ATOM_DICT = {}


BONDDICT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    1.5: Chem.BondType.AROMATIC,
}

PT = Chem.GetPeriodicTable()

FEATURES2ATOM = {}

ACD = {(0, 0): [999999]}  # dummy can have any max valence

GDBSET = [
    (6, 0, 0),
    (7, 0, 0),
    (7, 1, 0),
    (8, 0, 0),
    (8, -1, 0),
    (9, 0, 0),
]  # CNOF and charged N and O included because it has nitro groups
GDB17SET = GDBSET + [
    (16, 0, 0),
    (16, 1, 0),
    (17, 0, 0),
    (35, 0, 0),
    (53, 0, 0),
]  # Cl, Br, I, S, P, Si added
FULLSET = GDB17SET + [(14, 0, 0), (15, 0, 0)]  # P, Si added


CHEMB_OCCURENCE= [
    ((6, 2, 1, 1, 0, 0), 32259076),
    ((6, 3, 0, 1, 0, 0), 24725966),
    ((6, 2, 2, 0, 0, 0), 9587554),
    ((6, 1, 3, 0, 0, 0), 9471398),
    ((6, 2, 2, 1, 0, 0), 8736169),
    ((8, 1, 0, 0, 0, 0), 8396927),
    ((6, 3, 0, 0, 0, 0), 5458055),
    ((8, 2, 0, 0, 0, 0), 3608160),
    ((6, 3, 1, 1, 0, 0), 3585255),
    ((7, 2, 1, 0, 0, 0), 3265879),
    ((7, 3, 0, 1, 0, 0), 3254091),
    ((7, 2, 0, 1, 0, 0), 3111331),
    ((6, 3, 1, 0, 0, 0), 1965361),
    ((8, 1, 1, 0, 0, 0), 1692311),
    ((9, 1, 0, 0, 0, 0), 1691386),
    ((8, 2, 0, 1, 0, 0), 1571464),
    ((17, 1, 0, 0, 0, 0), 1243946),
    ((6, 2, 1, 0, 0, 0), 1093706),
    ((16, 2, 0, 1, 0, 0), 917074),
    ((6, 4, 0, 1, 0, 0), 797743),
    ((7, 2, 1, 1, 0, 0), 744177),
    ((6, 4, 0, 0, 0, 0), 733393),
    ((7, 3, 0, 0, 0, 0), 732848),
    ((7, 1, 2, 0, 0, 0), 680971),
    ((16, 4, 0, 0, 0, 0), 569988),
    ((16, 2, 0, 0, 0, 0), 507740),
    ((6, 2, 0, 0, 0, 0), 387806),
    ((8, 1, 0, 0, -1, 0), 357569),
    ((35, 1, 0, 0, 0, 0), 357100),
    ((7, 2, 0, 0, 0, 0), 351099),
    ((7, 3, 0, 0, 1, 0), 287691),
    ((7, 1, 0, 0, 0, 0), 265315),
    ((6, 1, 2, 0, 0, 0), 226632),
    ((16, 1, 0, 0, 0, 0), 178634),
    ((14, 4, 0, 0, 0, 0), 77139),
    ((53, 1, 0, 0, 0, 0), 58219),
    ((15, 4, 0, 0, 0, 0), 58186),
    ((16, 4, 0, 1, 0, 0), 51140),
    ((7, 3, 1, 1, 1, 0), 41949),
    ((7, 3, 0, 1, 1, 0), 41123),
    ((7, 1, 1, 0, 0, 0), 30982),
    ((16, 1, 1, 0, 0, 0), 29981),
    ((6, 1, 1, 0, 0, 0), 28404),
    ((7, 2, 0, 0, 1, 0), 19041),
    ((7, 3, 1, 0, 1, 0), 17102),
    ((16, 3, 0, 0, 0, 0), 14745),
    ((7, 1, 0, 0, -1, 0), 14203),
    ((7, 4, 0, 0, 1, 0), 11334),
    ((7, 2, 2, 0, 1, 0), 10589),
    ((7, 4, 0, 1, 1, 0), 8306),
    ((15, 3, 0, 0, 0, 0), 7954),
    ((15, 4, 0, 1, 0, 0), 6913),
    ((5, 3, 0, 0, 0, 0), 6603),
    ((7, 1, 3, 0, 1, 0), 6197),
    ((5, 3, 0, 1, 0, 0), 6133),
    ((7, 2, 1, 1, 1, 0), 5704),
    ((14, 4, 0, 1, 0, 0), 5104),
    ((7, 2, 2, 1, 1, 0), 4419),
    ((34, 2, 0, 0, 0, 0), 4106),
    ((50, 4, 0, 0, 0, 0), 3585),
    ((16, 3, 0, 1, 0, 0), 3552),
    ((14, 3, 0, 0, 0, 0), 3101),
    ((6, 1, 0, 0, -1, 0), 3040),
    ((6, 2, 0, 1, 0, 0), 2720),
    ((15, 3, 0, 1, 0, 0), 2624),
    ((34, 2, 0, 1, 0, 0), 2593),
    ((14, 2, 0, 0, 0, 0), 2309),
    ((15, 1, 2, 0, 0, 0), 2290),
    ((15, 4, 0, 0, 1, 0), 2220),
    ((14, 1, 0, 0, 0, 0), 1854),
    ((16, 3, 0, 0, 1, 0), 1771),
    ((15, 3, 0, 0, 1, 0), 1746),
    ((5, 1, 0, 0, 0, 0), 1628),
    ((15, 2, 1, 0, 0, 0), 1386),
    ((7, 2, 1, 0, 1, 0), 1373),
    ((16, 1, 0, 0, -1, 0), 1272),
    ((34, 1, 0, 0, 0, 0), 972),
    ((32, 4, 0, 0, 0, 0), 811),
    ((80, 2, 0, 0, 0, 0), 791),
    ((7, 2, 0, 0, -1, 0), 752),
    ((16, 3, 0, 1, 1, 0), 751),
    ((15, 2, 0, 0, 0, 0), 711),
    ((52, 2, 0, 0, 0, 0), 661),
    ((8, 2, 0, 1, 1, 0), 644),
    ((14, 3, 0, 1, 0, 0), 603),
    ((15, 5, 0, 1, 0, 0), 549),
    ((5, 2, 0, 0, 0, 0), 530),
    ((15, 5, 0, 0, 0, 0), 526),
    ((5, 4, 0, 0, -1, 0), 509),
    ((13, 3, 0, 0, 0, 0), 505),
    ((7, 1, 2, 0, 1, 0), 472),
    ((33, 3, 0, 0, 0, 0), 447),
    ((6, 1, 2, 0, -1, 0), 435),
    ((33, 4, 0, 0, 0, 0), 379),
    ((7, 2, 0, 1, -1, 0), 377),
    ((5, 4, 0, 1, -1, 0), 356),
    ((16, 6, 0, 0, 0, 0), 337),
    ((15, 2, 0, 1, 0, 0), 304),
    ((15, 4, 0, 1, 1, 0), 66),
    ((15, 1, 3, 0, 1, 0), 33),
    ((80, 1, 0, 0, 0, 0), 32),
    ((53, 2, 0, 0, 1, 0), 8),
    ((34, 1, 0, 0, -1, 0), 1),
    ((13, 2, 0, 0, -1, 0), 1),
    ((15, 1, 0, 0, 0, 0), 1),
]

CHEMBL_ATOMTYPES = [f[0] for f in CHEMB_OCCURENCE if f[1]>1000]

ALLOWED_ATOMS = []
ATOMTYPES = []


W_ATOM = Chem.MolFromSmiles("[W]")


class mol_partial:
    """
    partial mol class. this corresponds to a node on the search tree. dummy
    atoms [*] stand for unresolved atoms in the intermediary molecule.
    """

    def __init__(
        self,
        mol=None,
        fromFeatures=None,
        targetfp=None,
        depth=0,
        atomdepth=0,
        scoreOnInit=True,
        step=[-1],
    ):
        self.mol = mol
        if fromFeatures:
            self.mol = get_atom(fromFeatures, oneLessDummy=False)
        self.fp = get_nodummies_ecfp(self.mol)
        self.smiles = Chem.MolToSmiles(self.mol)
        self.target_fp = targetfp
        if scoreOnInit:  # for minor performance gain on extend
            self.score = scores([self.mol], self.target_fp)[0][0]
        self.depth = depth
        self.atomdepth = atomdepth
        self.step = step

    def __lt__(self, other):
        """
        "less" comparison function to allow heapq to sort
        """
        if self.score + self.atomdepth < other.score + other.atomdepth:
            return True
        else:
            return False

    def __le__(self, other):
        """
        "less or equal" comparison function to allow heapq to sort
        """
        if self.score + self.atomdepth <= other.score + other.atomdepth:
            return True
        else:
            return False

    def expand_node(self, corpus=[]):
        """
        function to add new atoms on a partial molecule
        """
        new_mols = []
        c, b, d = get_dummy_info(self.mol)
        if len(c) > 0:
            current_atom = self.mol.GetAtomWithIdx(c[0])
            dummycount = get_dummy_count(current_atom)
            cf = features_from_atom(current_atom)
            fullcorpus = corpus + count_ringclosure_opportunities(self.mol, c)
            all_strs = get_atom_bond_tuples(cf, dummies=dummycount, corpus=fullcorpus)
            for abt in all_strs:
                new_mol = mol_partial(
                    mol=W_ATOM,
                    targetfp=self.target_fp,
                    depth=self.depth,
                    atomdepth=self.atomdepth,
                    scoreOnInit=False,
                    step=copy.copy(self.step),
                )
                new_mol.mol = extend_mol(
                    self.mol, [t[0] for t in abt], c, b, d, [t[1] for t in abt]
                )
                new_mol.smiles = Chem.MolToSmiles(new_mol.mol)
                new_mol.depth += 1
                new_mol.atomdepth += len(abt)
                new_mols.append(new_mol)
        nm_scores, nm_fps = scores([x.mol for x in new_mols], self.target_fp, [get_invariants(a) for a in corpus])
        new_mols2 = []
        for i, nmscore in enumerate(nm_scores):
            if nmscore < 99999999999:
                new_mols[i].score = nmscore
                new_mols[i].fp = nm_fps[i]
                new_mols2.append(new_mols[i])
        new_mols = new_mols2
        return new_mols


def acd(atomicnum, charge):
    """
    given an atomic number and charge, this gives the maximal valence.
    note that this can give overestimate e.g. carbocation will have max valence
    of 3 in reality but this function will give 5.
    """
    if (atomicnum, charge) not in ACD:
        ACD[(atomicnum, charge)] = [
            valence + charge for valence in PT.GetValenceList(atomicnum)
        ]
    return ACD[(atomicnum, charge)]


def remove_dummies(searchpath):
    """
    given a search path, remove dummies and close rings. this can be used to
    rank partial solutions for the user
    """
    resolved_mols = []
    for pm in searchpath:
        mol = pm.mol
        dummies = mol.GetAtomsMatchingQuery(DUMMY_PATTERN)
        dummies_idx = [a.GetIdx() for a in dummies]
        if len(dummies) > 0:
            for d in sorted(dummies_idx)[::-1]:
                mol.ReplaceAtom(d, Chem.MolFromSmiles("[H]").GetAtoms()[0])
            for b in mol.GetBonds():
                if str(b.GetBondType()) == "UNSPECIFIED":
                    b.SetBondType(Chem.BondType.SINGLE)
        try:
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            clean_aromaticity(mol)
            resolved_mols.append(mol)
        except:
            pass
    return resolved_mols


def unlikely_hypervalent(features_list):
    """
    hypervalent atoms (considered here to be S,P and halogens) tend to not
    have hydrogens connected. so these get filtered.
    also, limit halogen degree to 3.
    also, if in ring, set degree to max 4
    set degree max to 6
    """

    pruned_features_list = []
    for features in features_list:
        valencelist = PT.GetValenceList(features[0])
        n,d,h,inring,charge,isotope = features
        if len(valencelist) > 1:
            if d + h > min(valencelist) and h > 0: #unlikely hydrogen containnig atoms
                pass
            else:
                if n in [53, 35, 17] and d + h > 3: #hypervalent halogens
                    pass
                else:
                    if d * inring>4 or d>6: # heptavalence and hypervalent ringatom
                        pass
                    else:
                        if n==14 and d > 4: # hypervalent Si 
                            pass
                        else:
                            pruned_features_list.append(features)
        else:
            pruned_features_list.append(features)
    return pruned_features_list


def get_best_partial_solution(searchpath, targetfp):
    """
    helper function to get partial solution in case reconstruction fails.
    """
    resolved_mols = remove_dummies(searchpath)
    resolved_fps = [MFPGEN.GetFingerprint(m) for m in resolved_mols]
    best_TS = 0
    if len(resolved_mols) > 0:
        TSs = DataStructs.BulkTanimotoSimilarity(targetfp, resolved_fps)
        best_TS, best_partial = sorted(
            zip(TSs, resolved_mols), key=lambda pair: pair[0]
        )[-1]
    else:
        best_partial = None
    try:
        clean_aromaticity(best_partial)
    except:
        pass
    return best_partial, best_TS


def features_to_atom(features):
    """
    quick way to convert a tuple or list of features to the corresponding ecfp0
    atom
    """
    if features not in FEATURES2ATOM:
        FEATURES2ATOM[features] = get_atom(features)
    return FEATURES2ATOM[features]


def count_ringclosure_opportunities(mol, current_atoms):
    """
    to check which ring closures need to be added and scored
    """
    rco = []
    if (
        features_from_atom(mol.GetAtomWithIdx(current_atoms[0]))[3] == 1
        and len(current_atoms) > 1
    ):
        for j, a in enumerate(current_atoms[1:]):
            if features_from_atom(mol.GetAtomWithIdx(current_atoms[j + 1]))[3] == 1:
                rco.append((0, 0, 0, 1, 0, -1 - j))
    return rco


def get_nodummies_ecfp(mol, extra_info=False):
    """
    this function calculates the ECFP for a given molecule but removes any
    fragment with a *, ie a dummy, in there
    """
    try:
        fp = MFPGEN.GetFingerprint(mol, additionalOutput=AO)
        bim = AO.GetBitInfoMap()
        dcmol = Chem.Mol(mol)  # deepcopy(mol)
        try:
            Chem.SanitizeMol(dcmol)
        except Exception as e:
            flags = Chem.SanitizeFlags.SANITIZE_CLEANUP
            try:
                Chem.SanitizeMol(dcmol, sanitizeOps=flags)
            except Exception as e2:
                print("sanitization error:", e2)
                print(Chem.MolToSmiles(dcmol))
        tdm = Chem.GetDistanceMatrix(
            dcmol
        )  # done on copied mol because unsanitized rwmol has problem with dm
        dummies = mol.GetAtomsMatchingQuery(DUMMY_PATTERN)
        dummies_idx = [
            dummies.__getitem__(i).GetIdx() for i in range(len(dummies))
        ]  # faster this way ...
        dummy_neighbors = {}
        for i in range(0, RADIUS + 1):
            dummy_neighbors[i] = []
        for d_idx in dummies_idx:
            for i, d in enumerate(tdm[d_idx]):
                if d <= RADIUS:
                    u = d
                    while u <= RADIUS:
                        dummy_neighbors[u].append(i)
                        u += 1
        newfp = DataStructs.ExplicitBitVect(len(fp))
        onbits = set([])
        higher_radius=[]
        zero_radius=[]
        for bit in bim:
            for bit_tuple in bim[bit]:
                if bit_tuple[0] not in dummy_neighbors[bit_tuple[1]]:
                    onbits.add(bit)
                    if bit_tuple[1] > 0:
                        higher_radius.append(bit)
                    else:
                        zero_radius.append(bit)
        higher_radius = len(list(set(higher_radius).difference(set(zero_radius)))) #count unique bits only
        for bit in onbits:
            newfp.SetBit(bit)
    except Exception as e:
        higher_radius = 0
        print("no dummies ECFP failed with exception", e)
        newfp = MFPGEN.GetFingerprint(Chem.MolFromSmiles(""), additionalOutput=AO)
        newfp.SetBitsFromList([i for i in range(len(fp))])
    if extra_info:
        return newfp, higher_radius
    else:
        return newfp


def get_atom(features, oneLessDummy=True):
    if oneLessDummy:
        smi = (
            f'[{"" if features[5]==0 else int(PT.GetMostCommonIsotopeMass(features[0])+features[5])}{PT.GetElementSymbol(features[0])}H{features[2]}+{features[4]}]'
            + "(*)" * (features[1] - 1)
        )
    else:
        smi = (
            f'[{"" if features[5]==0 else int(PT.GetMostCommonIsotopeMass(features[0])+features[5])}{PT.GetElementSymbol(features[0])}H{features[2]}+{features[4]}]'
            + "(*)" * features[1]
        )
    smi = smi.replace(
        "+-", "-"
    )  # otherwise above smi doesnt parse for negatively charged atoms
    m = Chem.MolFromSmiles(smi)
    m = Chem.RWMol(m)
    if features[3] == 1:
        ri = m.GetRingInfo()
        ri.AddRing((0,), (0,))
    m.GetAtoms()[0].SetNumRadicalElectrons(0)
    return m


def get_dummy_info(mol):
    """
    get info about:
    which atoms are connected to dummies (curr_atoms)
    which bonds are connected to dummies (bonds)
    which atoms are dummies (dummies_idx)
    """
    dummies = mol.GetAtomsMatchingQuery(DUMMY_PATTERN)
    bonds = [a.GetBonds()[0].GetIdx() for a in dummies]
    dummies_idx = [a.GetIdx() for a in dummies]
    curr_atoms = sorted(list({a.GetNeighbors()[0].GetIdx() for a in dummies}))
    return curr_atoms, bonds, dummies_idx


def get_dummy_count(atom):
    """
    returns the amount of dummies adjacent to the input atom
    """
    return len([a for a in atom.GetNeighbors() if a.GetAtomicNum() == 0])

    
def get_onbit_counts(tfp,mol,corpus_onbits):
    """
    alternative to nodummies ecfp to get onbit counts with more info to put
    in the heuristic. this is useful for the scoring function
    """
    fp = MFPGEN.GetFingerprint(mol, additionalOutput=AO)
    bim = AO.GetBitInfoMap()
    dcmol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(dcmol)
    except Exception as e:
        flags = Chem.SanitizeFlags.SANITIZE_CLEANUP
        try:
            Chem.SanitizeMol(dcmol, sanitizeOps=flags)
        except Exception as e2:
            print("sanitization error:", e2)
            print(Chem.MolToSmiles(dcmol))
    tdm = Chem.GetDistanceMatrix(dcmol)
    # done on copied mol because unsanitized rwmol has problem with dm
    dummies = mol.GetAtomsMatchingQuery(DUMMY_PATTERN)
    dummies_idx = [
        dummies.__getitem__(i).GetIdx() for i in range(len(dummies))
    ]  # faster this way ...
    dummy_neighbors = {}
    for i in range(0, RADIUS + 1):
        dummy_neighbors[i] = []
    for d_idx in dummies_idx:
        for i, d in enumerate(tdm[d_idx]):
            if d <= RADIUS:
                u = d
                while u <= RADIUS:
                    dummy_neighbors[u].append(i)
                    u += 1
    onbits0 = set([])
    onbits = set([])
    collisions = set([])
    for bit in bim:
        for bit_tuple in bim[bit]:
            if bit_tuple[0] not in dummy_neighbors[bit_tuple[1]]:
                if bit_tuple[1] == 0:
                    onbits0.add(bit)
                else:
                    if bit in corpus_onbits:
                        collisions.add(bit)
                    else:
                        onbits.add(bit)
    collisions = collisions.difference(onbits0) # to not count onbits double
    return len(list(onbits0)), len(list(onbits)), len(list(collisions))

def scores(partial_mols, targetfp,corpusbits=[]):
    """
    better scoring.
    by penalizing higher radius fragments that have a corresponding bit that is
    also in the atom corpus, unproductive paths are discouraged
    """
    scores = []
    tonbits = targetfp.GetOnBits()
    fps = [get_nodummies_ecfp(m) for m in partial_mols]
    w=SCORE_WEIGHTS
    for i, pm in enumerate(partial_mols):
        nonmatchingbits = 0
        for onbit in fps[i].GetOnBits():
            if onbit not in tonbits:
                nonmatchingbits += 1
        if nonmatchingbits == 0:
            dummies = partial_mols[i].GetAtomsMatchingQuery(DUMMY_PATTERN)
            o1,o2,o3 = get_onbit_counts(targetfp,pm,corpusbits)
            score = len(tonbits) - w[0] * o1 - w[1] * o2 - w[2] * o3
            score += w[3] * len(dummies) ** 2
        else:
            score = 99999999999

        scores.append(score)
   
    return scores, fps





def generate_atom_types(allowed_atoms=ALLOWED_ATOMS):
    """
    generate atom types from input dict which has (atomic num,charge, isotope)
    as the keys and the unsaturated degree as the entry. isotope is always set
    to 0 in this workflow. The output is atomtypes in the form:

    (atomicnum,degree,hydrogens,isinring,charge,isotope)

    which are ready for generating radius 0 ecfp from.
    """
    invariants = []
    for aa in allowed_atoms:
        degrees = acd(aa[0], aa[1])
        atomn = aa[0]
        charge = aa[1]
        isotope = aa[2]
        for degree in degrees:
            for i in range(1, degree + 1):
                invariants.append([atomn, i, degree - i, 0, charge, isotope])
                if i > 1 and degree - i >= 0:  # one unsaturation
                    invariants.append([atomn, i - 1, degree - i, 0, charge, isotope])
                if i > 2 and degree - i >= 0:  # two unsaturation
                    invariants.append([atomn, i - 2, degree - i, 0, charge, isotope])
    ring_invariants = [inv[:3] + [1] + inv[4:] for inv in invariants if inv[1] > 1]
    invariants += ring_invariants
    invariants = unlikely_hypervalent([tuple(inv) for inv in invariants])
    return list(set(invariants))


def get_invariants(invariant):
    """
    input atom type in the form:

        [atomicnum,degree,hydrogens,isinring,charge,isotope]

    get radius 0 invariants corresponding to the atoms
    """
    m = get_atom(invariant, oneLessDummy=False)
    inv = MFPGEN.GetFingerprint(m, fromAtoms=[0], additionalOutput=AO)
    bim = AO.GetBitInfoMap()
    for k in bim:
        for l in bim[k]:
            if l[1] == 0:
                inv = k
    return inv


def get_atom_bond_tuples(startatom, dummies, corpus, maxLength=999999):
    """
    given an atom and how much unresolved dummies it has, output all possible
    coombinations of atom types + bond orders that are allowed.
    """
    if startatom[3] == 0:
        corpus = [x for x in corpus if x[3] == 0 or (x[3] == 1 and x[1] > 2)]
    else:
        if startatom[1] <= 2:
            corpus = [x for x in corpus if x[3] == 1]
    ab_combos = [x for x in itertools.product(corpus, [k for k in BONDDICT])]
    if dummies<4:
        all_strs = [
            x
            for x in itertools.combinations_with_replacement(ab_combos, dummies)
            if sum([math.floor(y[1]) for y in x])
            <= max(acd(startatom[0], startatom[4])) - startatom[1] + dummies
        ] 
    else:
        #high dummies. introduce to not have probs with pent and hexvalent P S
        all_strs = [
            [x]*dummies
            for x in ab_combos
        ]
    if len(all_strs) > maxLength:
        all_strs = all_strs[:maxLength]

    return all_strs


def features_from_atom(atom):
    """
    convenience function for getting the 6 ECFP atom features from an atom
    """
    features = [atom.GetAtomicNum()]
    features.append(atom.GetDegree())
    features.append(atom.GetNumImplicitHs() + atom.GetNumExplicitHs())
    features.append(int(atom.IsInRing()))
    features.append(atom.GetFormalCharge())
    features.append(0)  # TODO
    return features


def clean_aromaticity(mol):
    """the aromatic status of atoms is not taken into account during ECFP
    calculation. in order to clean up final molecules, this function will
    turn every atom with at least one aromatic bond aromatic.
    and maybe a ring which has only aromatic atoms should also have its bonds
    turned aromatic, let me think about this"""
    for i, a in enumerate(mol.GetAtoms()):
        for n in a.GetBonds():
            if str(n.GetBondType()) == "AROMATIC":
                a.SetIsAromatic(True)
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetIsAromatic() and not a.IsInRing():
            for n in a.GetBonds():
                if str(n.GetBondType()) == "AROMATIC":
                    n.SetBondType(Chem.BondType.SINGLE)
            a.SetIsAromatic(False)
    try:
        for r in Chem.GetSSSR(mol):
            if len(r) == sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
                for i, idx in enumerate(r):
                    mol.GetBondBetweenAtoms(idx, r[(i + 1) % len(r)]).SetBondType(
                        Chem.BondType.AROMATIC
                    )
    except Exception as e:
        pass
    global AO
    global MFPGEN
    MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE)
    AO = rdFingerprintGenerator.AdditionalOutput()
    AO.AllocateBitInfoMap()


def extend_mol(mol, atoms, curr_atoms, dummies_bonds_idx, dummies_idx, bondtypes):
    """
    function for substitution dummies with new atoms and ring closing
    operations
    """
    newmol = Chem.RWMol(mol)  # deepcopy(mol)
    newmol = Chem.RWMol(newmol)
    error = False
    ri = newmol.GetRingInfo()
    idx_to_remove = []
    inring_atoms = []
    for i, f in enumerate(atoms):
        if f[5] >= 0:
            acount = len(newmol.GetAtoms())
            at = features_to_atom(f)
            at_atoms = [a for a in at.GetAtoms()]
            newmol.ReplaceAtom(dummies_idx[i], at_atoms[0])
            newmol.GetBondWithIdx(dummies_bonds_idx[i]).SetBondType(
                BONDDICT[bondtypes[i]]
            )
            if f[3] == 1:
                ri.AddRing((dummies_idx[i],), (0,))
            if len(at_atoms) > 1:
                for j, a in enumerate(at_atoms[1:]):
                    newmol.AddAtom(a)
                    newmol.AddBond(dummies_idx[i], acount + j)
        else:
            inring_atoms = [a.IsInRing() for a in newmol.GetAtoms()]
            if -f[5] < len(curr_atoms):
                if (
                    len(
                        Chem.GetShortestPath(
                            newmol, curr_atoms[0], curr_atoms[-1 * f[5]]
                        )
                    )
                    > 2
                ):  # check this number
                    newbondidx = newmol.AddBond(
                        curr_atoms[0],
                        curr_atoms[-1 * f[5]],
                        order=BONDDICT[bondtypes[i]],
                    )
                    idx_to_remove.append(dummies_idx[i])
                    # doing the below because +1 doesnt work for some reason
                    nb = [
                        a.GetIdx()
                        for a in newmol.GetAtomWithIdx(
                            curr_atoms[-1 * f[5]]
                        ).GetNeighbors()
                        if a.GetAtomicNum() == 0
                    ]
                    idx_to_remove.append(
                        nb[0]
                    )  # why does i need to be removed here when the dummy is otherwise untouched?
                    for i, isinring in enumerate(inring_atoms):
                        if isinring:
                            ri.AddRing((i,), (newbondidx,))
                else:
                    # skipped because they were too close
                    error = True
            else:
                print("THIS SHOULD NOT HAPPEN. NOT ENOUGH DUMMIES TO CLOSE RING")
                error = True
    # note: i do it in this way becayse batch edit messes up ringinfo
    for idx in sorted(idx_to_remove)[::-1]:  # removeatom messes up ring info
        newmol.RemoveAtom(idx)
        inring_atoms.pop(idx)
    for i, isinring in enumerate(inring_atoms):
        if isinring:
            ri.AddRing((i,), (newbondidx,))
    if error:
        newmol = W_ATOM
    return newmol


def initialize_atomtypes(mode="GDB11", atomtypes=[(6, 1, 3, 0, 0, 0)]):
    """
    get a list of unique atomtype tuples. the user can also provide a
    custom list by changing the mode to something different and providing
    atomtypes explicitly. 
    """
    global ALLOWED_ATOMS
    global ATOMTYPES
    global MODE
    MODE = mode
    if mode == "GDB11":
        ALLOWED_ATOMS = GDBSET
        ATOMTYPES = sorted(generate_atom_types(GDBSET), key=lambda x: x[1])
    elif mode == "GDB17":
        ALLOWED_ATOMS = GDB17SET
        ATOMTYPES = sorted(generate_atom_types(GDB17SET), key=lambda x: x[1])
    elif mode == "FULL":
        ALLOWED_ATOMS = FULLSET
        ATOMTYPES = sorted(generate_atom_types(FULLSET), key=lambda x: x[1])
    elif mode == "CHEMBL":
        ATOMTYPES = CHEMBL_ATOMTYPES
    else:
        print("invalid mode provided. using user provided atomtypes")
        ATOMTYPES = atomtypes
    return

def set_score_weights(weights):
    """
    helper function to make changing the scoring function weights easier
    """
    global SCORE_WEIGHTS
    SCORE_WEIGHTS = weights

def set_fp_settings(radius=3, fpsize=4096):
    """
    helper function to make changing the FP settings easier
    """
    global RADIUS
    global FPSIZE
    global MFPGEN
    global AO
    RADIUS = radius
    FPSIZE = fpsize
    MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE)
    AO = (
        rdFingerprintGenerator.AdditionalOutput()
    )  # does this need to be reinitialized?
    AO.AllocateBitInfoMap()


def get_fp(mol):
    """
    helper function to get morgan FP so the radius and fpsize changes are
    persistent.
    """
    return MFPGEN.GetFingerprint(mol)


initialize_atomtypes(mode=MODE)
