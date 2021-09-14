#import rdkit 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
smiles= 'CCc1ccccc1CCCCCCCCC'

mol = Chem.MolFromSmiles(smiles)
#moldata.append(mol)

#    baseData = np.arange(1, 1)
#    i = 0
#    for mol in moldata:

#mol
desc_MolWt = Descriptors.MolWt(mol)
print(desc_MolWt)

