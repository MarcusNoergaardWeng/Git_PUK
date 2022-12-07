import pandas as pd


def ase_to_pandas(ase_db, position="", subscript="", *columns):
    """
    Indices are disregarded because they might not be consistent among different databases
    :param ase_db: ASE database object containing the structures of interest
    :param position: if specified, select position of interest (e.g., top-top, top-bridge, ...)
    :param subscript: if specified, add a subscript to the energy column (useful for merging different dfs)
    :return: pd.DataFrame with chemical formulas (as indices) and energies
    """
    selection = None
    if position:
        selection = "position=" + position

    formulas = [row.toatoms().symbols.formula.format("metal") for row in ase_db.select(selection=selection)]
    energies = [row.energy for row in ase_db.select(selection=selection)]
    for col in columns:
        column = [row.col for row in ase_db.select(selection=selection)]

    header = "E"
    if subscript:
        header += "_" + subscript

    df = pd.DataFrame(list(zip(formulas, energies)), columns=["formula", header])
    return df.set_index("formula")


# db_IS = connect("IS_COOR.db")
# db_TS = connect("TS_COOR.db")
# db_FS = connect("FS_COOR.db")
#
# IS_df = ase_to_pandas(db_IS)
# TS_df = ase_to_pandas(db_TS)
# FS_df = ase_to_pandas(db_TS)
#
# print(IS_df)
# print(TS_df)
# print(FS_df)
#
# print(f"the activation energy for an Au surface is: {TS_df.loc['Au64CO2', 'energy'] - IS_df.loc['Au64CO2', 'energy']}")

#
#
# help(db_FS.select)
#
# for row in db_IS.select():
#     print(row.id)
#     print(row.toatoms().symbols.formula.format("metal"))
#     print(row.energy)
#
#
# for row in db_FS.select():
#     atom = row.toatoms()
#     print(atom.symbols)
#     print(atom.__dict__["_calc"])
#     print(atom._calc)
#     print(type(atom._calc))
#     # print(atom._calc[1])
#     break
#
# for row in db_IS.select(columns=["id", "formul", "energy"]):
#     # print(row.formula)
#     list_of_values = list(row.__dict__.items())
#     print(list_of_values)
#     print(row.formul)
#     atoms = row.toatoms()
#     print(atoms)
#     print(atoms.get_chemical_formula())
#     print(row.symbols)
#     # formula = row.toatoms().get_chemical_formula()
#     # print(formula)
