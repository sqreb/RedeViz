import pandas as pd
from typing import List
from numpy.random import uniform, poisson
from scipy.stats import dirichlet

class GeneExprInfo(object):
    def __init__(self, fname: str, cell_type_li: List[str]) -> None:
        expr_df = pd.read_csv(fname, sep="\t")
        self.expr_df = expr_df
        self.cell_type_li = cell_type_li
        self.gene_li = expr_df["Gid"].to_numpy()
        for loc in set(expr_df["Location"]):
            if loc not in ["Nucleus", "Cytosol", "All"]:
                raise ValueError("""Location must be one of "Nucleus", "Cytosol" and "All".""")
        NCR_li = uniform(2, 10, expr_df.shape[0])
        NCR_li[expr_df["Location"]=="All"] = 1
        NCR_li[expr_df["Location"]=="Cytosol"] = 1 / NCR_li[expr_df["Location"]=="Cytosol"]
        self.NCR_li = NCR_li
        self.expr_dict = dict()
        self.total_expr_dict = dict()
        for cell_type in cell_type_li:
            if cell_type not in expr_df.columns:
                raise ValueError(f"Can not find expression information of {cell_type} in {fname}.")
            self.expr_dict[cell_type] = expr_df[cell_type].to_numpy()
            self.total_expr_dict[cell_type] = self.expr_dict[cell_type].sum()
        self.ave_expr_li = expr_df[cell_type_li].to_numpy().mean(1)
    
    def simulate_gene_point_number(self, cell_type: str, N: float):
        mu_li = N * dirichlet.rvs(self.expr_dict[cell_type]+1)[0]
        gene_point_num = poisson(mu_li)
        res_li = list()
        for gid, num, NCR in zip(self.gene_li, gene_point_num, self.NCR_li):
            if num == 0:
                continue
            res_li.append((gid, num, NCR))
        return res_li
    
    def simulate_ave_point_number(self, N: float):
        mu_li = N * dirichlet.rvs(self.ave_expr_li)[0]
        gene_point_num = poisson(mu_li)
        res_li = list()
        for gid, num, NCR in zip(self.gene_li, gene_point_num, self.NCR_li):
            if num == 0:
                continue
            res_li.append((gid, num, NCR))
        return res_li