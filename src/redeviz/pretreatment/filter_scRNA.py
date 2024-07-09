import numpy as np
import anndata


def filter_scRNA_main(args):
    sce = anndata.read_h5ad(args.input)
    X = sce.X
    UMI_per_cell = np.sum(X, 1)
    expr_gene_num = np.sum(X>1, 1)
    cond1 = UMI_per_cell > args.min_UMI_per_cell
    cond2 = expr_gene_num > args.min_expr_gene_per_cell
    cond = cond1 & cond2
    select_sce = sce[cond, :]

    if args.downsampling_cell_number is not None:
        if select_sce.shape[0] > args.downsampling_cell_number:
            select_index = np.random.choice(select_sce.shape[0], args.downsampling_cell_number, replace=False)
            select_sce = select_sce[select_index]
            
    select_sce.write_h5ad(args.output)
