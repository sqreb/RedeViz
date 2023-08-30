from redeviz.posttreatment.imputation import imputation_build_index_main, imputation_main
from redeviz.posttreatment.plot_gene_expr import plot_gene_expr_main
from redeviz.posttreatment.plot_gene_bin_expr import plot_gene_bin_expr_main
from redeviz.posttreatment.segment import segment_main
from redeviz.posttreatment.plot_phenotype import plot_phenotype_main
from redeviz.posttreatment.plot_cell_type import plot_cell_type_main
from redeviz.posttreatment.compare import compare_main


def parse_posttreatment_args(posttreatment_subparser):
    parser_plot_phenotype = posttreatment_subparser.add_parser('plot_phenotype', help='Phenotype visualization.')
    parser_plot_phenotype.set_defaults(func=plot_phenotype_main)
    parser_plot_phenotype.add_argument("--input", type=str, dest="input",
                        metavar="spot.pred.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_plot_phenotype.add_argument("--output", type=str, dest="output",
                        metavar="output.png", required=True, help="Output image")
    parser_plot_phenotype.add_argument("--keep-other", dest="keep_other", action="store_true", help="Force to impute other cell types")
    parser_plot_phenotype.add_argument("--min-denoise-spot-num", type=int, dest="min_spot_num", default=200)
    parser_plot_phenotype.add_argument("--denoise", dest="denoise", action="store_true", help="Remove scattered pixel")

    parser_plot_cell_type = posttreatment_subparser.add_parser('plot_cell_type', help='Cell type visualization.')
    parser_plot_cell_type.set_defaults(func=plot_cell_type_main)
    parser_plot_cell_type.add_argument("--input", type=str, dest="input",
                        metavar="spot.pred.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_plot_cell_type.add_argument("--output", type=str, dest="output",
                        metavar="output.png", required=True, help="Output image")
    parser_plot_cell_type.add_argument("--color", type=str, dest="color",
                        metavar="color.tsv", required=True, help="Color for each cell type")
    parser_plot_cell_type.add_argument("--keep-other", dest="keep_other", action="store_true", help="Force to impute other cell types")
    parser_plot_cell_type.add_argument("--min-denoise-spot-num", type=int, dest="min_spot_num", default=200)
    parser_plot_cell_type.add_argument("--denoise", dest="denoise", action="store_true", help="Remove scattered pixel")
    
    parser_impute = posttreatment_subparser.add_parser('impute', help='Gene expression imputation')
    impute_subparsers = parser_impute.add_subparsers(help='sub-command help')
    parser_build = impute_subparsers.add_parser('build', help='Build imputation index.')
    parser_build.set_defaults(func=imputation_build_index_main)
    parser_build.add_argument("--index", type=str, dest="index", metavar="index.pkl", required=True, help="Embedding information with pkl format")
    parser_build.add_argument("--sce", type=str, dest="sce", metavar="sce.h5ad", required=True, help="scRNA-seq count matrix with H5AD format")
    parser_build.add_argument("--gene-name-label", type=str, dest="gene_name_label", metavar="gene_name_label", required=True, help="Gene name label in SCE file")
    parser_build.add_argument("--embedding-smooth-sigma", type=float, dest="embedding_smooth_sigma", metavar="embedding_smooth_sigma", required=False, default=None, help="Embedding smooth sigma")
    parser_build.add_argument("--output", type=str, dest="output", metavar="imputation.pkl", required=True, help="Imputation index")

    parser_impute = impute_subparsers.add_parser('run', help='Run imputation.')
    parser_impute.set_defaults(func=imputation_main)
    parser_impute.add_argument("--input", type=str, dest="input", metavar="spot.pred.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_impute.add_argument("--index", type=str, dest="index", metavar="index.pkl", required=True, help="Embedding information with pkl format")
    parser_impute.add_argument("--spot", type=str, dest="spot", metavar="spot.tsv", required=True, help="Spot Bin1 file with TSV format")
    parser_impute.add_argument("--gene-list", type=str, dest="gene_list", metavar="gene_list.txt", required=False, default=None, help="Gene list for imputation")
    parser_impute.add_argument("--spot-pos-pos-label", nargs=2, type=str, dest="spot_pos_pos_label", metavar="spot_pos_pos_label", required=True, help="X and Y label in spot Bin1 file")
    parser_impute.add_argument("--spot-UMI-label", type=str, dest="spot_UMI_label", metavar="spot_UMI_label", required=True, help="UMI label in spot Bin1 file")
    parser_impute.add_argument("--embedding-smooth-sigma", type=float, dest="embedding_smooth_sigma", metavar="embedding_smooth_sigma", required=False, default=3.0, help="Embedding smooth sigma [default=3.0]")
    parser_impute.add_argument("--UMI-smooth-sigma", type=float, dest="UMI_smooth_sigma", metavar="UMI_smooth_sigma", required=False, default=10.0, help="UMI smooth sigma [default=3.0]")
    parser_impute.add_argument("--output", type=str, dest="output", metavar="imputation", required=True, help="Imputation folder")
    parser_impute.add_argument("--keep-other", dest="keep_other", action="store_true", help="Force to impute other cell types")
    parser_impute.add_argument("--min-denoise-spot-num", type=int, dest="min_spot_num", default=200)
    parser_impute.add_argument("--denoise", dest="denoise", action="store_true", help="Remove scattered pixel")
    
    parser_plot_gene_expr = posttreatment_subparser.add_parser('plot_gene_expr', help='Plot gene expression.')
    parser_plot_gene_expr.set_defaults(func=plot_gene_expr_main)
    parser_plot_gene_expr.add_argument("-R", type=str, dest="R", metavar="R.expr.npz", required=False, default=None)
    parser_plot_gene_expr.add_argument("-G", type=str, dest="G", metavar="G.expr.npz", required=False, default=None)
    parser_plot_gene_expr.add_argument("-B", type=str, dest="B", metavar="B.expr.npz", required=False, default=None)
    parser_plot_gene_expr.add_argument("-o", "--output", type=str, dest="output", metavar="output.png", required=True)
    
    parser_plot_gene_bin_expr = posttreatment_subparser.add_parser('plot_gene_bin_expr', help='Plot gene bin expression.')
    parser_plot_gene_bin_expr.set_defaults(func=plot_gene_bin_expr_main)
    parser_plot_gene_bin_expr.add_argument("-i", "--input", type=str, dest="input", metavar="input.tsv", required=True)
    parser_plot_gene_bin_expr.add_argument("--x-label", type=str, dest="x_label", metavar="x_label", required=True)
    parser_plot_gene_bin_expr.add_argument("--y-label", type=str, dest="y_label", metavar="y_label", required=True)
    parser_plot_gene_bin_expr.add_argument("--gene-name-label", type=str, dest="gene_name_label", metavar="gene_name_label", required=True)
    parser_plot_gene_bin_expr.add_argument("--UMI-label", type=str, dest="UMI_label", metavar="UMI_label", required=True)
    parser_plot_gene_bin_expr.add_argument("-R", type=str, dest="R", metavar="R_gene_name", required=False, default=None)
    parser_plot_gene_bin_expr.add_argument("-G", type=str, dest="G", metavar="G_gene_name", required=False, default=None)
    parser_plot_gene_bin_expr.add_argument("-B", type=str, dest="B", metavar="B_gene_name", required=False, default=None)
    parser_plot_gene_bin_expr.add_argument("-o", "--output", type=str, dest="output", metavar="output.png", required=True)
    
    parser_segment = posttreatment_subparser.add_parser('segment', help='Segmentation.')
    parser_segment.set_defaults(func=segment_main)
    parser_segment.add_argument("--input", type=str, dest="input",
                        metavar="spot.pred.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_segment.add_argument("--smooth-radius", type=int, dest="smooth_radius",
                        metavar="smooth_radius", required=False, default=0, help="Smooth kernel radius (default=0)")
    parser_segment.add_argument("--receptive-radius", type=float, dest="receptive_radius",
                        metavar="receptive_radius", required=False, default=210.0, help="Spatial window radius in mean shift algorithm (default=20.0)")
    parser_segment.add_argument("--embedding-radius", type=float, dest="embedding_radius",
                        metavar="embedding_radius", required=False, default=5.0, help="Embedding window radius in mean shift algorithm (default=10.0)")
    parser_segment.add_argument("--merge-embedding-dist", type=float, dest="merge_embedding_dist",
                        metavar="merge_embedding_dist", required=False, default=5.0, help="Embedding distance threshold to merge regions into domain (default=5.0)")
    parser_segment.add_argument("--min-spot-per-domain", type=float, dest="min_spot_num",
                        metavar="min_spot_num", required=False, default=50, help="Minimum spot number for each domain (default=50)")
    parser_segment.add_argument("--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_segment.add_argument("--keep-other", dest="keep_other", action="store_true", help="Force to impute other cell types")
    parser_segment.add_argument("--min-denoise-spot-num", type=int, dest="min_spot_num", default=200)
    parser_segment.add_argument("--denoise", dest="denoise", action="store_true", help="Remove scattered pixel")
    
    parser_compare = posttreatment_subparser.add_parser('compare', help='Compare RedeViz results.')
    parser_compare.set_defaults(func=compare_main)
    parser_compare.add_argument("--input1", type=str, dest="input1",
                        metavar="spot.pred1.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_compare.add_argument("--input2", type=str, dest="input2",
                        metavar="spot.pred2.tsv", required=True, help="Enhanced file generate by RedeViz")
    parser_compare.add_argument("--SOR-cutoff", type=float, dest="sor_cutoff",
                        metavar="sor_cutoff", required=False, default=1.0, help="Signal other ratio cutoff.")
    parser_compare.add_argument("--SBR-cutoff", type=float, dest="sbr_cutoff",
                        metavar="sbr_cutoff", required=False, default=1.0, help="Signal background ratio cutoff.")
    parser_compare.add_argument("--output", type=str, dest="output",
                        metavar="output.tsv", required=True, help="Output file")
    parser_compare.add_argument("--keep-other", dest="keep_other", action="store_true", help="Force to impute other cell types")
    parser_compare.add_argument("--phenotype-level", dest="phenotype_level", action="store_true", help="Stat at phenotype level")
    parser_compare.add_argument("--min-denoise-spot-num", type=int, dest="min_spot_num", default=200)
    parser_compare.add_argument("--denoise", dest="denoise", action="store_true", help="Remove scattered pixel")
    