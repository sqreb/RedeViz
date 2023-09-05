from redeviz.enhance.index import build_embedding_info_main, build_embedding_index_main
from redeviz.enhance.enhance import enhance_main
from redeviz.enhance.image_index import build_embedding_image_index_main
from redeviz.enhance.img_enhance import img_enhance_main
import pickle

def run_enhance_main(args):
    with open(args.index, "rb") as f:
        dataset_dict = pickle.load(f)
    if "index_type" in dataset_dict.keys():
        if dataset_dict["index_type"] in ["Image", "ImageNorm"]:
            img_enhance_main(args)
            return None
    enhance_main(args)


def parse_enhance_args(enhance_subparser):
    parser_pretreat = enhance_subparser.add_parser('pretreat', help='Pretreat help')
    parser_pretreat.set_defaults(func=build_embedding_info_main)
    parser_pretreat.add_argument("-i", "--input", type=str, dest="input",
                        metavar="input.h5ad", required=True, help="scRNA-seq data with H5AD format.")
    parser_pretreat.add_argument("--cell-type-label", type=str, dest="cell_type_label",
                        metavar="cell_type_label", required=True, help="Cell type label in scRNA-seq dataset.")
    parser_pretreat.add_argument("--gene-id-label", type=str, dest="gene_id_label",
                        metavar="gene_id_label", required=True, help="Gene ID label in scRNA-seq dataset.")
    parser_pretreat.add_argument("--embedding", type=str, dest="embedding",
                        metavar="embedding", required=False, default="tSNE", choices=["tSNE", "UMAP"], 
                        help="Embedding method [tSNE, UMAP] (default=tSNE)")
    parser_pretreat.add_argument("--embedding-resolution", type=float, dest="embedding_resolution",
                        metavar="embedding_resolution", required=False, default=None, help="Embedding bin resolution [default value: 2.5 (Dim=2); 6 (Dim=3)]")
    parser_pretreat.add_argument("--embedding-dim", type=int, dest="embedding_dim",
                        metavar="embedding_dim", required=False, default=2, choices=[2, 3], help="Embedding dim [2, 3] (default=2)")
    parser_pretreat.add_argument("--n-neighbors", type=int, dest="n_neighbors",
                        metavar="n_neighbors", required=False, default=15, help="n_neighbors params in UMAP method (default=15)")
    parser_pretreat.add_argument("--min-cell-num", type=int, dest="min_cell_num",
                        metavar="min_cell_num", required=False, default=2, help="Minimum reference cell number in one embedding bin (default=2)")
    parser_pretreat.add_argument("--gene-blacklist", type=str, dest="gene_blacklist",
                        metavar="gene_blacklist.txt", required=False, default=None, help="Genes need to remove from scRNA-seq data")
    parser_pretreat.add_argument("--max-expr-ratio", type=float, dest="max_expr_ratio",
                        metavar="max_expr_ratio", required=False, default=0.05, help="Max gene expression filtering ratio [0, 1]")
    parser_pretreat.add_argument("--cell-state-topN-gene", type=int, dest="cell_state_topN_gene",
                        metavar="cell_state_topN_gene", required=False, default=2, help="topN gene for filtering cell state (default=2)")
    parser_pretreat.add_argument("--cell-state-topN-gene-max-ratio", type=float, dest="cell_state_topN_gene_max_ratio",
                        metavar="cell_state_topN_gene_max_ratio", required=False, default=0.3, help="Max gene expression ratio for topN gene in each cell state (default=0.3)")
    parser_pretreat.add_argument("--no-HVG", dest="no_HVG", action="store_true")
    parser_pretreat.add_argument("--merge-method", type=str, dest="merge_method",
                        metavar="merge_method", required=False, default="None", choices=["None", "log"], 
                        help="Merge method [None, log] (default=None)")
    parser_pretreat.add_argument("--device-name", type=str, dest="device_name",
                        metavar="device_name", required=False, default="CPU", 
                        help="Device name (default=CPU)")
    parser_pretreat.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")

    parser_index = enhance_subparser.add_parser('build', help='Build index help')
    parser_index.set_defaults(func=build_embedding_index_main)
    parser_index.add_argument("-i", "--input", type=str, dest="input",
                        metavar="pretreat.pkl", required=True, help="Pretreat file.")
    parser_index.add_argument("--bin-size", nargs="*", type=int, dest="bin_size",
                        metavar="bin_size", required=False, default=[9, 21], help="Bin size (default=[9, 21])")
    parser_index.add_argument("--UMI-per-spot", type=float, dest="UMI_per_spot",
                        metavar="UMI_per_spot", required=False, default=1.5, help="UMI per spot (default=1.5)")
    parser_index.add_argument("-o", "--output", type=str, dest="output",
                        metavar="index.pkl", required=True, help="Output file")           
    parser_index.add_argument("--norm-method", type=str, dest="norm_method",
                        metavar="norm_method", required=False, default="None", choices=["None", "log"], 
                        help="Normalize method [None, log] (default=None)")
    parser_index.add_argument("--device-name", type=str, dest="device_name",
                        metavar="device_name", required=False, default=None, 
                        help="Device name (default=auto)")
    
    parser_img_norm_index = enhance_subparser.add_parser('build_img', help='Build image index help')
    parser_img_norm_index.set_defaults(func=build_embedding_image_index_main)
    parser_img_norm_index.add_argument("-i", "--input", type=str, dest="input",
                        metavar="pretreat.pkl", required=True, help="Pretreat file.")
    parser_img_norm_index.add_argument("--select-gene", type=str, dest="select_gene",
                        metavar="select_gene.txt", required=True, help="Selected gene")
    parser_img_norm_index.add_argument("--bin-size", nargs="*", type=int, dest="bin_size",
                        metavar="bin_size", required=False, default=[12, 18, 21], help="Bin size (default=[12, 18, 21])")
    parser_img_norm_index.add_argument("--UMI-per-spot", type=float, dest="UMI_per_spot",
                        metavar="UMI_per_spot", required=False, default=0.5, help="UMI per spot (default=0.5)")
    parser_img_norm_index.add_argument("-o", "--output", type=str, dest="output",
                        metavar="index.pkl", required=True, help="Output file")
    parser_img_norm_index.add_argument("--device-name", type=str, dest="device_name",
                        metavar="device_name", required=False, default=None, 
                        help="Device name (default=auto)")
    
    parser_main = enhance_subparser.add_parser('run', help='Run SpatialSegment')
    parser_main.set_defaults(func=run_enhance_main)
    parser_main.add_argument("-s", "--spot", type=str, dest="spot",
                        metavar="spot.tsv", required=True, help="Spot information")
    parser_main.add_argument("-i", "--index", type=str, dest="index",
                        metavar="index.pkl", required=True, help="SpatialSegment index")
    parser_main.add_argument("--x-index-label", type=str, dest="x_index_label",
                        metavar="x_index_label", required=False, default="spot_x_index", 
                        help="X index label: (default: x_index_label)")
    parser_main.add_argument("--y-index-label", type=str, dest="y_index_label",
                        metavar="y_index_label", required=False, default="spot_y_index", 
                        help="Y index label: (default: y_index_label)")
    parser_main.add_argument("--gene-name-label", type=str, dest="gene_name_label",
                        metavar="gene_name_label", required=False, default="Gid", 
                        help="Gene label (default: Gid)")
    parser_main.add_argument("--cell-radius", type=int, dest="cell_radius",
                        metavar="cell_radius", required=False, default=4, 
                        help="Cell radius (default: 11)")
    parser_main.add_argument("--UMI-label", type=str, dest="UMI_label",
                        metavar="UMI_label", required=False, default="UMI", 
                        help="UMI label (default: UMI)")
    parser_main.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output.tsv", required=True, help="Output file")
    parser_main.add_argument("--window-size", type=int, dest="window_size",
                        metavar="window_size", required=False, default=256, 
                        help="Window size (default: 256)")
    parser_main.add_argument("--update-num", type=int, dest="update_num",
                        metavar="update_num", required=False, default=2, 
                        help="Update epoch number (default: 2)")
    parser_main.add_argument("--max-expr-ratio", type=float, dest="max_expr_ratio",
                        metavar="max_expr_ratio", required=False, default=0.05, help="Max gene expression filtering ratio [0, 1]")
    parser_main.add_argument("--mid-signal-cutoff", type=float, dest="mid_signal_cutoff",
                        metavar="mid_signal_cutoff", required=False, default=None, 
                        help="Middle signal cutoff (default: Auto)")
    parser_main.add_argument("--neighbor-close-label-fct", type=float, dest="neighbor_close_label_fct",
                        metavar="neighbor_close_label_fct", required=False, default=0.2, 
                        help="Neighbor close label factor (default: 0.2)")
    parser_main.add_argument("--signal-cov-score-fct", type=float, dest="signal_cov_score_fct",
                        metavar="signal_cov_score_fct", required=False, default=1.0, 
                        help="Signal coverage score factor (default: 1.0)")
    parser_main.add_argument("--is-in-ref-score-fct", type=float, dest="is_in_ref_score_fct",
                        metavar="is_in_ref_score_fct", required=False, default=1.0, 
                        help="Is in reference embedding bin score factor (default: 1.0)")
    parser_main.add_argument("--argmax-prob-score-fct", type=float, dest="argmax_prob_score_fct",
                        metavar="argmax_prob_score_fct", required=False, default=1.0, 
                        help="Argmax probability score factor (default: 1.0)")
    parser_main.add_argument("--ave-bin-dist-cutoff", type=int, dest="ave_bin_dist_cutoff",
                        metavar="ave_bin_dist_cutoff", required=False, default=None, 
                        help="Average embedding bin distance for neighbor spots")
    parser_main.add_argument("--slice-x", type=int, nargs=2, dest="slice_x",
                        metavar="slice_x", required=False, default=None, 
                        help="Slice x region")
    parser_main.add_argument("--slice-y", type=int, nargs=2, dest="slice_y",
                        metavar="slice_y", required=False, default=None, 
                        help="Slice y region")
    parser_main.add_argument("--device-name", type=str, dest="device_name",
                        metavar="device_name", required=False, default=None, 
                        help="Device name (default=auto)")
