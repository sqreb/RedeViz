from redeviz.pretreatment.spot2bin import spot2bin_main
from redeviz.pretreatment.point2bin import point2bin_main
from redeviz.pretreatment.infer_spot_expr import infer_spot_expr_main
from redeviz.pretreatment.filter_scRNA import filter_scRNA_main
from redeviz.pretreatment.emb_info_select_gene import emb_info_select_gene_main
from redeviz.pretreatment.mask_ST_data import mask_ST_data_main
from redeviz.pretreatment.IF_denoise import IF_denoise_main

def parse_pretreatment_args(pretreatment_subparser):
    parser_spot2bin = pretreatment_subparser.add_parser('spot2bin', help='Spot binning')
    parser_spot2bin.set_defaults(func=spot2bin_main)
    parser_spot2bin.add_argument("--spot", type=str, dest="spot",
                        metavar="spot.tsv", required=True, help="Spot information")
    parser_spot2bin.add_argument("--bin-size", type=int, dest="bin_size",
                        metavar="bin_size", required=True, help="bin_size")
    parser_spot2bin.add_argument("--x-index-label", type=str, dest="x_index_label",
                        metavar="x_index_label", required=False, default="spot_x_index", 
                        help="X index label: (default: x_index_label)")
    parser_spot2bin.add_argument("--y-index-label", type=str, dest="y_index_label",
                        metavar="y_index_label", required=False, default="spot_y_index", 
                        help="Y index label: (default: y_index_label)")
    parser_spot2bin.add_argument("--gene-name-label", type=str, dest="gene_name_label",
                        metavar="gene_name_label", required=False, default="Gid", 
                        help="Gene label (default: Gid)")
    parser_spot2bin.add_argument("--UMI-label", type=str, dest="UMI_label",
                        metavar="UMI_label", required=False, default="UMI", 
                        help="UMI label (default: UMI)")
    parser_spot2bin.add_argument("--output", type=str, dest="output",
                        metavar="output.tsv", required=True, help="Output file")
    
    parser_point2bin = pretreatment_subparser.add_parser('point2bin', help='Point binning')
    parser_point2bin.set_defaults(func=point2bin_main)
    parser_point2bin.add_argument("--point", type=str, dest="point",
                        metavar="point.tsv", required=True, help="RNA points information")
    parser_point2bin.add_argument("--bin-size", type=float, dest="bin_size",
                        metavar="bin_size", required=True, help="bin_size")
    parser_point2bin.add_argument("--x-label", type=str, dest="x_label",
                        metavar="x_label", required=False, default="x", 
                        help="X label: (default: x)")
    parser_point2bin.add_argument("--y-label", type=str, dest="y_label",
                        metavar="y_label", required=False, default="y", 
                        help="Y label: (default: y)")
    parser_point2bin.add_argument("--gene-name-label", type=str, dest="gene_name_label",
                        metavar="gene_name_label", required=False, default="GeneSymbol", 
                        help="Gene label (default: GeneSymbol)")
    parser_point2bin.add_argument("--output", type=str, dest="output",
                        metavar="output.tsv", required=True, help="Output file")

    parser_infer_spot_expr = pretreatment_subparser.add_parser('infer_spot_expr', help='Infer average UMI per spot')
    parser_infer_spot_expr.set_defaults(func=infer_spot_expr_main)
    parser_infer_spot_expr.add_argument("--spot", type=str, dest="spot",
                        metavar="spot.tsv", required=True, help="Spot information")
    parser_infer_spot_expr.add_argument("--x-index-label", type=str, dest="x_index_label",
                        metavar="x_index_label", required=False, default="spot_x_index", 
                        help="X index label: (default: x_index_label)")
    parser_infer_spot_expr.add_argument("--y-index-label", type=str, dest="y_index_label",
                        metavar="y_index_label", required=False, default="spot_y_index", 
                        help="Y index label: (default: y_index_label)")
    parser_infer_spot_expr.add_argument("--UMI-label", type=str, dest="UMI_label",
                        metavar="UMI_label", required=False, default="UMI", 
                        help="UMI label (default: UMI)")
    parser_infer_spot_expr.add_argument("--output", type=str, dest="output",
                        metavar="output.txt", required=True, help="Output file")
    
    parser_filter_scRNA = pretreatment_subparser.add_parser('filter_scRNA', help='Filtering scRNA-seq data')
    parser_filter_scRNA.set_defaults(func=filter_scRNA_main)
    parser_filter_scRNA.add_argument("--input", type=str, dest="input",
                        metavar="input.h5ad", required=True, help="scRNA-seq data with H5AD format")
    parser_filter_scRNA.add_argument("--min-expr-gene-per-cell", type=int, dest="min_expr_gene_per_cell",
                        metavar="min_expr_gene_per_cell", default=5, required=False, help="Min expressed gene per cell (default=5)")
    parser_filter_scRNA.add_argument("--min-UMI-per-cell", type=int, dest="min_UMI_per_cell",
                        metavar="min_UMI_per_cell", default=10, required=False, help="Min UMI per cell (default=10)")
    parser_filter_scRNA.add_argument("--downsampling-cell-number", type=int, dest="downsampling_cell_number",
                        metavar="downsampling_cell_number", default=None, required=False, help="Downsampling cell number (default=None)")
    parser_filter_scRNA.add_argument("--output", type=str, dest="output",
                        metavar="output.h5ad", required=True, help="scRNA-seq data with H5AD format")

    parser_emb_info_select_gene = pretreatment_subparser.add_parser('emb_info_select_gene', help='Select gene with embedding data')
    parser_emb_info_select_gene.set_defaults(func=emb_info_select_gene_main)
    parser_emb_info_select_gene.add_argument("--input", type=str, dest="input",
                        metavar="emb_info.pkl", required=True, help="Embedding information with pkl format")
    parser_emb_info_select_gene.add_argument("--gene-list", type=str, dest="gene_li",
                        metavar="gene_li.txt", required=True)
    parser_emb_info_select_gene.add_argument("--min-UMI-per-bin", type=int, dest="min_UMI_num",
                        metavar="min_UMI_num", default=100, required=True)
    parser_emb_info_select_gene.add_argument("--min-gene-per-bin", type=int, dest="min_gene_num",
                        metavar="min_gene_num", default=20, required=True)
    parser_emb_info_select_gene.add_argument("--output", type=str, dest="output",
                        metavar="output.pkl", required=True)

    parser_mask_ST_data = pretreatment_subparser.add_parser('mask_ST_data', help='Mask ST data')
    parser_mask_ST_data.set_defaults(func=mask_ST_data_main)
    parser_mask_ST_data.add_argument("--spot", type=str, dest="spot",
                        metavar="spot.tsv", required=True, help="Spot information")
    parser_mask_ST_data.add_argument("--x-index-label", type=str, dest="x_index_label",
                        metavar="x_index_label", required=False, default="spot_x_index", 
                        help="X index label: (default: x_index_label)")
    parser_mask_ST_data.add_argument("--y-index-label", type=str, dest="y_index_label",
                        metavar="y_index_label", required=False, default="spot_y_index", 
                        help="Y index label: (default: y_index_label)")
    parser_mask_ST_data.add_argument("--UMI-label", type=str, dest="UMI_label",
                        metavar="UMI_label", required=False, default="UMI", 
                        help="UMI label (default: UMI)")
    parser_mask_ST_data.add_argument("--mask-model", type=str, dest="mask_model",
                        metavar="mask_model", required=False, default="bin", 
                        help="Mask model [bin] (default: bin)", choices=["bin"])
    parser_mask_ST_data.add_argument("--smooth-sd", type=int, dest="smooth_sd",
                        metavar="smooth_sd", required=False, default=2)
    parser_mask_ST_data.add_argument("--cell-diameter", type=int, dest="cell_diameter",
                        metavar="cell_diameter", default=30, required=False)
    parser_mask_ST_data.add_argument("--shift-cutoff", type=int, dest="shift_cutoff",
                        metavar="shift_cutoff", default=0, required=False)
    parser_mask_ST_data.add_argument("--output", type=str, dest="output",
                        metavar="output", required=True)

    parser_IF_denoise = pretreatment_subparser.add_parser('IF_denoise', help='Denoising Immunofluorescence Images')
    parser_IF_denoise.set_defaults(func=IF_denoise_main)
    parser_IF_denoise.add_argument("--image", type=str, dest="image",
                        metavar="image.qptiff", required=True, help="CODEX image file with qptiff format")
    parser_IF_denoise.add_argument("--method", type=str, dest="method",
                        metavar="method", required=True, choices=["Noise2Fast", "rescale"])
    parser_IF_denoise.add_argument("--output", type=str, dest="output",
                        metavar="output.qptiff", required=True)
    