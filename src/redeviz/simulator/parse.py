from redeviz.simulator.simulate_by_emb_info import simulate_by_emb_info_main
from redeviz.simulator.simulate_by_scRNA import simulate_by_scRNA_main
from redeviz.simulator.simulate_by_cfg import simulate_by_cfg_main
from redeviz.simulator.point2spot import point2spot_main
from redeviz.simulator.point2img import point2img_main
from redeviz.simulator.extract_simulated_expr import extract_simulated_expr_main

def parse_simulate_args(simulate_subparser):
    parser_simulate_by_emb_info = simulate_subparser.add_parser('simulate_by_emb', help='Simulate ST data by embedding information.')
    parser_simulate_by_emb_info.set_defaults(func=simulate_by_emb_info_main)
    parser_simulate_by_emb_info.add_argument("-i", "--input", type=str, dest="input",
                        metavar="input.pkl", required=True, help="scRNA-seq index")
    parser_simulate_by_emb_info.add_argument("--NCR-label", type=str, dest="NCR_label",
                        metavar="NCR_label", required=False, default=None, help="Nucleus Cytosol Ratio label in scRNA-seq data")
    parser_simulate_by_emb_info.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_simulate_by_emb_info.add_argument("--x-range", type=int, dest="x_range",
                        metavar="x_range", required=False, default=1e6, help="x-range")
    parser_simulate_by_emb_info.add_argument("--y-range", type=int, dest="y_range",
                        metavar="y_range", required=False, default=1e6, help="x-range")
    parser_simulate_by_emb_info.add_argument("--nucleis-radius", type=float, nargs=2, dest="nucleis_radius",
                        metavar="nucleis_radius", required=False, default=(1500, 3000), help="Range of nucleis radius")
    parser_simulate_by_emb_info.add_argument("--nucleus-shift-factor", type=float, dest="nucleus_shift_factor",
                        metavar="nucleus_shift_factor", required=False, default=0.2)
    parser_simulate_by_emb_info.add_argument("--uniform-noise-expr-ratio", type=float, dest="uniform_noise_expr_ratio",
                        metavar="uniform_noise_expr_ratio", required=False, default=0.0001)
    parser_simulate_by_emb_info.add_argument("--diffusion-noise-expr-ratio", type=float, dest="diffusion_noise_expr_ratio",
                        metavar="diffusion_noise_expr_ratio", required=False, default=0.01)
    parser_simulate_by_emb_info.add_argument("--nucleus-RNA-capture-efficiency", type=float, dest="nucleus_RNA_capture_efficiency",
                        metavar="nucleus_RNA_capture_efficiency", required=False, default=1.0)

    parser_simulate_by_scRNA = simulate_subparser.add_parser('simulate_by_scRNA', help='Simulate ST data by scRNA-seq data.')
    parser_simulate_by_scRNA.set_defaults(func=simulate_by_scRNA_main)
    parser_simulate_by_scRNA.add_argument("-i", "--input", type=str, dest="input",
                        metavar="input.h5ad", required=True, help="scRNA-seq data in H5AD format")
    parser_simulate_by_scRNA.add_argument("--cell-type-label", type=str, dest="cell_type_label",
                        metavar="cell_type_label", required=True, help="Cell type label in scRNA-seq data")
    parser_simulate_by_scRNA.add_argument("--gene-id-label", type=str, dest="gene_id_label",
                        metavar="gene_id_label", required=True, help="Gene ID label in scRNA-seq data")
    parser_simulate_by_scRNA.add_argument("--NCR-label", type=str, dest="NCR_label",
                        metavar="NCR_label", required=False, default=None, help="Nucleus Cytosol Ratio label in scRNA-seq data")
    parser_simulate_by_scRNA.add_argument("--min-cell-num", type=int, dest="min_cell_num",
                        metavar="min_cell_num", required=False, default=20, help="Minimum number of cells for cell types in scRNA-seq data")
    parser_simulate_by_scRNA.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_simulate_by_scRNA.add_argument("--x-range", type=int, dest="x_range",
                        metavar="x_range", required=False, default=3e6, help="x-range")
    parser_simulate_by_scRNA.add_argument("--y-range", type=int, dest="y_range",
                        metavar="y_range", required=False, default=3e6, help="x-range")
    parser_simulate_by_scRNA.add_argument("--nucleis-radius", type=float, nargs=2, dest="nucleis_radius",
                        metavar="nucleis_radius", required=False, default=(1500, 3000), help="Range of nucleis radius")
    parser_simulate_by_scRNA.add_argument("--nucleus-shift-factor", type=float, dest="nucleus_shift_factor",
                        metavar="nucleus_shift_factor", required=False, default=0.2)
    parser_simulate_by_scRNA.add_argument("--uniform-noise-expr-ratio", type=float, dest="uniform_noise_expr_ratio",
                        metavar="uniform_noise_expr_ratio", required=False, default=0.0001)
    parser_simulate_by_scRNA.add_argument("--diffusion-noise-expr-ratio", type=float, dest="diffusion_noise_expr_ratio",
                        metavar="diffusion_noise_expr_ratio", required=False, default=0.01)
    parser_simulate_by_scRNA.add_argument("--nucleus-RNA-capture-efficiency", type=float, dest="nucleus_RNA_capture_efficiency",
                        metavar="nucleus_RNA_capture_efficiency", required=False, default=1.0)

    parser_point2spot = simulate_subparser.add_parser('point2spot', help='Format simulated ST data into spot format.')
    parser_point2spot.set_defaults(func=point2spot_main)
    parser_point2spot.add_argument("-i", "--input", type=str, dest="input",
                        metavar="RNA_point.tsv", required=True, help="RNA point generated by SISVA::simulator.")
    parser_point2spot.add_argument("--cell-shape-info", type=str, dest="cell_shape_info",
                        metavar="cell_shape_info.tsv", required=True, help="Cell shape information file generated by SISVA::simulator.")
    parser_point2spot.add_argument("--x-range", type=float, dest="x_range",
                        metavar="x_range", required=True, help="x_range")
    parser_point2spot.add_argument("--y-range", type=float, dest="y_range",
                        metavar="y_range", required=True, help="y_range")
    parser_point2spot.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_point2spot.add_argument("--gap-size", type=float, dest="gap_size",
                        metavar="gap_size", required=False, default=500)
    parser_point2spot.add_argument("--spot-size", type=float, dest="spot_size",
                        metavar="spot_size", required=False, default=250)
    parser_point2spot.add_argument("--UMI-per-spot", type=float, dest="UMI_per_spot",
                        metavar="UMI_per_spot", required=False, default=0.13)

    parser_point2img = simulate_subparser.add_parser('point2img', help='Plot simulated ST data.')
    parser_point2img.set_defaults(func=point2img_main)
    parser_point2img.add_argument("-i", "--input", type=str, dest="input",
                        metavar="RNA_point.tsv", required=True, help="RNA point generated by SISVA::simulator.")
    parser_point2img.add_argument("-r", "--resolution", type=float, dest="resolution", default=100,
                        metavar="resolution", required=False, help="resolution")
    parser_point2img.add_argument("-b", "--barcode", type=str, dest="barcode",
                        metavar="barcode", required=True, help="barcode.tsv")
    parser_point2img.add_argument("--x-range", type=float, dest="x_range",
                        metavar="x_range", required=True, help="x_range")
    parser_point2img.add_argument("--y-range", type=float, dest="y_range",
                        metavar="y_range", required=True, help="y_range")
    parser_point2img.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_point2img.add_argument("--max-UMI-per-px", type=int, dest="max_UMI_per_px",
                        metavar="max_UMI_per_px", required=False, default=1, help="Max UMI per px")
    
    parser_extract_simulated_expr = simulate_subparser.add_parser('extract_simulated_expr', help='Extract simulated gene expression.')
    parser_extract_simulated_expr.set_defaults(func=extract_simulated_expr_main)
    parser_extract_simulated_expr.add_argument("--emb-info", type=str, dest="emb_info",
                        metavar="emb_info.pkl", required=True, help="scRNA-seq index")
    parser_extract_simulated_expr.add_argument("--gene-list", type=str, dest="gene_list", metavar="gene_list.txt", required=False, default=None, help="Gene list")
    parser_extract_simulated_expr.add_argument("--spot-info", type=str, dest="spot_info", metavar="Spot.CellType.tsv", required=True, help="Spot information")
    parser_extract_simulated_expr.add_argument("--cell-shape", type=str, dest="cell_shape", metavar="CellShapeInfo.tsv", required=True, help="Cell shape information")
    parser_extract_simulated_expr.add_argument("-o", "--output", type=str, dest="output",
                        metavar="output", required=True, help="Output folder")
    parser_extract_simulated_expr.add_argument("--device-name", type=str, dest="device_name",
                        metavar="device_name", required=False, default="CPU", 
                        help="Device name (default=CPU)")