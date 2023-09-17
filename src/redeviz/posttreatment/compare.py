import pandas as pd
from redeviz.posttreatment.utils import filter_pred_df


def load_data(fname, sor_cutoff, sbr_cutoff, keep_other, denoise, min_spot_num):
    spot_df = pd.read_csv(fname, sep="\t")
    spot_df = spot_df[spot_df["LabelTransfer"]!="Background"]
    if not keep_other:
        spot_df = spot_df[spot_df["LabelTransfer"]!="Other"]
    if denoise:
        spot_df = filter_pred_df(spot_df, min_spot_in_region=min_spot_num)

    spot_df["sor"] = spot_df["RefCellTypeScore"] / spot_df["OtherCellTypeScore"]
    spot_df["sbr"] = spot_df["RefCellTypeScore"] / spot_df["BackgroundScore"]
    spot_df = spot_df[spot_df["sor"]>sor_cutoff]
    spot_df = spot_df[spot_df["sbr"]>sbr_cutoff]
    spot_df = spot_df[["x", "y", "ArgMaxCellType", "EmbeddingState"]]
    return spot_df

def compare_main(args):
    f_spot1 = args.input1
    f_spot2 = args.input2
    f_out = args.output
    sor_cutoff = args.sor_cutoff
    sbr_cutoff = args.sbr_cutoff
    phenotype_level = args.phenotype_level
    keep_other = args.keep_other

    spot_df1 = load_data(f_spot1, sor_cutoff, sbr_cutoff, keep_other, args.denoise, args.min_spot_num)
    spot_df1.columns = ["x", "y", "CellType1", "EmbeddingState1"]

    spot_df2 = load_data(f_spot2, sor_cutoff, sbr_cutoff, keep_other, args.denoise, args.min_spot_num)
    spot_df2.columns = ["x", "y", "CellType2", "EmbeddingState2"]

    merge_df = spot_df1.merge(spot_df2, how="inner")

    if phenotype_level:
        res_df = merge_df.groupby(["CellType1", "EmbeddingState1", "CellType2", "EmbeddingState2"]).size().reset_index(name="SpotNum")
    else:
        res_df = merge_df.groupby(["CellType1", "CellType2"]).size().reset_index(name="SpotNum")
    res_df.to_csv(f_out, sep="\t", index_label=False, index=False)
