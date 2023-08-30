
# Install RedeViz
`conda create -n RedeViz python=3.9`

`conda activate RedeViz`

`git clone https://github.com/sqreb/RedeViz.git`

`cd redeviz`

Install opencv-contrib-python with cuda version (https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)

Install Pytorch with cuda version (https://pytorch.org/get-started/locally)

`pip install -r requirements.txt`

`python setup.py install`

# Example
## Download reference scRNA-seq data
`cd demo`

Download `Eye_simple.h5ad` from https://figshare.com/account/projects/169788/articles/23506638.

## Extract phenotypic information from scRNA-seq
`RedeViz enhance pretreat -i Eye_simple.h5ad --cell-type-label free_annotation --gene-id-label gene_symbol --embedding UMAP --embedding-dim 3 --min-cell-num 2 -o emb_info`

## Simulate RNA points (500,000 nm x 500,000 nm, gap size: 500 nm, spot size: 200 nm, UMI per spot: 1.5)
`RedeViz simulator simulate_by_emb -i emb_info/pretreat.pkl --x-range 500000 --y-range 500000 -o ST_data`

`RedeViz simulator point2spot -i ST_data/RNA_points.tsv -o ST_data --cell-shape-info ST_data/CellShapeInfo.tsv --x-range 500000 --y-range 500000 --gap-size 500 --spot-size 220 --UMI-per-spot 1.5`

## Plot expected gene expression profile in simulated ST data.
`RedeViz simulator extract_simulated_expr --emb-info emb_info/pretreat.pkl --gene-list gene_li.txt --spot-info ST_data/Spot.CellType.tsv --cell-shape ST_data/CellShapeInfo.tsv -o simulated_gene_expr`

`RedeViz posttreatment plot_gene_expr -R simulated_gene_expr/KRT12.simulated.npz -G simulated_gene_expr/KRT19.simulated.npz -B simulated_gene_expr/CD74.simulated.npz -o simulated_gene_expr/KRT12_KRT19_CD74.png`

## In-situ enhancement
`RedeViz enhance build -i emb_info/pretreat.pkl -o index.pkl`

`RedeViz enhance run -s ST_data/spot.tsv -i index.pkl -o ST.augment.tsv`

## Phenotype visualization
`RedeViz posttreatment plot_phenotype --input ST.augment.tsv --output ST.phenotype.png`

## Cell type visualization
`RedeViz posttreatment plot_cell_type --input ST.augment.tsv --color celltype.color.tsv --output ST.celltype.png`

## Segmentation
`RedeViz posttreatment segment --input ST.augment.tsv --smooth-radius 1 --receptive-radius 24 --embedding-radius 5 --merge-embedding-dist 5 --min-spot-per-domain 300 --output small_domain`

`RedeViz posttreatment segment --input ST.augment.tsv --smooth-radius 60 --receptive-radius 200 --embedding-radius 20 --merge-embedding-dist 15 --min-spot-per-domain 4000 --output large_domain`

## Gene expression imputation
`RedeViz posttreatment impute build --index emb_info/pretreat.pkl --sce Eye_simple.h5ad --gene-name-label gene_symbol --output ST.impute.index.pkl`

`RedeViz posttreatment impute run --input ST.augment.tsv --index ST.impute.index.pkl --spot ST_data/spot.tsv --gene-list gene_li.txt --spot-pos-pos-label spot_x_index spot_y_index --spot-UMI-label UMI --output ST_imputation`

`RedeViz posttreatment plot_gene_expr -R ST_imputation/KRT12.imputate.npz -G ST_imputation/KRT19.imputate.npz -B ST_imputation/CD74.imputate.npz -o ST_imputation/KRT12_KRT19_CD74.png`

## Differentiation SISVA
`RedeViz posttreatment compare --input1 ST.augment.ref1.tsv --input2 ST.augment.ref2.tsv --output ST.ref1.ref2.compare.tsv`

Note: Requires in-situ augmentation of the same ST data using different reference data in advance