import json
from redeviz.simulator.cell import CellMapInfo, CellMap


def simulate_by_cfg_main(args):
    with open(args.cfg , "r") as f:
        cfg = json.load(f)
    cell_map_info = CellMapInfo(cfg)
    cell_map = CellMap(cell_map_info)
    cell_map.simulate_all()
    cell_map.write_dataset(args.output)

