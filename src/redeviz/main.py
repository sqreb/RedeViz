import torch as tr
import sys
import argparse
from redeviz.enhance.parse import parse_enhance_args
from redeviz.simulator.parse import parse_simulate_args
from redeviz.pretreatment.parse import parse_pretreatment_args
from redeviz.posttreatment.parse import parse_posttreatment_args
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def parse_args(args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='RedeViz help')
    enhance_parser = subparsers.add_parser('enhance', help='Enhance ST data')
    augment_subparser = enhance_parser.add_subparsers(help='sub-command help')
    parse_enhance_args(augment_subparser)

    simulator_parser = subparsers.add_parser('simulator', help='Simulate ST data')
    simulator_subparser = simulator_parser.add_subparsers(help='sub-command help')
    parse_simulate_args(simulator_subparser)

    pretreatment_parser = subparsers.add_parser('pretreatment', help='Pre-treatment before RedeViz enhancement')
    pretreatment_subparser = pretreatment_parser.add_subparsers(help='sub-command help')
    parse_pretreatment_args(pretreatment_subparser)

    posttreatment_parser = subparsers.add_parser('posttreatment', help='Post-treatment after RedeViz enhancement')
    posttreatment_subparser = posttreatment_parser.add_subparsers(help='sub-command help')
    parse_posttreatment_args(posttreatment_subparser)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.func(args)

def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
