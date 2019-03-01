from argparse import ArgumentParser


def get_option(epoch):
    argparser = ArgumentParser()
    argparser.add_argument('-e', '--epoch', type=int,
                           default=epoch,
                           help='Specify number of epoch')
    return argparser.parse_args()
