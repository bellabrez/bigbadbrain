import argparse

def main(args):
    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--flies', nargs='+', type=str,
        help='Supply fly numbers separated by spaces. Can also supply specific experiments with flynum_exptnum, ie 33_0. Will run on all\
        expts if no expt specified.')
    parser.add_argument('-c', '--channels', choices=['red','green','both'], help='which brain channels to use', default='green', type=str)

    parser.add_argument('-v', '--visual', action='store_true', help='')
    parser.add_argument('--v_bin_size', default=100, type=int, help='')
    parser.add_argument('--v_pre_dur', default=500, type=int, help='')
    parser.add_argument('--v_post_dur', default=1500, type=int, help='')

    parser.add_argument('-b', '--behavior', action='store_true', help='')
    parser.add_argument('--b_signs', nargs='+', choices=['abs', 'df_abs','plus', 'minus', 'df'], type=str)
    parser.add_argument('--b_behaviors', nargs='+', choices=['dRotLabY', 'dRotLabZ', 'dRotLabX', 'dRotLabY', 'speed'], type=str)
    
    parser.add_argument('--datadir', action='store_true', help='append supplied directory to Lukes data directory')
    args = parser.parse_args()
    main(args)