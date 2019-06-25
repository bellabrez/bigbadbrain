import os
import sys

def main(args):
    print(args[0])
    sys.stdout.flush()

if __name__ == '__main__':
    main(sys.argv[1:])