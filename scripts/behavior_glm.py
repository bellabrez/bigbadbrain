import os
import sys

def main(args):
    print(args[0])
    print(args[1])
    print(args[2])
    print(args[3])
    sys.stdout.flush()

if __name__ == '__main__':
    main(sys.argv[1:])