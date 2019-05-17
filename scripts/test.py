from time import sleep
import sys

print('in code!')
for i in range(21):
    #print('i: {}'.format(i))
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    sleep(10)
