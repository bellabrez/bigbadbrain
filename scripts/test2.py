from time import sleep
import sys

print('in code!')
for i in range(21):
    print('i: {}\n'.format(i))
    sys.stdout.flush()
    sleep(5)
