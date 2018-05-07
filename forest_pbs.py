#!/usr/bin/env python3

import os
from subprocess import Popen, PIPE

x_fname = '/home/pavlop/imtm/hts/x.txt'
y_fname = '/home/pavlop/imtm/hts/y.txt'
dir_name = '/home/pavlop/imtm/hts/forest_p3000_c1000_alg3_m3_t5_realx'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for i in range(19):
    pbs_name = os.path.join(dir_name, "forest_job_%i.pbs" % i)
    script = """
    #!/bin/bash
    #PBS -l select=1:ncpus=32:mem=10gb
    #PBS -k eo
    source activate rdkit-1709
    cd /home/pavlop/python/tree
    python3 forest_mp.py -x %s -y %s -o %s -t 25 -m 3 -p 3000 -n 1000 -c 25 -a 3
    """ % (x_fname,
           y_fname,
           os.path.join(dir_name, 'forest_%i.pkl' % i))
    with open(pbs_name, "wt") as f:
        f.write(script)
    p = Popen(['qsub', pbs_name], stdout=PIPE, encoding='utf8')
    outs, errs = p.communicate(timeout=30)
    print("job id: %s was submitted" % outs.strip())
    print(errs)
