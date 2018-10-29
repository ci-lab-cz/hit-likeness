#!/usr/bin/env python3

import os
import time
import pickle
import argparse
from subprocess import Popen, PIPE


def get_ntrees_per_node(ntrees, nnodes):
    a = ntrees % nnodes
    if not a:
        return [int(ntrees / nnodes)] * nnodes
    else:
        b = ntrees // nnodes
        return [b + 1] * a + [b] * (nnodes - a)


def get_jobs(job_ids, unfinished=True):
    unfinished_jobs = []
    p = Popen(['qstat'], stdout=PIPE, encoding='utf8')
    outs, errs = p.communicate(timeout=30)
    lines = outs.split('\n')
    if len(lines) > 2:
        for line in lines[2:]:
            job = [s for s in line.strip().split(' ') if s]  # Job_id Name User Time_Use S Queue
            if job and job[0] in job_ids and job[4] != 'E':
                unfinished_jobs.append(job[0])
    if unfinished:
        return unfinished_jobs
    else:
        return list(set(job_ids).difference(unfinished_jobs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a decision tree.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-o', '--output', metavar='output.pkl', required=False, default=None,
                        help='pickled tree object (networkx). If missing the file will be stored with automatically '
                             'generated name in the dir with the descriptor input file. Default: None.')
    parser.add_argument('-t', '--ntree', metavar='INTEGER', required=False, default=50,
                        help='number of trees to build. Default: 50.')
    parser.add_argument('-m', '--nvar', metavar='INTEGER', required=False, default=3,
                        help='number of randomly chosen variables used to split nodes. '
                             'Values 0 and less indicate to use all variables. Default: 3.')
    parser.add_argument('-s', '--nsamples', metavar='INTEGER', required=False, default=0.67,
                        help='portion of randomly chosen compounds to train each tree. Should be greater than 0 and '
                             'less or equal to 1. Default: 0.67.')
    parser.add_argument('-p', '--min_parent', metavar='INTEGER', required=False, default=3000,
                        help='minimum number of items in parent node to split. Default: 3000.')
    parser.add_argument('-n', '--min_child', metavar='INTEGER', required=False, default=1000,
                        help='minimum number of items in child node to create. Default: 1000.')
    parser.add_argument('-a', '--algorithm', metavar='INTEGER', required=False, default=2,
                        help='the number of a splitting algorithm. Default: 2.')
    parser.add_argument('-d', '--nnodes', metavar='INTEGER', required=False, default=10,
                        help='number of computational nodes used to built a forest. Default: 10.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, default=32,
                        help='number of CPUs per node used to built a forest. Default: 32.')
    parser.add_argument('-e', '--env', metavar='CONDA_ENV_NAME', required=False, default='rdkit-1709',
                        help='name of the conda environment. Default: rdkit-1709.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = os.path.abspath(v)
        if o == "y": y_fname = os.path.abspath(v)
        if o == "ntree": ntree = int(v)
        if o == "nvar": nvar = int(v)
        if o == "nsamples": nsamples = float(v)
        if o == "min_child": min_child_num = int(v)
        if o == "min_parent": min_parent_num = int(v)
        if o == "output": out_fname = v
        if o == "algorithm": algorithm = int(v)
        if o == "verbose": verbose = v
        if o == "ncpu": ncpu = int(v)
        if o == "nnodes": nnodes = int(v)

    if out_fname is None:
        out_fname = os.path.join(os.path.dirname(x_fname),
                                 "forest_%s_t%i_v%i_p%i_c%i_alg%i.pkl" %
                                 (os.path.basename(x_fname).rsplit('.', 1)[0],
                                  ntree,
                                  nvar,
                                  min_parent_num,
                                  min_child_num,
                                  algorithm))
    out_fname = os.path.abspath(out_fname)

    job_dir = os.path.dirname(out_fname)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    job_ids = dict()
    for i, trees in enumerate(get_ntrees_per_node(ntree, nnodes)):
        pbs_name = os.path.join(job_dir, out_fname.replace('.pkl', '_batch%i.pbs') % i)
        script = """
        #!/bin/bash
        #PBS -l select=1:ncpus=%i
        #PBS -k eo
        source activate rdkit-1709
        cd %s
        """ % (ncpu, os.path.abspath(__file__))
        batch_fname = out_fname.replace('.pkl', '_batch%i.pkl') % i
        script += f'python3 forest.py -x {x_name} -y {y_fname} -o {out_fname} -t {trees} -m {nvar} ' \
                  f'-p {min_parent_num} -n {min_child_num} -c {ncpu} -a {algorithm}'
        with open(pbs_name, "wt") as f:
            f.write(script)
        p = Popen(['qsub', pbs_name], stdout=PIPE, encoding='utf8')
        outs, errs = p.communicate(timeout=30)
        id = outs.strip()
        print("job id: %s was submitted" % outs.strip())
        print(errs)
        job_ids[id] = batch_fname

    while get_jobs(job_ids, unfinished=True):
        time.sleep(20)

    forest = []
    for fname in job_ids.values():
        forest.extend(pickle.load(open(fname, 'rb')))
    pickle.dump(forest, open(out_fname, 'wb'))
