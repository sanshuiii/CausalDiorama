#!/usr/bin/env bash
# Start (or confirm) the runner of one GPU node; run on that node. Queue and status: @CD_QUEUE@/{queue,status}/<node>.
# The runner owns all 8 GPUs of the node and runs the queued tasks one after another (shared/tools/runner.py).
NODE=${1:?node}; Q=@CD_QUEUE@; W=@CD_NODE@/compbench_v1
mkdir -p $W/tools $W/logs $Q/queue/$NODE/main $Q/status/$NODE
rsync -a @CD_SHARED@/tools/ $W/tools/
if [ -f $W/runner.pid ] && kill -0 "$(cat $W/runner.pid)" 2>/dev/null; then echo "runner already running pid=$(cat $W/runner.pid)"; exit 0; fi
rm -f $Q/queue/$NODE/STOP $Q/queue/$NODE/RESTART
RUNNER_SHARED=$Q setsid nohup @CD_NODE@/cwm_h3_reproduction_v1/.venv/bin/python $W/tools/runner.py $NODE >> $W/logs/runner.out 2>&1 < /dev/null &
echo $! > $W/runner.pid; sleep 3; echo "runner started pid=$(cat $W/runner.pid)"; tail -2 $W/logs/runner.out
