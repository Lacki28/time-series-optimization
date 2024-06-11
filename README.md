# time-series-optimisation

Time series optimisation for cloud workloads is a known challenge, in this repository we will test different methods to
accurately predict cloud workload using the Google cluster trace (
2011 ) [https://research.google/resources/datasets/cluster-workload-traces/].

# How to run

python3 -m venv .venv && source .venv/bin/activate <br>
pip install -r requirements.txt && pip install "ray[train]" <br>
export PYTHONPATH=$PYTHONPATH:$(pwd) <br>
cd ./models/ <br>
python3 lstm.py <br>