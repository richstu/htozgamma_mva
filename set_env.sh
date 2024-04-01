source /homes/jbkim/Linux/el7_v2/root-6.30.04/bin/thisroot.sh
export LD_LIBRARY_PATH=/homes/jbkim/Linux/el7_v2/lib:/homes/jbkim/Linux/el7_v2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export WORK_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PATH=/homes/jbkim/Linux/el7_v2/bin:$WORK_DIR/scripts:${PATH:+:${PATH}}
