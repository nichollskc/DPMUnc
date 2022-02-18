# See if a name for the conda environment has been passed
if [ -z "$UNC_CLUST_CONDA_ENV" ]
then
    UNC_CLUST_CONDA_ENV='DPMUnc'
    echo "No environment name supplied, using default of '$UNC_CLUST_CONDA_ENV'"
fi

# Ensure conda activate command is available
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Try to activate environment - use 'cmd || echo' format to avoid CircleCI build failing
# if the environment doesn't exist - CircleCI runs in -e mode so any failing line causes
# the script to fail
echo "Attempting to activate environment $UNC_CLUST_CONDA_ENV"
unset UNC_CLUST_ACTIVATE_RETVAL
conda activate $UNC_CLUST_CONDA_ENV || { UNC_CLUST_ACTIVATE_RETVAL=$?; \
                                        echo "'conda activate $UNC_CLUST_CONDA_ENV' failed"; }

# Check if the conda activate command failed (i.e. does RETVAL variable exist?)
# If so, this should only be because the environment doesn't exist
if [[ -z "$UNC_CLUST_ACTIVATE_RETVAL" ]]
then
    echo "Conda environment $UNC_CLUST_CONDA_ENV activated. Setup finished"
else
    if [[ $(which mamba) ]]
    then
        CONDA_ALTERNATIVE="mamba"
    else
        CONDA_ALTERNATIVE="conda"
    fi
    echo "Conda environment $UNC_CLUST_CONDA_ENV doesn't exist yet. Installing it using $CONDA_ALTERNATIVE"
    $CONDA_ALTERNATIVE env create -f environment.yml -n $UNC_CLUST_CONDA_ENV
    echo "Installed environment $UNC_CLUST_CONDA_ENV"
fi
