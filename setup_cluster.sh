# Color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

function red {
    printf "${RED}$@${NC}\n"
}

function green {
    printf "${GREEN}$@${NC}\n"
}

function yellow {
    printf "${YELLOW}$@${NC}\n"
}

yes="y"
no="n"
luis="luis"
pc2="pc2"

echo ""
echo "Hi and welcome to the ${BOLD}AutoML repo template${NORMAL} setup!"
echo "First we'll take care of the cluster."
echo -e "Which cluster are you using, ${BOLD}${RED}luis${NC}${NORMAL} or ${BOLD}${RED}pc2${NC}${NORMAL}?"
read cluster

echo ""
echo -e "What's your username on this cluster? (e.g. ${BOLD}${RED}intexmli${NC}${NORMAL} for PC2)"
read username

echo -e "What's your folder name on this cluster? (e.g. ${BOLD}${RED}teimer${NC}${NORMAL} for PC2) You can leave it blank for LUIS."
read userfoldername

echo $(green "Great, let's get started!")

echo ""
echo $(yellow "Adding general commands to bashrc...")

echo "# >>>>>>>>>>>>>>>>>>>>>>>> AUTO ML REPO TEMPLATE" >> $HOME/.bashrc
echo "Adding 'alias dirsize=du -hs * | sort -h' to bashrc."
echo 'alias dirsize="du -hs * | sort -h"' >> $HOME/.bashrc; 
while read -r line; do 
    echo "Adding $line to bashrc."
    echo $line >> $HOME/.bashrc; 
done < "general_bash_aliases.txt"

echo ""
echo $(yellow "Adding cluster-specific commands to bashrc...")

while read -r line; do 
    echo "Adding $line to bashrc."
    echo $line >> $HOME/.bashrc; 
done < "$cluster/${cluster}_bash.txt"

if [ "$cluster" = "$pc2" ] ; then
    # Definitions for the local machine to use rsync
    line="export PC2USER=$username"
    echo "Adding $line to bashrc."
    echo $line >> $HOME/.bashrc
    if [ -z ${PC2PFS+x} ]; 
        then echo "export PC2PFS=/scratch" >> $HOME/.bashrc; fi
    
    # General definitions
    while read -r line; do
        echo "Adding $line${userfoldername} to bashrc."
        echo $line${userfoldername}'"' >> $HOME/.bashrc;

    echo "Initally loading conda"
    ml lang
    ml Anaconda3
    done < "$cluster/pc2_bash_username.txt"
fi
if [ "$cluster" = "$luis" ] ; then
    echo "Initally loading conda"
    module load Miniconda3
fi
echo "# <<<<<<<<<<<<<<<<<<<<<<<< AUTO ML REPO TEMPLATE" >> $HOME/.bashrc

echo ""
echo $(yellow "Setting up conda...")
conda init
source $HOME/.bashrc
conda config --append envs_dirs $ENVDIR
mkdir $REPODIR
mkdir $ENVDIR
conda env create -f environment.yml
conda activate template

echo ""
echo $(yellow "Now for some cleanup...")
if [ "$cluster" = "$luis" ] ; then
    rm -r "pc2"
    mv "luis/README.md" "LUIS_infos.md"
    mv "luis/cpu_example.sh" "cpu_example.sh"
    mv "luis/gpu_example.sh" "gpu_example.sh"
    rm -r "luis"
else
    rm -r "luis"
    mv "pc2/README.md" "PC2_infos.md"
    mv "pc2/singularity" "."
    mv "pc2/cpu_example.sh" "cpu_example.sh"
    mv "pc2/gpu_example.sh" "gpu_example.sh"
    rm -r "pc2"
fi

echo ""
echo $(green "All done! You won't have to do this again on this cluster, simply re-open your shell and use the command ${BOLD}'make-project'${NORMAL} to make repos from now on.")