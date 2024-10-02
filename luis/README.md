# Working on LUIS
You'll need a LUIS account as well as access from the university network to access LUIS. There's a [documentation](https://docs.cluster.uni-hannover.de/doku.php) with a lot of information about the cluster infrastructure and usage. We collected some important points below, but do take a look!

There's also a [quickstart guide](https://docs.google.com/document/d/1KHCfJ2ikdw-fqRa_H1kEmwHHjOiJ04HvPR_9LDgGOSQ/edit?usp=sharing) with some more useful slurm commands. 

## Queues & Resources
We have access to the ai, tnt and ainlp queues - though the latter should only be used for GPU jobs and not by students.
These include the following resources: 

| Queue    | Nodes | Cores per node  | GPUs                          | Timelimit  |
| -------- | ----- | --------------- | ----------------------------- | ---------- |
| ai       | 6     | 2x254, 4x128    | 2x 32 A100 @ 20GB             | 8-08:00:00 |
| ainlp    | 2     | 1x56, 1x128     | 1x 12 A100 @ 20GB , 1x 8 A100 | 8-08:00:00 |
| tnt      | 6     | 112             | 5x 8 RTX3090, 1x 8 RTXA6000   | 8-08:00:00 |

As for disk resources, you have three directories: $HOME, $BIGWORK and $PROJECT. The command `checkquota` will show you how much disk space is available and how many files you're allowed to keep there.

## Package Management
There are two important package categories to remember on LUIS: user packages and global installs.
Global installs are things like CMake that you might need, but can't install yourself. There's an [overview](https://docs.cluster.uni-hannover.de/doku.php?id=guide:modules_and_application_software) on the LUIS documentation, but you have to load them yourself when you want to use them. This looks like so:

```
module load GCC/10.3.0
```

For user packages, e.g. python, it's HIGHLY recommended to keep them in $BIGWORK to avoid running out of memory in $HOME. The only thing this changes, is the location of the conda environments, in usage it will be identical. So as long as your bashrc contains the correct conda path, you won't have to worry at all.
You'll still have to activate your conda environment before running jobs, just like you normally would:

```
conda activate <your_env>
```

Our example runscripts include both conda and module loading the way you should probably do it in your own scripts.

## Interactive Jobs
If you want to debug something on the cluster or monitor a job manually, you can use an interactive job to run your commands on a node directly. You can request resources in the same way as you would do in a slurm bash script:
```
salloc --time=1:00:00 --nodes=1 --ntasks=16 --mem-per-cpu=4G
```

## Data Storage
You have a limited amount of storage - especially $HOME is fairly small. Be aware that as soon as you go over the hard limit or use too much memory for too long, you'll be unable to write to disk! Therefore you should aim to work on $BIGWORK as much as you can and clean up after yourself, e.g. by archiving projects once they're done, ideally somewhere else than the cluster.

Obviously you can also store data on $PROJECT. You should make sure not to monopolize this directory, though. It has a limit of 10TB, so there should be enough space for everyone.

You can also go over your limits by having too many files (although that's harder) and large error files, even if they're in tmp and not immediately accessible to you, count towards these limits as well. Therefore you should be careful not to spam error messages in cluster jobs, not remove error files without deleting the corresponding jobs (as then you won't be able to delete the log written to tmp anymore!) and not write too many very small files. If you run into these issues regularly, revising your code structure might be a good idea.

## Permissions
Your own $BIGWORK and $HOME directories belong to you. Others won't have read or write access there. In $PROJECT, a directory belongs to whoever creates it - that means if you want to share data, you need to explicitly set the permissions there! A good option that includes subdirs is:

```
cmod -R 770
```

Important: the compute nodes have more access restrictions than login! That means you won't be able to access the internet, most ports or $PROJECT in your jobs!

## Best Practices
- Use $HOME as little as possible
- Check your quotas regularly
- Don't reserve whole nodes if you don't actually need them 
- SSH in your IDE, e.g. VSCode so you can work on LUIS end-to-end as long as you stay within your data budgets