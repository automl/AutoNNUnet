# The AutoML Project Template
This is a template you can use to get started on our research clusters and generate new project repositories with important starter code.
A quick overview of current content:

- **Setup script for clusters**, including useful bash aliases and optional conda installation
- **Template for new repos** with default settings for students, research papers and packages
- **Code quality features** like linting, formating and docstring checks
- **Tools for improving team workflows**: github actions, pre-commit, commit standards and testing
- **Our automl documentation** setup built in
- **Standard plotting** functions like performance over time, probability of improvement and DeepCave plots
- **Starter code for slurm** experiment setups with hydra and PyExperimenter
- Integrated **tracking of emissions, time and memory** 
- Suggestions for **quality of life features** like termination handling
- **Integration with GitHub** to push your new project to a sparkling new repo

Further ideas and contributions are welcome!

## Using this Template
You can use this template locally as well as on LUIS or PC2.

<details>
<summary>On the Clusters</summary>
Before using this template on LUIS or PC2, you should run the cluster setup script once. 
Afterwards, you can generate repos freely without touching it again.
It will provide you with some useful commands for your bashrc and make sure everything is generated in the correct places.
Additionally, if you plan on using conda on the cluster you chose, the script will install conda and its paths for you.

#### Cluster setup
Start by cloning this repo. The default paths we'll use are $BIGWORK on LUIS and $SCRATCH/projects on PC2.

For LUIS, run:
```bash
cd $BIGWORK
git clone https://github.com/LUH-AI/automl_repo_template.git
cd automl_repo_template
```

And for PC2:
```bash
cd $PC2PFS/hpc-prf-intexml/<your-username>
mkdir projects
cd projects
git clone https://github.com/LUH-AI/automl_repo_template.git
cd automl_repo_template
```

Now you can start the cluster setup with:
```bash
sh setup_cluster.sh
```

You will be asked a few questions and then the script will tell you what is happening. 
Afterwards, it's a good idea to check your bashrc to make sure everything looks alright and to familiarize yourself with your new shortcuts:
```bash
vim $HOME/.bashrc
```
Feel free to edit the paths and aliases any way you like, but ideally you'll keep the names intact so they work the same on both clusters.

#### GitHub CLI
Something we recommend for convenience is the [GitHub CLI](https://cli.github.com/). 
It's not mandatory, but will make pushing new projects to GitHub easier.
Make sure to have installed it and logged into the client before usage:
```bash
conda install gh --channel conda-forge
gh auth login
```

#### Project Generation

In the setup, we gave you a shortcut to run everything required. From anywhere on the cluster, run:
```bash
make-project
```
</details>

<details>
<summary>Local Usage</summary>
The first step is to clone the template repo:
```bash
git clone https://github.com/LUH-AI/automl_repo_template.git
```

You need to install the cookiecutter-pypackage to generate new packages. 
You can do this in a conda environment or even your base python. If you're not doing this in a conda environment, the generation process can generate you one afterwards:
```bash
pip install cookiecutter
```
This is the only dependency you'll need, but we do recommend the GitHub CLI. 
You can find the installation instructions for your OS [here](https://cli.github.com/).
Make sure to have installed it and logged into the client before usage.

What's left to do is run:
```bash
source automl_repo_template/cookiecutter.sh
```

Note that 'source' has been tested on bash, if you use alternate shells, you might run into issues!
</details>


In either case, generation works the same:
You'll be taken to a questionaire with different options, most of which will be answered by the default you chose (either 'package', 'student' or 'research').
Your own details like e-mail and name will be saved after first use. 
Still, check your answers to make sure everything looks right to you. 
If you don't know what an option means, trusting the default is a good option!
After generating the project, you'll be asked about your project's name again for technical reasons, please enter it exactly as above! 
Then we'll install your dependencies if you choose and take care of initiating Git. 

If you have the gh client installed, the last step is pushing your repo. 
A word of warning here: by default gh will use your private account for this!
To avoid this, type the repo name as 'automl/repo'.
There will be a reminder once you get to that stage.
Afterwards you're all done with the setup and can get to coding!

## Digging into the Code
As you'll see, there are quite a few components generated for you. Here are a few suggestions of where to start:

#### If you first want to explore the code features like termination handling:
  - first go to `cli.py` in the project directory. Here you'll see how we suggest handling emission and time tracking and how your chosen command line interface is implemented.
  - then check out `<project-name>.py` - we use a function from here and also a context manager that can help us handle terminations, e.g. because of an error in the code somewhere.
  - try running your example with a few different inputs, maybe even with errors to see how the tracking and exception handling works
  - your results should include a file named `memray.bin` - this is where all the memory allocation is tracked. Run 'memray summary memray.bin' to get an idea of the memory usage (see their documentation for more options).

#### If you're mainly curious about submitting cluster jobs:
  - there are example bash scripts for the clusters that show you how single submission files can look like, it makes sense to see how this can look for you.
  - look at your bashrc: here you'll find some handy tools to check on the cluster, e.g. cluster-info or squeue-me
  - likely you'll have either chosen PyExperimenter or Hydra for this category - both come with pre-configured options for cluster deployment. Try switching from local execution to the cluster and check on your jobs!
  - the experiment_handling notebook can then help you see where which results are stored and how to load them
  - in the plotting notebook, there'll be some examples for plotting, subsitute the dummy data with your own and check how it looks. 

#### If you want to start coding right away:
  - add dependencies you'll need to `pyproject.toml`
  - add your code to `cli.py` and `<project-name>.py` in the project directory
  - write some testcases, examples and possibly documentation
  - use things like cluster examples, experiment handling and plotting if they come up, ignore or delete else

## Featured Packages
Here we explain some of the included packages you might not now but are very useful!

- pre-commit: checks and formats your code before you commit
- commitzen: CLI, [standardized commit](https://www.conventionalcommits.org/en/v1.0.0/) messages, helps keeping overview of history
- ruff: very fast python linter, replaces flake8, isort and black
- bump-my-version: manages project versioning and modifies project files accordingly (works with tagging)
- memray: memory profiler, can track memory allocations in Python code, native extensin modules and in the interpreter
- codecarbon: tracks CO2 emissions and proposes optimizations

## More on the Generation Options
These are all options in the generation process. Most of them should be self-explanatory, some of them might be new to you:

1. **"project_name"**, default: AutoML-Example-Project"

This is the name of your project and also the top level directory that will be created for it

2. **"project_short_description"**, default: "Python Boilerplate that contains all the code you need to create a Python package."

This is a short description of your project.

3. **"version"**, default: "0.1.0"

You project's version

4. **"command_line_interface"**, default: ["Hydra", "Fire (automating argparse)", "No command-line interface"]

The command line interface you'd like to use. The options are Hydra, Fire (which wraps argparse so you can specify your arguments as you would in a normal function) or no preset.

5. **"use_pyexperimenter"**, default: "n"

Whether to use the PyExperimenter for experiment setup & execution.

6. **"python_version"**, default: "3.10"

The python version you'd like to use.

7. **"author_full_name"**, default: "AutoML Hannover"

You own name. This will be saved as a preset after first use.

8. **"email"**, default: "y.name@ai.uni-hannover.de"

Your own e-mail. This will be saved as a preset after first use.

9. **"github_username"**, default: "LUHAI"

Your own GitHub username. This will be saved as a preset after first use.

10. **"pypi_username"**, default: "automl_hannover"

Either your PyPI username or the default for the group account. This will be saved as a preset after first use.

11. **"demo_code"**, default: "y"

Whether to keep the demo code or not.

12. **"use_pytest"**, default: "y"

Whether to use pytest for testing (including code coverage).

13. **"use_ruff_format"**, default: "y"

Whether to use ruff for linting and formatting.

14. **"use_mypy_typing"**, default: "y"

Whether to use mypy for typechecking.

15. **"use_pydoc_style"**, default: "y"

Whether to use pydocstyle to check docstrings.

16. **"use_docs"**, default: "y"

Whether to include docs.

17. **"use_hd5_for_data_loading"**, default: "n"

Whether to use hd5 for faster data collection - this is likely only relevant for Hydra users and also depends on your hardware. You may need to install additional dependencies outside of this setup.

18. **"add_badges"**, default: "y"

Whether to add badges to the ReadMe (e.g. for passing tests).

19. **"add_citation_file"**, default: "y"

Whether to add a citation file.

20. **"use_release_helpers"**, default: "n"

Whether to include release helpers like bump-my-version or twine.

21. **"use_pypi_deployment_with_travis"**, default: "n"

Whether to use travis for deployment. Note that you will have to configure travis yourself!

22. **"use_github_workflows"**, default: "y"

Whether to use automated workflows for testing, docs, etc.

23. **"create_author_file"**, default: "y"

Whether to create a separate author file.

24. **"open_source_license"**, default: "BSD license"

The license to use. Please note that the BSD license should be the default unless you have good reasons to deviate!

25. **"project_slug"**, default: "{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}"

How your project will be imported in code (e.g. 'hello_world' instead of the name Hello-World). We recommend using the default for this.