"""Console script for {{cookiecutter.project_slug}}."""
from __future__ import annotations

import sys

import memray
from codecarbon import track_emissions
{%- if cookiecutter.command_line_interface|lower == 'fire' %}
import fire{%- endif %}
{%- if cookiecutter.command_line_interface|lower == 'hydra' %}
import hydra{%- endif %}
{%- if cookiecutter.use_pyexperimenter|lower == 'y' %}
from py_experimenter.experimenter import PyExperimenter{%- endif %}

from {{cookiecutter.project_slug}} import cool_things
{%- if cookiecutter.command_line_interface|lower == 'fire' %}
@track_emissions(offline=True, country_iso_code="DEU")
def main(args={}):
    """Console script for {{cookiecutter.project_slug}}."""
    with memray.Tracker("memray.bin"):
        cool_things(args)
        return 0{%- endif %}
{% if cookiecutter.command_line_interface|lower == 'hydra' and cookiecutter.use_pyexperimenter == 'n' %}
@hydra.main(version_base=None, config_path="configs", config_name="base")
@track_emissions(offline=True, country_iso_code="DEU")
def main(cfg):
    """Console script for {{cookiecutter.project_slug}}."""
    with memray.Tracker("memray.bin"):
        print(f"Hello, I am a test! My ID is {cfg.id}")
        print("\n")
        print("Add arguments to this script like this:")
        print("     'python {{cookiecutter.project_slug}}/cli.py +hello=world'")
        print("Or use a yaml config file to store your arguments.")
        print("See click documentation at https://hydra.cc/docs/intro/")
        print("\n")
        cool_things(cfg)
        with open("./performance.csv", "w+") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            f.write("1,0.1,0.2,0.3,0.4\n")
        with open("./done.txt", "w+") as f:
            f.write("yes")
        return 0
{%- endif %}
{% if cookiecutter.use_pyexperimenter == 'y' %}
def main():
    """Console script for {{cookiecutter.project_slug}}."""
    experimenter = PyExperimenter(experiment_configuration_file_path="configs/base.cfg", name='example')
    experimenter.fill_table_from_config()
    experimenter.get_table()
    experimenter.execute(cool_things, max_experiments=-1)
    experimenter.get_table()
    return 0
{%- endif %}

if __name__ == "__main__":
    {%- if cookiecutter.command_line_interface|lower == 'fire' %}
    sys.exit(fire.Fire(main)){%- else %}
    sys.exit(main())  # pragma: no cover
    {%- endif %}