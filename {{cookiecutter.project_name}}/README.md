# {{ cookiecutter.project_name }}
{% if cookiecutter.add_badges == 'y' %}
[![PyPI Version](https://img.shields.io/pypi/v/{{cookiecutter.project_slug}}.svg)](https://pypi.python.org/pypi/{{cookiecutter.project_slug}})
[![Test](https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/actions/workflows/pytest.yaml/badge.svg)](https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/actions/workflows/pytest.yaml)
[![Doc Status](https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/actions/workflows/docs.yaml/badge.svg)](https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}/actions/workflows/docs.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
{% endif %}

{{ cookiecutter.project_short_description }}

{% if cookiecutter.open_source_license != 'Not open source' -%}
- Free software: {{ cookiecutter.open_source_license }}
- Documentation: https://{{ cookiecutter.project_slug | replace("_", "-") }}.readthedocs.io.
{% endif %}

## Features

- TODO

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
