import automl_sphinx_theme

from {{cookiecutter.project_slug}} import copyright, __author__, __version__, __name__


options = {
    "copyright": copyright,
    "author": __author__,
    "version": __version__,
    "name": __name__,
    "html_theme_options": {
        "github_url": "https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_name}}",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    }
}

automl_sphinx_theme.set_options(globals(), options)
