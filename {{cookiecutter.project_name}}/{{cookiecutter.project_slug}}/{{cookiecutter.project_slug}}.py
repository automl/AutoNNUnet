"""Main module."""
from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class HandleTermination:
    """This is a context manager that handles different termination signals.
    The comments show how to save a model and optimizer states
    even if there's an error in the code.
    """

    def __init__(self, directory):
        """Initialize the context manager with logdir."""
        # self.model = model
        # self.optimizer = optimizer
        self.directory = Path(directory)

    def __enter__(self):
        self.old_sigterm_handler = signal.signal(signal.SIGTERM, self.handle_sigterm)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """An exception was raised, so save the model and optimizer states
        with an exception tag before re-raising.
        """
        signal.signal(signal.SIGTERM, self.old_sigterm_handler)
        if exc_type is not None:
            logger.info("Oh no, there was an exception!")
            # torch.save({
            #    'model_state_dict': self.model.state_dict(),
            #    'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.directory / f'checkpoint_exc.pth')

        # Everything ran perfectly, so save the final model and optimizer states
        if exc_type is None:
            logger.info("All clear!")
            # torch.save({
            #    'model_state_dict': self.model.state_dict(),
            #    'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.directory / f'checkpoint_final.pth')

        return False

    def handle_sigterm(self, signum, frame): # noqa: ARG002
        """Save the model and optimizer states before exiting."""
        logger.info(f"Saving checkpoint to {self.directory}/checkpoint_sigterm.pth")
        # torch.save({

        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        # }, self.directory / f'checkpoint_sigterm.pth')
        sys.exit(0)

{% if cookiecutter.use_pyexperimenter == 'y' %}
def cool_things(parameters, result_processor, custom_config):{% else %}
def cool_things(cfg):{%- endif %}
    """As the name suggests, this does cool things."""
    with HandleTermination(os.getcwd()):
        {%- if cookiecutter.command_line_interface|lower == 'hydra' and not cookiecutter.use_pyexperimenter == 'y' %}
        print(f"Your current config is: {cfg}"){%- endif %}
        {% if cookiecutter.use_pyexperimenter == 'y' %}
        result_processor.process_logs({"train_scores": {"epoch": 1, "train_loss": 0.1, "train_acc": 0.2, "val_loss": 0.3,"val_acc": 0.4}})
        print(f"Your current parameters are: {parameters}")
        print(f"Your current custom config is: {custom_config}")
        result_processor.process_results({"done": "yes"}){%- endif %}

