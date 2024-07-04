import argparse

parser = argparse.ArgumentParser(description="Run INDICT generation process")

# Data configuration 
parser.add_argument(
    "--task", type=str, default='mitre', 
    choices={'mitre', 'instruct', 'autocomplete', 'promptinject', 'frr', 'interpreter', 'cvs'}, 
    help="Type of generation tasks to be run",
)
parser.add_argument(
    "--prev_trial", type=str, default=None, 
    help="Path to the last generation iteration",
)

# Agent configuration
parser.add_argument(
    "--strategy", type=str, default='indict_llama', 
    choices={'indict_llama', 'indict_commandr'}, 
    help="Generation strategy",
)
parser.add_argument(
    "--model", type=str, default='llama3-8b-instruct', 
    help='Base model to initialize llm agents',
)

# Generation configuration 
parser.add_argument(
    "--debug", action='store_true', 
    help="Enable this to debug with a single sample",
)
parser.add_argument(
    "--override", action='store_true', 
    help="Enable this to override past generation output",
)
parser.add_argument(
    "--suffix", type=str, default='', 
    help='Suffix to output path',
)