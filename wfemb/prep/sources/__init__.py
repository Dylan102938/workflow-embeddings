"""Converters from other datasets to AgentNet's record format.

  python -m wfemb.prep.sources <dataset> [--download] [--raw DIR] [--workers N]

Each writes DATA/added/sources/<dataset>/; wfemb.prep.add_datasets then cleans the sessions and prepares them.
"""
