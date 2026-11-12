"""Converters from other datasets to AgentNet's record format.

  python -m wfemb.prep.sources <dataset> [--raw DIR] [--workers N]

Each writes DATA/sources/<dataset>/; wfemb.prep.extend then cleans the sessions and appends them to the prepared set.
"""
