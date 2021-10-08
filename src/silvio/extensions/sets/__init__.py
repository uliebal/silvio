"""
Sets are themed collections of various tools and utilities. They allow for developers to keep
related features in the same directory, but ideally these features should all be generalized and
distributed among the `extension.{modules|records|tools|utils}` sub-packages.

Use a `set` if you are not able to fully integrate the features in a generalist manner.
Sub-packages inside `sets` are meant to be internal only and you may make them available for public
by writing acessors in `extension.{modules|records|tools|utils}` files.
"""