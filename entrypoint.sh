#!/bin/bash

# Pass parameters to the container when run
# Maybe a bad idea?
accelerate launch train.py "$@"
