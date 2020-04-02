#!/bin/bash

python compile.py build_ext --inplace
while [ "$1" != "" ]; do
    case $1 in
        -t | --test)
           python -m unittest tests.py
           ;;
        -h | --help)
            exit 1
            ;;
        * )
            usage
            exit 1
    esac
    shift
done
