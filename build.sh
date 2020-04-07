#!/bin/bash

python compile.py build_ext --inplace
while [ "$1" != "" ]; do
    case $1 in
        -t | --test)
		python -m unittest unit_tests.py
           ;;
        -h | --help)
            exit 1
            ;;
	-tv | --test-verbose)
		python -m  unittest unit_tests.py --verbose
		;;
        * )
            usage
            exit 1
    esac
    shift
done
