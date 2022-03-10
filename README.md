# An Accelerator for Format Agnostic Tensor Expressions (FATE)

This is a discrete event simulator for the FATE Accelerator using 'simpy'.

Steps to run:

Head over to the cloned repositroy and do the following:
1. `pip install -e .`
2. `pip install -r requirements.txt`
3. `cd fate`
4. `python3 TB/tests/test_elementwise_multiplication.py` runs a simple FATE graph on the accelerator.

Use `parameters.py` to change any of the accelerator parameters. 
