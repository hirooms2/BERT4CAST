#!/bin/bash
python main.py --max_body_len=100 --name=roBERTa_0 --reg_term=0
python main.py --max_body_len=100 --name=roBERTa_001 --reg_term=0.001
python main.py --max_body_len=100 --name=roBERTa_002 --reg_term=0.002
python main.py --max_body_len=100 --name=roBERTa_005 --reg_term=0.005
python main.py --max_body_len=100 --name=roBERTa_01 --reg_term=0.01
python main.py --max_body_len=100 --name=roBERTa_02 --reg_term=0.02
python main.py --max_body_len=100 --name=roBERTa_05 --reg_term=0.05
python main.py --max_body_len=100 --name=roBERTa_1 --reg_term=0.1
