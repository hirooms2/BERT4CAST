#!/bin/bash
python main.py --max_body_len=100 --batch_size=32 --reg_term=1
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.5
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.2
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.1
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.01
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.02
python main.py --max_body_len=100 --batch_size=32 --reg_term=0.05
python main.py --max_body_len=100 --batch_size=32 --reg_term=0