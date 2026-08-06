date < NUL
time < NUL
set PYTHONIOENCODING=utf-8
python -OO completeness.py  > completeness.txt
python -OO incompleteness.py > incompleteness.txt
python -OO delay_bounds.py > delay_bounds.txt
python -OO runtime.py > runtime.txt
date < NUL
time < NUL
