============================= test session starts ==============================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /app
configfile: pyproject.toml
plugins: md-report-0.7.0, anyio-4.9.0, benchmark-4.0.0, cov-4.1.0
collected 13 items

tests/test_evaluate_expression.py s                                      [  7%]
tests/test_fleks.py sssssssssss.                                         [100%]


-------------------------------------------- benchmark: 1 tests --------------------------------------------
Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
------------------------------------------------------------------------------------------------------------
test_load             4.9527  5.5062  5.0983  0.1003  5.0770  0.1133      25;4  196.1435     121           1
------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
======================== 1 passed, 12 skipped in 4.59s =========================
|             filepath              | [92mpassed[0m | [93mskipped[0m | SUBTOTAL |
| --------------------------------- | -----: | ------: | -------: |
[48;2;32;32;32m|[0m[48;2;32;32;32m[93m tests/test_evaluate_expression.py [0m[48;2;32;32;32m|[0m[48;2;32;32;32m[90m      0 [0m[48;2;32;32;32m|[0m[48;2;32;32;32m[93m       1 [0m[48;2;32;32;32m|[0m[48;2;32;32;32m[93m        1 [0m[48;2;32;32;32m|[0m
[40m|[0m[40m[93m tests/test_fleks.py               [0m[40m|[0m[40m[92m      1 [0m[40m|[0m[40m[93m      11 [0m[40m|[0m[40m[93m       12 [0m[40m|[0m
[48;2;0;0;0m|[0m[48;2;0;0;0m[93m TOTAL                             [0m[48;2;0;0;0m|[0m[48;2;0;0;0m[92m      1 [0m[48;2;0;0;0m|[0m[48;2;0;0;0m[93m      12 [0m[48;2;0;0;0m|[0m[48;2;0;0;0m[93m       13 [0m[48;2;0;0;0m|[0m
