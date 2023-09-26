set PYTHONPATH=C:\Users\15291\ICLR2024\
FOR /L %%i IN (2,1,4) DO (
   python example_sin_motivation.py --num_points %%i
)
