set PYTHONPATH=C:\Users\15291\ICLR2024\
FOR /L %%i IN (5,1,15) DO (
   python example_tos_Matern.py --num_points %%i
)
