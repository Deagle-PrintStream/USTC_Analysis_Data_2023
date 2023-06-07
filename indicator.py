
__all__=["indicators"]

import pandas as pd
from math import sqrt
import os,sys

os.chdir(sys.path[0])

def indicators(df:pd.DataFrame,rule:set[str],consqent:str="R1",col_name:str="REPEAT")->dict[str,float|int] | None:
	P_x:float=0
	P_y:float=0
	P_x_y:float=0
	P_x_and_y:float=0

	total:float=len(df)
	y_count=len(df.loc[df[col_name]==consqent])
	P_y=y_count/total

	x_count=0
	x_and_y_count=0

	for i in range(0,len(df)):
		combination=set(df.iloc[i])
		if rule.issubset(combination):
			x_count+=1
			if consqent in combination:
				x_and_y_count+=1

	P_x=x_count/total
	P_x_and_y=x_and_y_count/total
	P_x_y=P_x_and_y/P_y

	lift:float=P_x_y/P_y
	interest:float=P_x_and_y/(P_x*P_y)
	PS:float=P_x_and_y-P_x*P_y
	phi_coef:float=PS/(sqrt(P_x*P_y*(1-P_x)*(1-P_y)))

	return dict({"lift":lift,"interest":interest,"PS":PS,"phi_coef":phi_coef})




