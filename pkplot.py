import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


class PokePlot():
	
	def __init__(self, rows=1, cols=1, uniformX=False,uniformY=False):
		self.rows, self.cols = rows,cols
		self.fig, self.axes = plt.subplots(rows, cols, sharex=uniformX,sharey=uniformY)
		
		
	
	def scatter(self,
					x, y, 										
					row=1,col=1,								
					xlabel	= 'X', 		ylabel	= 'Y',			
					sizes	= None,		colors	= (1,0,0,.75), 	
					format	= 'o', 		title	= 'Scatter'):	
		# Fixing random state for reproducibility
		if self.rows == 1 and self.cols==1:
			ax = self.axes
		else:
			ax = self.axes[row][col]

		#sizes = ms?
		ax.scatter(x,y, marker=format, c=colors)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		
	def show(self):
		plt.show()
		
	