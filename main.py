##python3 -m pip install numpy pillow matplotlib scikit-learn
import numpy as np
from random import randint

from pkimg import PokemonImage
from pokedex import Pokedex
from pkplot import PokePlot

ALL = 0.5 # proportion to add to training 

tInt 			= 'i8'
tFloat 			= 'f8'
tStr			= 'U20'
ImgCsv 		= 'pokemon.zip'  # https://www.kaggle.com/kvpratama/pokemon-images-dataset
DataCsv	 	= 'pokemon.csv'  # https://www.kaggle.com/alopex247/pokemon



Number		= 'Number'	
Name		= 'Name'
Type1		= 'Type1' 	
Type2		= 'Type2' 		
TotalStat	= 'TotalStat'
HP			= 'HP'
Attack		= 'Attack'	
Defense		= 'Defense'	
SpAtk		= 'SpAtk'		
SpDef		= 'SpDef'		
Speed		= 'Speed'		
Generation	= 'Generation'		
IsLegendary	= 'IsLegendary'
Color		= 'Color'
HasGender	= 'HasGender'
ProbMale	= 'ProbMale' 
EggGroup1	= 'EggGroup1'
EggGroup2	= 'EggGroup2'
HasMegaEvol = 'HasMegaEvol'
HeightM		= 'HeightM'	
WeightKG	= 'WeightKG'	
CatchRate 	= 'CatchRate'	
BodyStyle 	= 'BodyStyle'

#create str to int map for tStr
PKMN = {
			Number		:	(	Number		 , tInt		),	
			Name		:	(	Name		 , tStr		),
			Type1		: 	(	Type1		 , tStr		),	
			Type2		: 	(	Type2		 , tStr		),		
			TotalStat	:	(	TotalStat	 , tStr		),	
			HP			:	(	HP			 , tInt		),
			Attack		:	(	Attack		 , tInt		),	
			Defense		:	(	Defense		 , tInt		),
			SpAtk		:	(	SpAtk		 , tInt		),	
			SpDef		:	(	SpDef		 , tInt		),
			Speed		:	(	Speed		 , tInt		),	
			Generation	:	(	Generation	 , tInt		),
			IsLegendary	: 	(	IsLegendary	 , tStr		),	
			Color		: 	(	Color		 , tStr		),
			HasGender	: 	(	HasGender	 , tStr		),	
			ProbMale	:	(	ProbMale	 , tFloat	),
			EggGroup1 	:	(	EggGroup1 	 , tStr		),	
			EggGroup2	:	(	EggGroup2	 , tStr		),
			HasMegaEvol	:	(	HasMegaEvol	 , tStr		),	
			HeightM		:	(	HeightM		 , tFloat	),
			WeightKG	:	(	WeightKG	 , tFloat	),	
			CatchRate	:	(	CatchRate	 , tFloat	),
			BodyStyle	: 	(	BodyStyle	 , tStr		)		
		}				
	

def encounter(pkmn):



	# per generation, get average total state
	return pkmn

def startJourney(filename):

	pkmn = np.genfromtxt(filename, skip_header=1, dtype=list(PKMN.values()), delimiter=',')
	#input(pkmn)
	pokedex  = Pokedex(pkmn)
	pokedex.catchem(ALL)
	
	# train and score on specific categories
	categories = [PKMN[Attack], PKMN[SpAtk],PKMN[SpDef],PKMN[Defense], PKMN[Speed], PKMN[HeightM], PKMN[WeightKG]]
	results 	= PKMN[Type1] # 1 dimensional
	pokedex.train(categories, results)
	score = pokedex.score(categories, results) 
	
	print(f'Pokedex Score : {score}')
	return
	
	
	pkimg = PokemonImage(ImgCsv)
	pkplot = PokePlot(rows=1, cols=1)
	x = data[Number]
	y = data[TotalStat]
	
	hp = normalize(data[HP])
	zeros = np.zeros(hp.size*2)[:,None].reshape((hp.size,2))
	ones = np.ones(hp.size)[:,None]
	col = hp[:, None]
	print (ones)
	trail = np.append(zeros,ones, axis=1)
	
	colors = np.append(col,trail, axis=1)
	
	#colors = colors.reshape((hp.size, 3))
	input(colors)
	pkplot.scatter(	x,  y, colors=colors)
	pkplot.show()
	#pkimg.show(randint(1, len(data)))
	
def normalize(x):
	min,max = np.array([x]).min(),np.array([x]).max()
	bound = max-min
	return np.array(list(map(lambda v : (v-min)/bound, x ) ) ) 
	
	
def numberToColor(number,ceil):
	# maps from color to number
	c = [0,0,0,1]
	i = 0
	while number > 0:
		n = number%10
		number = number/10
		c[i] += (n*i	% ceil)
		#c[i] %= 255
		i+=1
		i%=3
	return normalize(c)

if __name__ == '__main__':
	startJourney(DataCsv)
	
