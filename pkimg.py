import io
import zipfile
import matplotlib.pyplot as plt
from PIL import Image

class PokemonImage():
	def __init__(self, img_zip, ext='png'):
		# open zip archive
		self.img_zip = img_zip[:img_zip.rfind('.')]
		print(self.img_zip)
		self.archive = zipfile.ZipFile(img_zip, 'r')
		#for each image (.png)entry, read
		self.ext = ext
		#print(archive.namelist())
	
	# display, pokemon with id=number 
	def show(self, number):
		bytes = self.archive.read(self.img_zip+'/'+str(number)+'.'+self.ext) 
		imgdata = Image.open(io.BytesIO(bytes))
		imgdata.show()
		
	def close(self):
		plt.close()