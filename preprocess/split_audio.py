##########################################
##########################################
#Splits audio files into x second segments
#To change x, go to line 13
##########################################


# from pydub import AudioSegment
import os
from os import listdir
from os.path import isfile, join

segment_length = 3
num_segments = 30/segment_length


#Trim audio based on start time and duration of audio.
def trim_audio(input_audio_file,output_audio_file,start_time,duration):
	# print "input file:        ", input_audio_file
	# print "output file:       ", output_audio_file
	cmdstring = "sox %s %s trim %s %s" %(input_audio_file,output_audio_file,start_time,duration)
	os.system(cmdstring)


## current path of the directory
mypath = "/home/sabith/hmm-rnn/data/genres"
print (mypath)

## all categories in the current folder
categories = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

## lists all files for each category
for cat in categories:
	print ("CURRENT CATEGORY:", cat)
	# goes to folder of category
	cur_path = mypath + "/" + cat
	
	# all the audio files in the catgory
	cat_files = [f for f in listdir(cur_path) if isfile(join(cur_path, f))]
	
	# segments each audio file and outputs to directory cat_{length_of_segment}
	for aud in cat_files:

		# current audio file
		in_path = cur_path + "/" + aud
		
		# creates number of segments
		for segment in range (0,num_segments):
			out_path = cur_path+"/"+cat+"_"+str(segment_length)+"/"+str(segment)+aud
			trim_audio(in_path, out_path, segment*segment_length, segment_length)

	
	

	




#print (onlyfiles)