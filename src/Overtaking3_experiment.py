# Jiuyang Bai Overtaking_v1.3_Tomato3
'''
baijiuyang@hotmail.com
10/03
Revision: the condition, stimuli, and procedure are changed.
When the trial start, the orientation pole will turn green. And participant will walk to it.
The leader no longer appears right at the beginning of the trial but some time between 3~4 seconds 
1 meter in front of the participant. The leader will move at a constant speed v0
This version does not use recorded instruction anymore. The experimenter will read the instructions.

Freewalk output columns: participant x y z, participant yaw pitch row, time
Experimental trial output columns: leader x y z, participant x y z, participant yaw pitch row, time
'''


import math
import lights # turns on the lights
import random
import csv
#from os.path import exists
import os
import steamvr
# Vizard Imports
import viz
import vizact
import viztracker


# The following libraries are won't work outside of the VENLab without their 
# associated dependencies, but are requried for experiments within the VENLab.
# When experiment is ready to be moved to the VENLab, we'll re-include them.

import emergencyWalls

#####################################################################################
# Constants
# Set to True when ready to have the experiment write data
DATA_COLLECT = True

# If program should run practice trials
DO_PRACTICE = False

# If run free walk trials
DO_FREEWALK = False

# If program crashes, start trials here
START_ON_TRIAL = 13
# Total number of trials in experiment
FREEWALK_TRIALS = 4 # 4 trials each session
FREEWALK_SESSIONS = 2 # 1 session before practice 1 session after experiment
PRACTICE_TRIALS = 4
TOTAL_TRIALS = 60    # 1(d0) *  6(v0) * 10(reps) = 60
TRIAL_LENGTH = 12 # 12 seconds


# Used for file naming, currently placeholders
EXPERIMENT = 'Overtaking_v1.3_Tomato3'
NICKNAME = 'Tomato3'
EXPERIMENTER_NAME = 'jiuyangbai'

# Set output directory for writing data
OUTPUT_DIR = '/'.join(['Data', EXPERIMENTER_NAME, EXPERIMENT,(NICKNAME + '_Output')]) + '/'
INPUT_DIR = '/'.join(['Data', EXPERIMENTER_NAME, EXPERIMENT,(NICKNAME + '_Input')]) + '/'
SOUND_DIR = '/'.join(['sounds', EXPERIMENTER_NAME, EXPERIMENT]) + '/'
MODEL_DIR = 'Models/' + EXPERIMENTER_NAME + '/'
AVATAR_DIR = 'Avatars/'



# Orientation constants
POLE_TRIGGER_RADIUS = 0.3 # How close participant must be to home pole
THRESHOLD_THETA = 10 # Maximum angle participant can deviate when looking at orienting pole
ORIENT_TIME = 3 # How long participant must orient onto pole

# The dimension of the room space used for experiment
DIMENSION_X = 9.0 # the length of the shorter side in meter
DIMENSION_Z = 11.0 # the lengtth of the longer side in meter
DIAGONAL = (DIMENSION_X**2 + DIMENSION_Z**2)**(1.0/2)# The length of the diagonal line of the experimental space
ROOM_ANGLE = math.atan(DIMENSION_X/DIMENSION_Z) # the anger (in radian) between the diagonal and the shorter edge of the room
DIAGONAL_UNIT = [math.sin(ROOM_ANGLE), 0, math.cos(ROOM_ANGLE)]
# Home and Orient Pole positions (x,z,y)
HOME_POLE = [[DIMENSION_X/2, 0.0, DIMENSION_Z/2], [-DIMENSION_X/2, 0.0, -DIMENSION_Z/2]]
ORI_POLE = [[DIMENSION_X/2 - DIMENSION_X/3, 0.0, DIMENSION_Z/2 - DIMENSION_Z/3], \
			[-DIMENSION_X/2 + DIMENSION_X/3, 0.0, -DIMENSION_Z/2 + DIMENSION_Z/3]]


# Describe the end-trial trigger line (end line) in the intersense coordinate system
# the end line is perpendicular to walking direction
K = -DIMENSION_X/DIMENSION_Z # The slope of the end line
END_DIS = 2.0 # The distance between end line to home pole position of following trial
# The intercept of the end line for two home poles respectively
B = [-(DIAGONAL/2 - END_DIS) / math.cos(ROOM_ANGLE), (DIAGONAL/2 - END_DIS) / math.cos(ROOM_ANGLE)]

#####################################################################################
# Settings

IPD = viz.input('Please enter notes(IPD):')

# Dialog box asking for type of control and subject number
HMD = 'Odyssey'
MONITOR = 'PC Monitor'
controlOptions = [HMD,MONITOR]
controlType = controlOptions[viz.choose('How would you like to explore? ', controlOptions)]

subject = viz.input('Please enter the subject number:','')
subject = str(subject).zfill(2)



# Use keyboard controls
# Controls:
# q - Strafe L		w - Forward		e - Strafe R
# a - Turn L		s - Back		d - Turn R
#
# y - Face Up		r - Fly Up
# h - Face Down		f - Fly Down
if controlType == MONITOR:
	HZ = 60
	headTrack = viztracker.Keyboard6DOF()
	link = viz.link(headTrack, viz.MainView)
	headTrack.eyeheight(1.6)
	link.setEnabled(True)
	viz.go()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	
# Use HMD
elif controlType == HMD:
	HZ = 90
	# viz.fullscreen.x is the inital x position when run	
	# add Odyssey tracker
	ODTracker = steamvr.HMD().getSensor()

	# add the virtual tracker, link it to the MainView and set an offset to get the eye position
	link = viz.link(ODTracker, viz.MainView)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


viz.clip(.001,1000) # Adjust size of the viewing frustum
viz.go()


######################################################################################################
# Helper functions
def goToStage(nextTrialStage):
	global trial_stage
	trial_stage = nextTrialStage
	print 'Going to: ' + trial_stage

	
def endLine(x):
	# takes the x value and gives the z value of the end trigger line defined in the intersense coordinate system
	return x*K + B[trial_num%2]
	
def moveTarget(target, spd, isAvatar=False, stride=1.0, frequency=0.5/0.3):
	# unit of spd is m/s
	global time_elapsed, HZ
	pos = target.getPosition()
	dpos = spd * time_elapsed
	if trial_num%2 == 1: 
		# start from (-, 0, -) corner of the room	
		dz = float(dpos)*math.cos(ROOM_ANGLE)
		dx = float(dpos)*math.sin(ROOM_ANGLE)
	else:                
		# start from (+, 0, +) corner of the room
		dz = -float(dpos)*math.cos(ROOM_ANGLE)
		dx = -float(dpos)*math.sin(ROOM_ANGLE)
	if isAvatar:
		targetFrequncy = dpos/stride/time_elapsed
		target.speed(targetFrequncy*1.3/frequency + 0.2)
	
	target.setPosition([pos[0] + dx, 0, pos[2] + dz])
	

def countDown(t):
	global time, reset_countDown, time_stamp
	isTimeUp = False
	if reset_countDown:
		time_stamp = time
		reset_countDown = False
	if time - time_stamp > t:
		isTimeUp = True
	return isTimeUp
	
def writeCSVFile(fileName, data, time):
	strData = [str(round(t,4)) for t in data+[time]]
	with open(fileName, 'a') as file:
		file.write(','.join(strData)+'\n')
	
def relativeOrientation(pos1, pos2):
	xrel = round(pos2[0]-pos1[0],4)
	zrel = round(pos2[2]-pos1[2],4)
	theta = 0
	if zrel == 0.0 and xrel > 0:
		theta = math.pi/2
	elif zrel == 0.0:
		theta = math.pi/2*3
	else:
		theta = math.atan(round(xrel,4)/round(zrel,4))
		if zrel < 0:
			theta += math.pi
		if zrel > 0 and xrel < 0:
			theta += math.pi*2
	return theta
	
def facing(lookingObjectPosn, lookedAtObjectPosn, lookingObjectYaw, thresholdTheta):
	"""lookingObjectPosn: position of object that is looking
	lookedAtObjectPosn: position of object looked at
	lookingObjectYaw: yaw of the object that is looking (degrees)
	thresholdTheta: viewing angle must be +/- this amount in order to be considered 'looking at' the object. degrees

	return: bool, whether the looking object is facing the looked-at object
	>>> universals.facing([0,0,0],[1,0,5],0,20)
	True
	>>> universals.facing([3,0,3],[1,0,0],210,20)
	True
	"""
	degRelOrientation = 180.0/math.pi*relativeOrientation(lookingObjectPosn, lookedAtObjectPosn) #radians
	degRelOrientation = (degRelOrientation+180)%360-180
	return math.fabs(degRelOrientation-lookingObjectYaw)<thresholdTheta
	
def distance(x,y,a,b):
	return ((x-a)**2+(y-b)**2)**.5
	
def inRadius(pos, center, radius):
	#This method takes in two poadsitions in [x,y,z] form and then returns true if the distance between them is less than the radius given
	if pos == '' or center == '' or radius == '':
		return False
	return (distance(pos[0],pos[2],center[0],center[2]) <= radius)
	
def inRadiusAndFacing():
	global POLE_TRIGGER_RADIUS, THRESHOLD_THETA, models
	cur_pos = viz.get(viz.HEAD_POS)
	cur_rot = viz.get(viz.HEAD_ORI)
	return inRadius(cur_pos, models['homePole'].getPosition(), POLE_TRIGGER_RADIUS) and facing(cur_pos, models['orientPole'].getPosition(), cur_rot[0], THRESHOLD_THETA)

def projection(a, b):
	# take 2d vectors a and b, returns the length of the projection of a on b
	return (a[0] * b[0] + a[2] * b[2]) / (b[0]**2 + b[2]**2)**0.5
#######################################################################################################
# Experiment


# loads experimental conditions
inputFile = INPUT_DIR + NICKNAME + '_subject' + subject + '.csv'
with open(inputFile, 'rb') as data:
	rows = csv.reader(data)
	conditions = [row for row in rows]
condition = ''
# loads practice conditions
if conditions[1][3] == 'avatar':
	inputFile = INPUT_DIR + NICKNAME + '_practice_avatar.csv'
else:
	inputFile = INPUT_DIR + NICKNAME + '_practice_pole.csv'
with open(inputFile, 'rb') as data:
	rows = csv.reader(data)
	practice_conditions = [row for row in rows]
	
viz.clearcolor(0,0.4,1.0) # blue world
models = {}
models['homePole'] = viz.add(MODEL_DIR + 'pole_blue.osgb')
models['orientPole'] = viz.add(MODEL_DIR + 'pole_red.osgb')
models['targetPole'] = viz.add(MODEL_DIR + 'pole_green.osgb') # Consist of three green poles with their original size [0.4, 3, 0.4] 
models['leaderPole'] = viz.add(MODEL_DIR + 'pole_yellow.osgb')
models['ground'] = viz.add(MODEL_DIR + 'Tomato3_ground.osgb')


# load avatars such as "CC2_f001_hipoly_A0.cfg", "CC2_m001_hipoly_A0.cfg"
# The stride length of avatars are around 1 meter, frequency is 0.6 second per step, 5/3 step per second.
avatars = []
for i in range(20):
	avatars.append(viz.add(AVATAR_DIR + 'CC2_f' + str(i+1).zfill(3) + '_hipoly_A0.cfg'))
for i in range(20):
	avatars.append(viz.add(AVATAR_DIR + 'CC2_m' + str(i+1).zfill(3) + '_hipoly_A0.cfg'))		
for i in range(40):
	avatars[i].visible(viz.OFF)
	
# Adjust models size
models['homePole'].setScale([0.6,0.45,0.6]) # the original size = [0.4 3 0.4]
models['leaderPole'].setScale([0.5,0.6,0.5]) # the original size = [0.4 3 0.4]

# the transparency
_alpha = 0.0
# Hide loaded models
models['homePole'].visible(viz.OFF)
models['targetPole'].visible(viz.OFF)
models['leaderPole'].visible(viz.OFF)
models['orientPole'].visible(viz.OFF)

# Sounds
sounds={}
sounds['End'] = viz.addAudio(SOUND_DIR + 'End.mp3')
sounds['Begin'] = viz.addAudio(SOUND_DIR + 'Begin.mp3')


# Initial data_collection, regardless of DATA_COLLECT
data_collect = False

# Initializes trial_stage, which is the current step of a given trial
# First part is the name of the specific stage (pretrial)
# Second part is practice (01) or experimental trials (02)
# Third part is the stage's position in the order of all stages (01)
goToStage('pretrial_00_01')

# Initializa setting to start free walk trials
if DO_FREEWALK == True:
	is_freewalk = True
else:
	is_freewalk = False
	goToStage('pretrial_01_01')

# Initially setting to start practice trials
if DO_PRACTICE == True:
	is_practice = True
else:
	is_practice = False
	goToStage('pretrial_02_01')
	
# Starts trial count at 1, unless otherwise specifed
if START_ON_TRIAL > 1:
	trial_num = START_ON_TRIAL
	
else:
	trial_num = 1
freewalk_session = 1
# Time counter set to 0
time = 0

instruction = True
reset_countDown = True
screenshot = 1
data_batch = ''
leader = None
leaderSpd = 0
cur_pos = ''
flag = True
avatarID = -1
########################################################################################################################
# Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop #
########################################################################################################################
########################################################################################################################
# Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop #
########################################################################################################################
########################################################################################################################
# Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop #
########################################################################################################################
########################################################################################################################
# Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop #
########################################################################################################################
########################################################################################################################
# Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop ## Master Loop #
########################################################################################################################

def masterLoop(num):
	# global variables within masterLoop
	global DATA_COLLECT, DO_PRACTICE, is_practice, is_freewalk, data_collect, trial_stage, trial_num, \
	freewalk_session, time, time_stamp, cur_pos, conditions, practice_conditions, condition, K, B, _alpha, \
	reset_countDown, controlType,instruction, screenshot, data_batch, leaderSpd, leader, avatarID,\
	time_elapsed, HZ, flag

	# Time elapsed since the last run of masterLoop and then added to the global time
	time_elapsed = viz.getFrameElapsed()
	time += time_elapsed
	

	if os.path.isfile(OUTPUT_DIR + 'image'+ str(screenshot) +'.bmp') == True:
		screenshot += 1
		
	vizact.onkeydown('p', viz.window.screenCapture, OUTPUT_DIR + 'image'+ str(screenshot) +'.bmp')
	
	# Current position and roation of the participant
	cur_pos = viz.get(viz.HEAD_POS)
	cur_rot = viz.get(viz.HEAD_ORI)
	#>> Will only work in VENLab <<
	emergencyWalls.popWalls(cur_pos) # Pops up the Emergency Walls when participant is close to physical room edge.

	
	##################
	# Begin Freewalk #
	##################
	
	if is_freewalk == True:
		
		
		# Writes Position and Rotation, but only when DATA_COLLECT set to True
		# and trial_stage has set data_collect to True
		if DATA_COLLECT and data_collect:			

			# Position: Target_x, Target_y, Target_z, Participant_x, Participant_y, Participant_z, Yaw, Pitch, Row, time stamp
			data = [cur_pos[0], cur_pos[1], cur_pos[2], cur_rot[0], cur_rot[1], cur_rot[2]]
			strData = [str(round(t,4)) for t in data+[time]]
			strData = ','.join(strData)+'\n'
			data_batch = data_batch + strData

		
		
		#########
		# 00 01 Freewalk Pretrial: sets up practice trial, establishes pole locations
		if trial_stage == 'pretrial_00_01':
			if flag:
				print '> Start Free Walking Session ' + str(freewalk_session) + ' Trial ' + str(trial_num) + ' ----------------------------------'
				flag = False
		
			# Set position of home pole (where participant stands to start trial)
			if models['homePole'].getVisible() == False:
				models['homePole'].setPosition(HOME_POLE[trial_num%2])
				models['homePole'].alpha(1.0)
				models['homePole'].visible(viz.ON)

			# Set position of orientation pole (where participant faces to start trial)
			if models['orientPole'].getVisible() == False:
				models['orientPole'].setPosition(HOME_POLE[(trial_num+1)%2])
				models['orientPole'].visible(viz.ON)
			
			if trial_num == 1 and freewalk_session == 1 and instruction:
				# placeholder for instructions
				instruction = False
			
			if not(trial_num == 1 and freewalk_session == 1) or countDown(5):
				# Move to next stage
				goToStage('orient_00_02')
				instruction = True
				reset_countDown = True				
				
		#########
		# 00 02 Orienting to Pole: Give time for participant to orient to the pole
		elif trial_stage == 'orient_00_02':
			flag = True
			if inRadiusAndFacing():
				if trial_num == 1 and instruction:					
					if freewalk_session == 2:
						pass # placeholder for instructions
					instruction = False
				if not(trial_num == 1) or (freewalk_session == 1) or (freewalk_session == 2 and countDown(5)):	
					# Move to stage 3
					goToStage('orient_00_02_wait')
					instruction = True
					reset_countDown = True
	
				
		#########
		# wait for orientation
		elif (trial_stage == 'orient_00_02_wait'):			
			if countDown(ORIENT_TIME):
				goToStage('inposition_00_03')
				reset_countDown = True
			if not inRadiusAndFacing():
				reset_countDown = True
		
		#########
		# 00 03 Freewalk In Position: proceeds once participant is standing on home and facing orient for three seconds
		elif (trial_stage == 'inposition_00_03'):
			print 'Free walk start'
			# Turn off home pole
			models['homePole'].visible(viz.OFF)
			models['orientPole'].visible(viz.OFF)
			models['targetPole'].setPosition(HOME_POLE[(trial_num+1)%2])
			models['targetPole'].visible(viz.ON)
			sounds['Begin'].play()
			
	
			# Start to collect data
			data_collect = True
		
			# initialize batch data output
			data_batch = ''
			time = 0
			
			# Move to Stage 4
			goToStage('target_00_04')
			
		#########
		# 00 04 Freewalk: Participants Moves
		elif (trial_stage == 'target_00_04'):
				
			# Detects participant location, moves to Stage 5 (Ends Trial) when participant reache the end line
			if (trial_num%2 == 1) and (cur_pos[2] > endLine(cur_pos[0])) or \
				(trial_num%2 == 0)and (cur_pos[2] < endLine(cur_pos[0])):
					
				goToStage('endtrial_00_05')
		
		#########
		# 00 05 End Freewalk Trial: Close out the trial and reset values for next practice trial or start Experiment
		elif trial_stage == 'endtrial_00_05':
		
			# Clears the target pole
			models['targetPole'].visible(viz.OFF)
			
			# save the data of this trial
			fileName = OUTPUT_DIR + NICKNAME + '_freewalk' + '_subj' + subject + '_s' + str(freewalk_session) + '_trial' + str(trial_num).zfill(3) + '.csv'
			with open(fileName, 'a') as file:
				file.write(data_batch)


			print 'End Freewalk Trial ' + str(trial_num)
			data_collect = False


			# End Check: When trial_num is greater than FREEWALK_TRIALS, end practice and start experiment block
			if trial_num == FREEWALK_TRIALS:
				print '>> End Freewalk Session<<'
				if freewalk_session == 2:
					print '>>> End Experiment <<<'
					goToStage('NULL')
					if instruction:
						sounds['End'].play()
						instruction = False
				elif freewalk_session == 1:
					goToStage('pretrial_01_01')
					is_freewalk = False
					trial_num = 1
					freewalk_session += 1
			# Returns to Stage 1, resets clock
			else:
				trial_num += 1
				goToStage('pretrial_00_01')
				instruction = True
		

	##################
	# Begin practice #
	##################
	
	elif is_practice == True:
		
	
		#########
		# 01 01 Practice Pretrial: sets up practice trial, establishes pole locations
		if trial_stage == 'pretrial_01_01':
			if flag:
				print '> Start Practice Trial ' + str(trial_num) + ' ----------------------------------'
				flag = False
			# load input
			if practice_conditions[trial_num][3] == 'avatar':
				avatarID = random.choice(range(len(avatars)))
				leader = avatars[avatarID]
				leader.state(5)
			else:
				leader = models['leaderPole']
				
			leaderSpd = 0
			# play instruction
			if trial_num == 1 and instruction:
				# placeholder for instructions
				instruction = False
				
			# Move to Stage 2
			if trial_num != 1 or countDown(5):
				goToStage('orient_01_02')
					
			
		#########
		# 01 02 Orienting to Pole: Give time for participant to orient to the pole
		elif (trial_stage == 'orient_01_02'):
			flag = True
			# Set position of home pole (where participant stands to start trial)
			if models['homePole'].getVisible() == False:
				models['homePole'].setPosition(HOME_POLE[trial_num%2])

			if _alpha < 1.0:
				models['homePole'].alpha(_alpha)
				models['homePole'].visible(viz.ON)
				_alpha += 1.0/HZ
	
			# Set position of orientation pole (where participant faces to start trial)
			if models['orientPole'].getVisible() == False:
				models['orientPole'].setPosition(HOME_POLE[(trial_num+1)%2])
				models['orientPole'].visible(viz.ON)
	
				
			if inRadiusAndFacing():					
				reset_countDown = True
				# Move to stage 3
				goToStage('orient_01_02_wait')
				instruction = True
				_alpha = 0.0
				# Current time
				time_stamp = time
				
		#########
		# wait for orientation
		elif (trial_stage == 'orient_01_02_wait'):
			if countDown(ORIENT_TIME):
				goToStage('inposition_01_03')
				reset_countDown = True
			if not inRadiusAndFacing():
				reset_countDown = True
				
				
		#########
		# 01 03 Practice In Position: proceeds once participant is standing on home and facing orient for three seconds
		elif trial_stage == 'inposition_01_03':
			
			print 'Practice Target Appears'
			
			# Turn off home pole and orientation pole
			models['homePole'].visible(viz.OFF)
			models['orientPole'].visible(viz.OFF)
			models['targetPole'].setPosition(HOME_POLE[(trial_num+1)%2])
			models['targetPole'].visible(viz.ON)
			sounds['Begin'].play()
			
			# Move to Stage 4
			goToStage('target_01_04')
			time = 0
			
		#########
		# 01 04 Moving leader: Leader Moves
		elif (trial_stage == 'target_01_04'):
			leaderOnset = float(practice_conditions[trial_num][4])
			d0 = float(practice_conditions[trial_num][1])
			v0 = float(practice_conditions[trial_num][2])
			if time > leaderOnset:
				if not leader.getVisible():
					models['targetPole'].visible(viz.OFF)
					# leader appear d0 meter in front of participant
					leader.setPosition(HOME_POLE[trial_num%2])
					leader.lookAt(HOME_POLE[(trial_num+1)%2], mode=viz.ABS_GLOBAL)
					home = models['homePole'].getPosition()						
					d = d0 + projection([x-y for x, y in zip(cur_pos, home)], [-x for x in home])
					moveTarget(leader, d/time_elapsed, conditions[trial_num][3] == 'avatar') # move the leader by d0 from homepole to leader					
					leader.visible(viz.ON)
				# leader moves
				moveTarget(leader, v0, practice_conditions[trial_num][3] == 'avatar')
			
			# Detects participant location, moves to Stage 6 (Ends Trial) when participant reache the end line
			if (trial_num%2 == 1) and (cur_pos[2] > endLine(cur_pos[0])) or \
				(trial_num%2 == 0)and (cur_pos[2] < endLine(cur_pos[0])):
				goToStage('endtrial_01_05')
		
		#########
		# 01 05 End Practice Trial: Close out the trial and reset values for next practice trial or start Experiment
		elif trial_stage == 'endtrial_01_05':
			
			# Clears the target pole
			leader.visible(viz.OFF)

			print 'End Practice Trial ' + str(trial_num)
			

			# End Check: When trial_num is greater than PRACTICE_TRIALS, end practice and start experiment block
			if trial_num >= PRACTICE_TRIALS:
				print '>> End Practice <<'
				goToStage('pretrial_02_01')
				is_practice = False
				trial_num = 1
			# Returns to Stage 1, resets clock
			else:
				trial_num += 1
				goToStage('pretrial_01_01')
			
	####################
	# Begin Experiment #
	####################
	
	elif is_practice == False:
		

		# Writes Position and Rotation, but only when DATA_COLLECT set to True
		# and trial_stage has set data_collect to True
		if DATA_COLLECT and data_collect:
			
			
			# Location of Target Pole
			leader_loc = leader.getPosition()
			
			leader_loc = leader.getPosition()
			# Position: Target_x, Target_y, Target_z, Participant_x, Participant_y, Participant_z, Yaw, Pitch, Row, time stamp
			data = [leader_loc[0], leader_loc[1], leader_loc[2], cur_pos[0], cur_pos[1], cur_pos[2], cur_rot[0], cur_rot[1], cur_rot[2]]
			strData = [str(round(t,4)) for t in data+[time]] + [str(avatarID)]
			strData = ','.join(strData)+'\n'
			data_batch = data_batch + strData

			# log IPD
			if trial_num == 1:
				file = open(OUTPUT_DIR + NICKNAME + '_subj' + subject + \
				'_IPD_' + str(IPD) + '.txt', 'a')
				file.close()
			
			
		#########
		# 02 01 Experiment Pretrial: sets up trial, establishes pole locations
		if trial_stage == 'pretrial_02_01':
			
			condition = ', '.join(conditions[trial_num][1:4])
			
			# Print start of trial, trial #, and type of trial [pos][speed][turn]
			if flag:
				print '> Start Trial ' + str(trial_num) + ': ' + condition + ' ----------------------------------'
				flag = False
			# load input
			if conditions[trial_num][3] == 'avatar':
				avatarID = random.choice(range(len(avatars)))
				leader = avatars[avatarID]
				leader.state(5)
			elif conditions[trial_num][3] == 'pole':
				leader = models['leaderPole']
						
			leaderSpd = 0
			
			# play instruction
			if trial_num == 1 and instruction:				
				instruction = False
			
			# Move to Stage 2			
			if trial_num != 1 or countDown(5):
				goToStage('orient_02_02')
				
				
		#########
		# 02 02 Orienting to Pole: Give time for participant to orient to the pole
		elif (trial_stage == 'orient_02_02'):
			flag = True
			## Set position of home pole (where participant stands to start trial)
			if models['homePole'].getVisible() == False:
				models['homePole'].setPosition(HOME_POLE[trial_num%2])
				
			if _alpha < 1.0:
				models['homePole'].alpha(_alpha)
				models['homePole'].visible(viz.ON)
				_alpha += 1.0/HZ
	
			# Set position of orient pole (where participant faces to start trial)
			if models['orientPole'].getVisible() == False:
				models['orientPole'].setPosition(HOME_POLE[(trial_num+1)%2])
				models['orientPole'].visible(viz.ON)

			if inRadiusAndFacing():					
				# Move to stage 3
				goToStage('orient_02_02_wait')
				_alpha = 0.0
				instruction = True
				reset_countDown = True				
	
		#########
		# wait for orientation
		elif (trial_stage == 'orient_02_02_wait'):
			if countDown(ORIENT_TIME):
				goToStage('inposition_02_03')
				reset_countDown = True
			if not inRadiusAndFacing():
				reset_countDown = True		
		
		#########
		# 02 03 In Position: proceeds once participant is standing on home and facing orient
		elif trial_stage == 'inposition_02_03':

			# Turn off home and orient poles
			models['homePole'].visible(viz.OFF)
			models['orientPole'].visible(viz.OFF)
			models['targetPole'].setPosition(HOME_POLE[(trial_num+1)%2])
			models['targetPole'].visible(viz.ON)
			sounds['Begin'].play()
			
			# Turn on data collection for this trial
			data_collect = True
				
			# Move to Stage 5
			goToStage('target_02_04')
			# initialize batch data output
			data_batch = ''
			time = 0

			
		#########
		# 02 04 Moving Leader: Leader Moves
		elif (trial_stage == 'target_02_04'):
			# read data
			leaderOnset = float(conditions[trial_num][4])
			d0 = float(conditions[trial_num][1])
			v0 = float(conditions[trial_num][2])
			if time > leaderOnset:
				if not leader.getVisible():
					models['targetPole'].visible(viz.OFF)
					# leader appear d0 meter in front of participant
					leader.setPosition(HOME_POLE[trial_num%2])
					leader.lookAt(HOME_POLE[(trial_num+1)%2], mode=viz.ABS_GLOBAL)
					home = models['homePole'].getPosition()
					d = d0 + projection([x-y for x, y in zip(cur_pos, home)], [-x for x in home])
					moveTarget(leader, d/time_elapsed, conditions[trial_num][3] == 'avatar') # move the leader by d0 from homepole to leader					
					leader.visible(viz.ON)
				# leader moves
				moveTarget(leader, v0, conditions[trial_num][3] == 'avatar')
				
				
			# Detects participant location, moves to Stage 6 (Ends Trial) when participant reache the end line
			if (trial_num%2 == 1) and (cur_pos[2] > endLine(cur_pos[0])) or \
				(trial_num%2 == 0)and (cur_pos[2] < endLine(cur_pos[0])):
				goToStage('endtrial_02_05')
				
			#########
		# 02 05 End Trial: Close out the trial and reset values for next trial
		elif trial_stage == 'endtrial_02_05':
			
			# Clears the target pole
			leader.visible(viz.OFF)

			# End data collection for this trial
			data_collect = False
			
			# save the data of this trial			
			fileName = OUTPUT_DIR + NICKNAME + '_subj' + subject + '_trial' + str(trial_num).zfill(3) + '_' + condition + '.csv'
			with open(fileName, 'a') as file:
				file.write(data_batch)
	
			print 'End Trial ' + str(trial_num)
			
			# When trial_num is greater than TOTAL_TRIALS, end experiment
			if trial_num == TOTAL_TRIALS:
				is_freewalk = True
				goToStage('pretrial_00_01')
				trial_num = 1
			# Returns to Stage 1, resets clock
			else:
				trial_num += 1
				goToStage('pretrial_02_01')

# Restarts the loop, at a rate of 60Hz
viz.callback(viz.TIMER_EVENT,masterLoop)
viz.starttimer(0,1.0/HZ,viz.FOREVER)

