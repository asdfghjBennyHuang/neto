import numpy as np
import lasagne
import pickle
import sys
import os
import shutil

###############

default_net_file = 'network.net'
default_func_file = 'funcs.pic'

###############


# pickle and cPickle somtimes exceed the default recurssion limit
# Too much recurssion would cause a lot usage of memory, but the python community decide not to have tail recursion
sys.setrecursionlimit(50000)

def get_result_tofola( path, network, recompile, direct_recompile=False ):
	
	load_network(network, os.path.join( path, default_net_file ) )

	if direct_recompile:
		re_t = recompile(network)
	
	# else try to load from file, recompile if the loading fail
	else:
		try:
			re_t = load_compiled_fn( os.path.join( path, default_func_file ))

		except FileNotFoundError:
			print( 'no compiled function found, compiling' )
			re_t = recompile(network)

		except EOFError:
			print( 'File found does not store the same amount of funtion as specified. Recompiling' )
			re_t = recompile(network)
	
	assert isinstance( re_t, tuple )
		
	print('Theano function finished')
	return re_t

def store_result_tofola( path, network, func_list, log_files='', rewrite=False ):

	'''
	This is a more general function to store the whole stuff, 
	we hope a set of the compiled function and trained network parameters can be stored in the same directory
	maybe also include the log file
	For better training result management
	'''

	# prepare the path needed for storing
	if os.path.isdir( path ):
		if not rewrite:
			inx = 0;
			while True:
				new_path = path + '_' + str(inx)
				if not os.path.isdir( new_path ):
					os.mkdir( new_path )
					
					# list all the file in the current directory,
					# and then move them into the new created folder which has a number at the end
					allfiles = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
					[ os.rename( os.path.join(path, f), os.path.join( new_path, f) ) for f in allfiles ]
					break
				else:
					inx += 1
	else:
		os.makedirs( path )

	# handle all the log files, first check if it's a list
	if log_files != '':
		try:
			_ = (i for i in log_files)
		except TypeError:
			log_files = [log_files]

		for the_file in log_files:
			if os.path.isfile( the_file ):
				# avoid OSError by limiting user's input cannot be in another folder
				if '/' not in the_file and '\\' not in the_file:
					shutil.copyfile( the_file, os.path.join( path, the_file ) )
				else:
					print( 'store_result_tofola not support move log file from another directory' )

	store_network(network, os.path.join( path, default_net_file ) )
	store_compiled_fn(func_list, os.path.join( path, default_func_file ) )
	
	print( 'store_result_tofola storing finished' )

def store_compiled_fn(fns, filename, old_dir='old_file'):
	'''
	fns : the list of funciton that are intended to stored into the file
	'''
	
	# if the filename exist, move it into the old dir, lest we delete something important
	store_to_old( filename, old_dir )

	# make sure the fns is a list of function
	fns = [fns] if not isinstance( fns, list ) else fns

	assert isinstance(filename, str)
	
	try:
		with open(filename, 'wb') as fobj:
			pickle.dump(fns, fobj, protocol=pickle.HIGHEST_PROTOCOL)
	
	except RecursionError:
		print( "RecursionError, retrying" )
		sys.setrecursionlimit(100000)
		
		# try again with higher limit.
		with open(filename, 'wb') as fobj:
			pickle.dump(fns, fobj, protocol=pickle.HIGHEST_PROTOCOL)

def load_compiled_fn(filename):
	'''
	num : the number of the function that are intended to be extracted from the file.
	Since there should be equivalent object waiting to unpack the returned tuple, the num should be known before executing.
	
	all of the error handling is passed to retrieback functions
	
	the filename is usually with .pic extension, meaning to be stored with pickle.
	'''
	
	if not filename.endswith( '.pic' ):
		print( 'please check the compiled function file name.' )
	
	with open(filename, 'rb') as fobj:
		ret_list = pickle.load(fobj)

	# should be ensured when storing the function
	assert isinstance(ret_list, list)

	return tuple(ret_list)

def store_network(net, filename, old_dir='old_file'):

	# if the filename exist, move it into the old dir, lest we delete something important
	store_to_old( filename, old_dir )
			
	val = lasagne.layers.get_all_param_values(net)

	with open(filename, 'wb') as file:
		pickle.dump(val, file, protocol=pickle.HIGHEST_PROTOCOL)
		
def load_network(net, filename):

	'''
	Load the network from file. 
	Usually the file with .net extension
	'''
	print(filename)
	if not filename.endswith( '.net' ):
		print( 'please check the network file name.' )

	try:
		with open(filename, 'rb') as file:
			val = pickle.load(file)
	
		lasagne.layers.set_all_param_values(net, val)
	
	except FileNotFoundError:
		print('no network found. Retrain')
	except Exception as err:
		print(type(err))
		print(err)
		print('loading network fail.')
	
def store_to_old( filename, old_dir ):
	
	'''
	if the filename exist, add a number to it end, keep doing this until there is not conflict
	return an integer represending the file name it finally get. 
	'''
	
	if os.path.isfile( filename ):
		# make the old directoy if not exist
		if not os.path.isdir( old_dir ):
			os.mkdir( old_dir )
			
		if '/' in filename and '\\' in filename:
			raise Exception( 'WTF happen to the filename?!' )
			
		# make the directoy for filename if not exist
		if '/' in filename:
			os.makedirs( '/'.join(filename.split('/')[0:-1]) )
		if '\\' in filename:
			os.makedirs( '/'.join(filename.split('\\')[0:-1]) )
			
		# if the file already exists in the old dir, extend its name 
		i = 0
		
		name = filename.split('.')[0]
		ext = filename.split('.')[1]
		
		while True:
			nfilename = name + '_' + str(i) + '.' + ext 
			if not os.path.isfile( os.path.join( old_dir, nfilename ) ):
				os.rename( filename, os.path.join( old_dir, nfilename ) )
				return i

			i += 1
	
	else:
		return 0
	
def retrieback(filename, network, num, recompile):

	'''
	
	This is for convienence and can reduce the code in the rnn files.
	
	network is the lasagne network object that the parameter are giing to load into.
	num is the number of functions expected to be returned.
	recompile should be a theano function that is going to compile the expected functions if the file is not fuond.
	'''

	print('try to load a network')
	try:
		load_network( network, filename+'.net' )
		
	except FileNotFoundError:
		print( 'no trained model found, train from scratch' )
		
	print('network finished')
	print('try to load compiled functions')

	# if num is specified to -1, directly recompile
	if num == -1:
		re_t = recompile(network)
	# else try to load from file, recompile if the loading fail
	else:
		try:
			re_t = load_compiled_fn( num, filename+'.pic')
		
		except FileNotFoundError:
			print( 'no compiled function found, compiling' )
			re_t = recompile(network)

		except EOFError:
			print( 'File found does not store the same amount of funtion as specified. Recompiling' )
			re_t = recompile(network)
		
	print('Theano function finished')
		
	return re_t
	
