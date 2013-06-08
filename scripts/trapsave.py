#######################################################################
#
# Program:      trapsave.py: Updated HDF 5 File Handling for trap simulations
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      January 29, 2013
#
# Changelog:    traphdf5.py
#               Dec 4, 2012: Created from twodtrap.py
#               
#           --->trapsave.py
#               Jan 29, 2013: forked from traphdf5.py
#                             Changed save file routine, now savenames
#                             come from datetime package
#                             Runs database is now a HDF-5 database
#                             A script converts this to a text database
#                             This allows for run paramaters to be added and
#                             removed.
#
# Purpose:      Contain functions that do most of the hdf5 file handling used
#               in routines
#               
########################################################################

#Imports
from __future__ import division

import h5py
import datetime as dt

def autof(h              ,
          folder = 'sims'):
  '''
  Automatically get file database, make a filename, and add headers to a 
  database file
  SYNTAX autof(h              ,
               folder = 'sims'): #option to change the directory for
                                 #different simulation types
  
  
  headers are specified by h
  '''
  t = dt.datetime.today()
  filename = (folder + "/run%s.hdf5" % t.isoformat(), 'r+' ) 
  
  db = h5py.File( folder + "db.hdf5" )
  fn = db.create_group( t.isoformat )  
  for entry in h:
    fn.create_dataset( entry, h[entry] )
  
  return filename
  
########################################################################

def createh5( name          ,
              erase = False ):
  '''
  Open a hdf5 file, checking that nothing is accidentally overwritten
  '''
  if erase == False:
    #open file, check if we can overwrite previous files
    try:
        f = h5py.File( name, 'w-' )
    except:
      print 'File exists already'
      print 'add kwarg erase = True to overwrite'
      return SystemError('Aborting so you do not lose precious data')
      
  #Overwrite existing file if erase = True
  elif erase == True: f = h5py.File( name, 'w' )
    
  return f
  
########################################################################

def readh5( name ):
  '''
  Open a hdf5 file for reading
  '''
  try:
    f = h5py.File( str(name) , 'r')
  except:
    print "It seems the file you tried to open doesn't exist."
    print "sorry about that :("
  
  headers = dict()
  
  print "Succesfully opened" + str(name)
  print "File headers are as follows"
  
  for name, value in f.attrs.iteritems():
    print name+":", value
    headers[ name ] = value
  
  print 'file headers are stored in the dictionary h5file.head'
  
  return f, headers
  
########################################################################
  
def fileinfo(filename):
  '''
  Print out header data for a HDF5 file
  '''
  print "Header data for " + str(filename)
  try: infile = h5file( filename, erase = False, read = True )
  except: return SystemError(str(filename) + " is not a valid hdf5 file")
  infile.f.close()
  print str(filename) + ' closed'
  
########################################################################
  
class h5file:
  '''
  The overarching HDF5 file class for handling these files
  USAGE: 
    a = h5file( filename    ,    #filepath to open
                erase       ,    #overwrite existing file?
                read = False ):  #reading in a file instead? Then True
  METHODS:
    self.add_data
    self.add_headers
    self.readpsi
    self.readxy
  Each method has its own documentation.
  '''
  def __init__( self        ,
                filename    ,    #filepath to open
                erase       ,    #overwrite existing file?
                read = False ):  #reading in a file instead? Then True
    self.name   = filename
    self.erase  = erase
    
    if read == False: 
      self.f = createh5( self.name, erase = erase )
    elif read == True: self.f, self.head = readh5( filename )
  
  def add_data(self, runname, xdata, ydata, psi, time):
    
    grp   = self.f.create_group( str(runname) )
    grp.attrs['time'] = time
    
    xvals = grp.create_dataset( 'xvals', data = xdata )
    yvals = grp.create_dataset( 'yvals', data = ydata )
    psi   = grp.create_dataset( 'psi'  , data = psi   )
    
    
  def add_headers(self, head_dict):
    '''
    Add headers to the main file, to explain relevant parameters
    SYNTAX: h5file.add_headers(head_dict):
    head_dict should be a dictionary
    '''
    
    for name, value in head_dict.iteritems():
      self.f.attrs[ str( name ) ] = value
  
  def readxy(self):
    '''
    Will return the grid for the chosen run
    x,y = self.readxy() 
    '''
    return self.f.get('0.0/xvals')[:], self.f.get('0.0/yvals')[:]
    
  def readpsi(self, i):
    '''
    Will return a specified psi
    p = self.readpsi('frame')
    frame is an integer (within a string) in the form 0.0, 1.0, etc.
    '''
    return self.f.get( str(i) + '/psi')[:]

################################################################################

def dbconvert( infilename, outfilename ):
  '''
  dbconvert( infilename, outfilename ):
  
  Convert HDF5 database of runs to a text file
  The database is made by the autof routine.
  '''
  
  db = h5py.File( infilename, 'r' )
  txt = file( outfilename, 'w' )
  
  timegroups = [ a[ time[0] ] for time in a.items() ]
  
  keylists = [ timekeys.items() for timekeys in timegroups ] #list all the keys
  
  keys = [str( item[0] ) for sublist in keylists for item in sublist]
                #flatten the keylist
  ukeys = list( set(keys) ) #unique keys
  ukeys.sort()
  
  ukeys.remove( 'g' ); ukeys.remove( 'G' )
  ukeys.insert( 0, 'g' ); ukeys.insert( 0, 'G' ); ukeys.insert( 0, 'filename' )
  
  txt.write(','.join( ukeys ) ) #write headings
  
  for group in timegroups:
    gp = db[ group ]
    vals = [ group ] #first line is date
    for key in ukeys:
      try:
        vals.append( gp[key].value )
      except:
        vals.append( '-' )
    txt.write(','.join( vals ) ) #write data
      
  db.close()
  txt.close()