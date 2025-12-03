# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'TIFF Series Reader'
jM02_3D_SIRT_CUDA_00tiff = TIFFSeriesReader(registrationName='JM02_3D_SIRT_CUDA_0.0.tiff', FileNames=['C:\\Users\\b304014\\Box\\zhihua\\02\\tomogram_alignment_recon_cropped_14nm.tif'])

# set active source
SetActiveSource(jM02_3D_SIRT_CUDA_00tiff)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
jM02_3D_SIRT_CUDA_00tiffDisplay = Show(jM02_3D_SIRT_CUDA_00tiff, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
jM02_3D_SIRT_CUDA_00tiffDisplay.Representation = 'Outline'

# get the material library
materialLibrary1 = GetMaterialLibrary()

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# set scalar coloring
ColorBy(jM02_3D_SIRT_CUDA_00tiffDisplay, ('POINTS', 'Tiff Scalars'))

# rescale color and/or opacity maps used to include current data range
jM02_3D_SIRT_CUDA_00tiffDisplay.RescaleTransferFunctionToDataRange(True, True)

# change representation type
jM02_3D_SIRT_CUDA_00tiffDisplay.SetRepresentationType('Volume')

# rescale color and/or opacity maps used to include current data range
jM02_3D_SIRT_CUDA_00tiffDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
jM02_3D_SIRT_CUDA_00tiffDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'TiffScalars'
tiffScalarsLUT = GetColorTransferFunction('TiffScalars')

# get opacity transfer function/opacity map for 'TiffScalars'
tiffScalarsPWF = GetOpacityTransferFunction('TiffScalars')

# get 2D transfer function for 'TiffScalars'
tiffScalarsTF2D = GetTransferFunction2D('TiffScalars')

# Rescale transfer function
tiffScalarsLUT.RescaleTransferFunction(0.0008, 0.003)

# Rescale transfer function
tiffScalarsPWF.RescaleTransferFunction(0.0008, 0.003)

# Rescale 2D transfer function
tiffScalarsTF2D.RescaleTransferFunction(0.0008, 0.003, 0.0, 1.0)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(622, 687)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [2139.234787199519, 1958.3697499389343, 148.97953357383093]
renderView1.CameraFocalPoint = [443.5000000000007, 443.5000000000003, 165.4999999999997]
renderView1.CameraViewUp = [0.025592788608141192, -0.039541823635178205, -0.9988901107503585]
renderView1.CameraParallelScale = 716.4587445024142


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------