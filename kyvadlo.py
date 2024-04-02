# Import numpy
import numpy as np
import xml.etree.ElementTree as ET


#123456789
a = 1010
print (a)

treeDiffVar = ET.parse('measurements_nooutlier_differentvar.xml')
DataDiffVar = tree.getroot()
treeSameVar = ET.parse('measurements_nooutlier_differentvar.xml')
DataSameVar = tree.getroot()
treeOutlier = ET.parse('measurements_nooutlier_differentvar.xml')
DataOutlier = tree.getroot()
